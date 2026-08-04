"""NLA actor inference via SGLang input_embeds — single-file, no nla package deps.

An NLA (Natural Language Autoencoder) pair is two fine-tuned LMs that together
map activation vectors to natural language and back:

  ACTOR  (activation verbalizer)  : hidden-state vector  →  text
                                    [inject vector as a 1-token embedding
                                     into a fixed prompt, then autoregress]

  CRITIC (activation reconstructor): text  →  hidden-state vector
                                    [truncated K+1-layer LM + Linear(d,d)
                                     head, extract at final token]

The round-trip — extract → ACTOR verbalizes → CRITIC reconstructs → MSE against
original — measures how well the verbalization captured the vector's content.
That MSE was the RL reward signal during actor training: low MSE means the
critic can recover the original direction from the actor's words alone.

This file contains both halves:
  NLAClient  — actor inference via SGLang input_embeds
  NLACritic  — load critic + reconstruct + score (optional, pure torch)

Ship alongside HF-format NLA actor + critic checkpoint dirs (each with
config.json, safetensors, tokenizer files, nla_meta.yaml).

Dependencies:
    uv pip install torch transformers safetensors httpx orjson pyyaml numpy
    uv pip install "sglang[all]>=0.5.6"   # tested against 0.5.6; input_embeds
                                          # API + --disable-radix-cache verified there
    # Optional (for --parquet CLI): uv pip install pyarrow

─────────────────────────────────────────────────────────────────────────────
Launch SGLang first (same checkpoint path as NLAClient below):

    python -m sglang.launch_server \\
        --model-path ./actor_hf \\
        --port 30000 \\
        --disable-radix-cache \\
        --mem-fraction-static 0.85 \\
        --trust-remote-code

  --disable-radix-cache is REQUIRED. Radix cache keys on token IDs;
  input_embeds requests have none, so different embed sequences alias
  to the same cache entry → silent garbage.

  Gemma-3 needs HF_TOKEN set (gated repo) and benefits from
  --context-length 512 (NLA prompts are short; default 8k wastes KV cache).

  Stock sglang>=0.5.6 works out of the box. For high-throughput use (>10
  req/s), sglang's input_embeds path has a FastAPI-validation bottleneck
  (~155ms/req on the event loop, caps concurrency at ~2) and a long-running
  server may hit a retract-path KV-slot crash. Upstream PRs open:
    sgl-project/sglang#20205 (numpy IPC), #20206 (SkipValidation),
    #20207 (bytes+shape transport — raw fp32 buffer, bypasses JSON),
    #20376 (slice input_embeds on chunk-overflow — correctness, pull this in),
    #14110 (retract fix).
  The bytes+shape transport is the interesting one if you're scanning large
  feature dictionaries — you may find JSON serialization the bottleneck.
  None change the wire API; this client code works unchanged.
─────────────────────────────────────────────────────────────────────────────

Usage:
    client = NLAClient("./actor_hf", sglang_url="http://localhost:30000")
    text = client.generate(activation_vector)    # activation: [d_model] array

    # or batched — one SGLang request per vector; SGLang's continuous batcher
    # packs them server-side. For real throughput, fire in parallel via
    # async httpx.
    texts = client.generate_batch(vectors, temperature=0.7)

    # custom prompt (must contain <INJECT> where the vector goes):
    text = client.generate(v, prompt="What is: <concept><INJECT></concept>?")
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ─── Constants ──────────────────────────────────────────────────────────────

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
INJECT_PLACEHOLDER = "<INJECT>"
# Embedding weight key suffixes across HF architectures (Llama/Qwen/Mistral/
# Gemma use embed_tokens; GPT-2 uses wte; Falcon uses word_embeddings).
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


# ─── Sidecar config ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    # L2-norm the vector gets rescaled to before injection. MANDATORY — the
    # model learned with this exact scale; raw-magnitude vectors are OOD.
    # Qwen7B: 150. Gemma-3-12B: 80000 (√d embed scaling inflates residual norms).
    injection_scale: float


def load_nla_config(
    checkpoint_dir: str | Path,
    tokenizer: Any,
    injection_scale_override: float | None = None,
) -> NLAConfig:
    """Parse {checkpoint_dir}/nla_meta.yaml and assert against live tokenizer.

    Catches the two most common silent-failure modes BEFORE the first request:
      - tokenizer version drift → injection char tokenizes differently
      - prompt template drift → neighbors no longer match
    Both produce CJK-flavoured output if not caught (the marker char's own
    embedding gets verbalized as the activation).
    """
    meta_path = Path(checkpoint_dir) / "nla_meta.yaml"
    assert meta_path.exists(), (
        f"no nla_meta.yaml at {checkpoint_dir!r}. Not an NLA checkpoint — "
        f"the sidecar ships alongside config.json/safetensors. If you "
        f"received a checkpoint without it, ask the provider for the sidecar."
    )
    meta = yaml.safe_load(meta_path.read_text())

    kind = meta["kind"]
    assert kind in ("nla_model", "nla_dataset"), f"unknown sidecar kind: {kind!r}"
    d_model = meta["d_model"] if kind == "nla_model" else meta["extraction"]["d_model"]

    # injection_scale: MANDATORY (unless override passed). Actor sidecars have
    # it; critic sidecars have null (critics don't inject). Dataset sidecars
    # deliberately don't — it's a training hyperparameter.
    inj_scale = meta.get("extraction", {}).get("injection_scale")
    if inj_scale is None:
        inj_scale = injection_scale_override
    assert inj_scale is not None, (
        f"nla_meta.yaml at {checkpoint_dir!r} has no extraction.injection_scale "
        f"(kind={kind!r}, role={meta.get('role')!r}). Actor checkpoints always "
        f"have it. If this is a critic sidecar or a dataset sidecar, pass "
        f"injection_scale_override explicitly."
    )

    t = meta["tokens"]
    cfg = NLAConfig(
        d_model=d_model,
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        actor_prompt_template=meta["prompt_templates"].get("av")
                              or meta["prompt_templates"]["actor"],
        injection_scale=float(inj_scale),
    )

    # encode(), NOT convert_tokens_to_ids(): byte-level BPE tokenizers (Qwen,
    # GPT-2) key on the byte-string representation, not the unicode char.
    # convert_tokens_to_ids('㈎') → None for Qwen; encode('㈎') → [149705].
    live_inj = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live_inj == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} → {live_inj}, sidecar says "
        f"[{cfg.injection_token_id}]. Multi-token = char split = wrong "
        f"tokenizer or vocab changed."
    )
    assert live_inj[0] != tokenizer.unk_token_id, (
        f"{cfg.injection_char!r} maps to UNK"
    )

    # Verify neighbors by tokenizing the canonical prompt. One-step
    # apply_chat_template(tokenize=True) handles BOS correctly for all
    # architectures (Gemma template includes <bos>; Qwen has none).
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )

    # transformers>=5 compat: apply_chat_template(tokenize=True) no longer
    # reliably returns list[int] - observed returning BatchEncoding (dict)
    # and rendered str on 5.12.1. Normalize to flat list[int].
    if isinstance(ids, str):  # returned rendered text
        input_ids = tokenizer.encode(ids, add_special_tokens=False)
    elif hasattr(ids, "keys"):  # BatchEncoding to ids
        input_ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):  # unwrap ids batch dim
        input_ids = ids[0]

    matches = [i for i, tok in enumerate(ids) if tok == cfg.injection_token_id]
    assert len(matches) == 1, (
        f"injection token appears {len(matches)}× in canonical prompt "
        f"(expected 1). Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1
    assert ids[p - 1] == cfg.injection_left_neighbor_id, (
        f"left neighbor drift: {ids[p-1]} vs sidecar "
        f"{cfg.injection_left_neighbor_id}"
    )
    assert ids[p + 1] == cfg.injection_right_neighbor_id, (
        f"right neighbor drift: {ids[p+1]} vs sidecar "
        f"{cfg.injection_right_neighbor_id}"
    )

    return cfg


# ─── Embedding table (load without materializing full model) ────────────────

def load_embedding_only(
    checkpoint_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Embedding:
    """Load ONLY the input embedding weight tensor from safetensors.

    safe_open reads the single key lazily (~2s for a 12B model vs ~30s for
    the full model). Returns a plain nn.Embedding — if the model's embedding
    class does extra work in forward (Gemma-3: ×√d), apply that scale
    separately after lookup (see resolve_embed_scale).
    """
    root = Path(checkpoint_dir)

    def _find_key(keys: list[str], where: str) -> str:
        m = [k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)]
        assert len(m) == 1, (
            f"expected exactly one input-embedding key in {where} "
            f"(suffixes {_EMBED_KEY_SUFFIXES!r}), got {m!r}"
        )
        return m[0]

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_key(list(weight_map), str(index_path))
        shard = root / weight_map[key]
    else:
        shard = root / "model.safetensors"
        assert shard.exists(), f"no model.safetensors or .index.json at {root!r}"
        with safe_open(str(shard), framework="pt") as f:
            key = _find_key(list(f.keys()), str(shard))

    with safe_open(str(shard), framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)

    vocab, d = weight.shape
    embed = torch.nn.Embedding(vocab, d, _weight=weight)
    embed.requires_grad_(False)
    embed.eval()
    return embed


# Explicit registry of model_types whose embedding forward() multiplies by √d.
# Mirrors nla/arch_adapters.py::_SCALED_EMBED_MODEL_TYPES — keep in sync.
# This file is OSS-standalone so cannot import arch_adapters; the registry is
# small and the drift hazard of a prefix-match (a hypothetical "phi-gemma-moe"
# would spuriously match .startswith("gemma")) is worse than a duplicated set.
_SCALED_EMBED_MODEL_TYPES = frozenset({
    "gemma", "gemma2", "gemma3", "gemma3_text", "t5",
})


def resolve_embed_scale(checkpoint_dir: str | Path) -> float:
    """1.0 for Qwen/Llama/Mistral; √hidden_size for Gemma/T5.

    Gemma3TextScaledWordEmbedding.forward() multiplies by √d (≈62 for
    d=3840). load_embedding_only returns a plain nn.Embedding, so that
    multiply never happens — all token embeddings are 62× too small.
    The injection vector (from residual-stream extraction) IS at full
    scale, so it dominates everything else → garbage.

    If your arch also scales embeddings in forward(), add its model_type
    to _SCALED_EMBED_MODEL_TYPES.
    """
    config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    model_type = getattr(text_cfg, "model_type", "") or ""
    if model_type in _SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(text_cfg.hidden_size)
    return 1.0


# ─── Pure injection math ────────────────────────────────────────────────────

def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Rescale to target_scale L2-norm. Zeros stay zero. Norm in fp32."""
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


def inject_at_marked_positions(
    input_ids: torch.Tensor,      # [1, T]
    embeddings: torch.Tensor,     # [1, T, d]
    vectors: torch.Tensor,        # [N, d]  — N=1 for single-prompt inference
    inj_id: int, left_id: int, right_id: int,
) -> torch.Tensor:
    """Overwrite embedding rows at valid injection positions. Clones first.

    Valid = token at p is inj_id AND tokens at p±1 match the sidecar neighbors.
    The neighbor check rejects false positives from the injection char
    appearing in pasted text / multi-turn context. Found count must equal
    vectors.shape[0] — if it doesn't, CRASH LOUD rather than silently serve
    ㈎-as-text.
    """
    seq_len = input_ids.shape[-1]
    assert input_ids.shape == embeddings.shape[:-1]
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1]
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    vec_idx = 0
    for b, p in (input_ids == inj_id).nonzero().tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1
    assert vec_idx == vectors.shape[0], (
        f"found {vec_idx} injection sites with correct neighbors, expected "
        f"{vectors.shape[0]}. Template drift, tokenizer mismatch, or prompt "
        f"missing the injection marker."
    )
    return out


# ─── Client ─────────────────────────────────────────────────────────────────

class NLAClient:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        sglang_url: str = "http://localhost:30000",
        injection_scale_override: float | None = None,
        device: str = "cpu",
    ):
        """
        checkpoint_dir: HF-format dir with nla_meta.yaml.
        sglang_url:     SGLang server root (NOT /generate — we append that).
        injection_scale_override: only set if the sidecar is wrong/missing.
            The model learned with the sidecar value; overriding = OOD.
        device: embedding lookup device. CPU is fine (~100 rows per request).
        """
        checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        # Pass override INTO load_nla_config so its assert doesn't fire on
        # critic/dataset sidecars that legitimately have injection_scale=null.
        self.cfg = load_nla_config(
            checkpoint_dir, self.tokenizer,
            injection_scale_override=injection_scale_override,
        )

        # bf16 storage: ~300MB (Qwen7B) / ~1GB (Gemma12B). Per-request
        # fp32 cast happens only on the ~100 looked-up rows — trivial.
        self.embed = load_embedding_only(checkpoint_dir, dtype=torch.bfloat16).to(device)
        self.embed_scale = resolve_embed_scale(checkpoint_dir)

        assert self.embed.weight.shape[1] == self.cfg.d_model, (
            f"embedding d={self.embed.weight.shape[1]} != sidecar "
            f"d_model={self.cfg.d_model}. Wrong checkpoint for this sidecar."
        )

        self.sglang_url = sglang_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(120.0))

        print(
            f"[NLAClient] {checkpoint_dir.name}: d_model={self.cfg.d_model} "
            f"inj_scale={self.cfg.injection_scale} embed_scale={self.embed_scale:.2f} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id})"
        )

    # ─── Core inference step ──────────────────────────────────────────────

    def _build_embeds(
        self, v_raw: torch.Tensor, prompt_content: str | None
    ) -> tuple[np.ndarray, int]:
        """Tokenize → embed → arch-scale → inject. Returns (embeds[T,d], prompt_len).

        prompt_content: user message content WITH <INJECT> placeholder. None
        uses the sidecar's canonical actor template (recommended — that's
        what the model was trained on).
        """
        if prompt_content is None:
            content = self.cfg.actor_prompt_template.format(
                injection_char=self.cfg.injection_char
            )
        else:
            assert INJECT_PLACEHOLDER in prompt_content, (
                f"custom prompt must contain {INJECT_PLACEHOLDER!r}"
            )
            content = prompt_content.replace(
                INJECT_PLACEHOLDER, self.cfg.injection_char
            )

        # One-step tokenize. Handles BOS correctly for all architectures —
        # Gemma's chat template includes <bos>, Qwen has none. The two-step
        # apply_chat_template(tokenize=False)→encode(add_special_tokens=False)
        # is equivalent but add_special_tokens=True there would double-BOS
        # Gemma (shifting every position by 1). Qwen has bos_token=None so
        # it's a silent noop there, which makes this easy to miss.
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
        )

        # transformers>=5 compat: apply_chat_template(tokenize=True) no longer
        # reliably returns list[int] - observed returning BatchEncoding (dict)
        # and rendered str on 5.12.1. Normalize to flat list[int].
        if isinstance(input_ids, str):  # returned rendered text
            input_ids = self.tokenizer.encode(input_ids, add_special_tokens=False)
        elif hasattr(input_ids, "keys"):  # BatchEncoding to ids
            input_ids = input_ids["input_ids"]
        if input_ids and isinstance(input_ids[0], list):  # unwrap ids batch dim
            input_ids = input_ids[0]

        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            # bf16 lookup → fp32 for injection math + numpy (no bf16 in numpy).
            # embed_scale: 1.0 for Qwen/Llama, √d for Gemma-3.
            embeds = (self.embed(ids_t.to(self.embed.weight.device))
                      * self.embed_scale).float()

        assert torch.isfinite(v_raw).all(), "activation has NaN/Inf"
        v_scaled = normalize_activation(
            v_raw.float().view(1, -1), self.cfg.injection_scale
        )

        injected = inject_at_marked_positions(
            ids_t, embeds.cpu(), v_scaled,
            self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id,
            self.cfg.injection_right_neighbor_id,
        )
        # SGLang wants [T, d] unbatched, contiguous for orjson's numpy path.
        return injected[0].contiguous().numpy(), len(input_ids)

    def _sglang_generate(
        self, embeds_np: np.ndarray, **sampling: object
    ) -> dict[str, Any]:
        # DO NOT also send input_ids. With both present, SGLang may use
        # input_ids for logprob bookkeeping while forwarding on input_embeds,
        # causing misalignment. Embeds-only is safe.
        #
        # orjson.OPT_SERIALIZE_NUMPY reads the fp32 buffer directly — no
        # 448K-Python-float intermediate from .tolist(). Matters at scale.
        sp = {"temperature": 1.0, "max_new_tokens": 200,
              "skip_special_tokens": False}
        sp.update(sampling)
        body = orjson.dumps(
            {"input_embeds": embeds_np, "sampling_params": sp},
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        resp = self._http.post(
            f"{self.sglang_url}/generate",
            content=body, headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        out = resp.json()
        return out[0] if isinstance(out, list) else out

    # ─── Public API ───────────────────────────────────────────────────────

    def generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> str:
        """Decode one activation vector.

        activation:  [d_model] raw vector — rescaled to cfg.injection_scale.
        prompt:      user-message content with <INJECT> marker. Default (None)
                     uses the sidecar's actor template — RECOMMENDED.
        extract_explanation:  strip <explanation> tags. False returns raw gen
                     (useful for debugging — if ALL outputs are CJK, or
                     describe a CJK char in English, injection likely failed).
        sampling:    SGLang sampling_params (temperature, max_new_tokens, top_p, ...).

        Known-noisy inputs (don't over-interpret poor decodes from these):
        - Early-sequence positions (first ~10 tokens): layer-K has seen few
          tokens, residual stream hasn't accumulated signal. Decodes trend
          toward training prior.
        - Occasional high-norm activations (some models produce rare spikes,
          e.g. Qwen layer-20 early newlines at ~14k vs typical ~100-170).
          Seen during training but rare — unsurprising if decode is poor.
        """
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, (
            f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        )

        embeds_np, _ = self._build_embeds(v, prompt)
        out = self._sglang_generate(embeds_np, **sampling)
        text = out["text"]

        if not extract_explanation:
            return text
        m = EXPLANATION_RE.search(text)
        if m is None:
            # Truncated gen (bump max_new_tokens) or model drift. Return
            # partial; log loudly so caller notices.
            print(f"[NLAClient] WARNING: no <explanation> tags. "
                  f"Raw[:200]={text[:200]!r}")
            return text
        return m.group(1).strip()

    def generate_batch(
        self,
        activations: Iterable[Iterable[float] | np.ndarray | torch.Tensor],
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> list[str]:
        """Sequential requests. SGLang's continuous batcher packs on its end.
        For real throughput, fire these in parallel via async httpx."""
        return [self.generate(v, prompt=prompt,
                              extract_explanation=extract_explanation,
                              **sampling)
                for v in activations]


# ─── CRITIC (activation reconstructor) ───────────────────────────────────────
#
# Optional — the actor is usable standalone. The critic closes the autoencoder
# loop: explanation text → predicted activation vector. MSE against the original
# gives a fidelity score (the RL training reward). Useful if you want to
# rank/filter actor decodes by how reliably they tracked the input.
#
# Architecture: first K+1 layers of the base model (K = extraction layer, e.g.
# K=20 for Qwen → 21 layers kept), final LayerNorm replaced with Identity,
# lm_head stripped, Linear(d,d) value_head bolted on. Extract at tokens[-1]
# (the prompt ends with a fixed suffix like '</text> <summary>').
#
# Checkpoint layout:
#   critic_hf/config.json           — num_hidden_layers = K+1 (pre-truncated)
#   critic_hf/model-*.safetensors   — truncated backbone
#   critic_hf/value_head.safetensors — the Linear head, loaded separately
#   critic_hf/nla_meta.yaml         — mse_scale + critic prompt template

# Final LayerNorm attribute name varies by arch. Qwen2/Llama/Mistral: "norm".
# GPT-2-style: "ln_f". Some: "final_layernorm". Extend if a new arch fails the
# constructor's assert with a clear message.
_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


class NLACritic:
    """Load an NLA critic and compute reconstruction MSE.

    Usage:
        critic = NLACritic("./critic_hf", device="cuda:0")
        mse, cos = critic.score(actor_output_text, original_activation)

    Both are returned because they carry identical information — MSE = 2(1−cos)
    under the L2-norm-to-√d normalization — but cos is usually the more
    intuitive thing to report externally. People know what cos=0.9 means;
    MSE=0.2 needs a lookup table. Pick one and be consistent.

        cos=1.0  → MSE=0.0   perfect
        cos=0.9  → MSE=0.2   good decode (typical for clean positions)
        cos=0.5  → MSE=1.0   mediocre
        cos=0.0  → MSE=2.0   orthogonal
        cos=−1.0 → MSE=4.0   antipodal (never seen in practice)

    On mse_scale vs injection_scale — different things, don't confuse them:

      injection_scale (e.g. 150 for Qwen) is the L2 norm the ACTOR expects
      vectors at — it matches the training-data distribution of activation
      norms. Get it wrong → the vector is OOD → injection fails → CJK output.

      mse_scale (√d_model ≈ 59.87 for Qwen) makes `.mean()` produce the
      d-agnostic `2(1-cos)` value. With both vectors at L2=s, per-element
      MSE is `2s²(1-cos)/d`; choosing s=√d makes s²/d=1. So the multiply
      IS load-bearing — without it you'd get `2(1-cos)/d ≈ 0.0005`. The √d
      choice also kept training-time gradient magnitudes reasonable. The
      returned MSE is already the final answer; don't rescale.
    """

    def __init__(self, checkpoint_dir: str | Path, *,
                 device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        checkpoint_dir = Path(checkpoint_dir)
        meta = yaml.safe_load((checkpoint_dir / "nla_meta.yaml").read_text())
        assert meta["role"] in ("critic", "ar"), (
            f"sidecar role={meta['role']!r}, expected 'critic' or 'ar'. "
            f"Point NLACritic at the AR (reconstructor) checkpoint, not the AV."
        )
        ms = meta["extraction"]["mse_scale"]
        assert ms is not None, (
            f"sidecar mse_scale is None (raw-MSE mode). NLACritic.score() is "
            f"direction-only (2(1-cos)) and requires a numeric mse_scale; this "
            f"checkpoint was trained without normalization and is not supported here."
        )
        self.mse_scale: float = float(ms)
        self.template: str = (meta["prompt_templates"].get("ar")
                              or meta["prompt_templates"]["critic"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        # BOS invariant: training tokenized critic prompts with
        # add_special_tokens=True (reward.py, nla_generate.py). For Gemma/Llama
        # this prepends BOS; for Qwen (bos_token=None) it's a no-op. Dropping
        # BOS shifts position-0 meaning → degraded reconstruction everywhere
        # (observed: Gemma fve_nrm 0.31 vs 0.77). reconstruct() below uses
        # add_special_tokens=True — this assert catches if that ever flips.
        probe = self.tokenizer("x", add_special_tokens=True)["input_ids"]
        bos = self.tokenizer.bos_token_id
        assert bos is None or probe[0] == bos, (
            f"tokenizer has bos_token_id={bos} but add_special_tokens=True "
            f"produced first token {probe[0]}. Critic was trained with BOS "
            f"prefix — reconstruct() must match."
        )

        # config.json already has the truncated num_hidden_layers (K+1) — the
        # checkpoint was produced by training, not on-the-fly truncation here.
        backbone = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True,
        )
        # Strip lm_head (critic never emits logits) and final LN (value head
        # sees raw residual-stream output of block K, not the normed version).
        backbone.lm_head = torch.nn.Identity()
        inner = backbone.model  # Qwen2ForCausalLM.model → Qwen2Model
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(
                f"no final-LN attribute on {type(inner).__name__} — tried "
                f"{_FINAL_LN_ATTRS!r}. Add the arch's attr name to that list."
            )

        d = backbone.config.hidden_size
        self.value_head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        head_path = checkpoint_dir / "value_head.safetensors"
        assert head_path.exists(), (
            f"no value_head.safetensors at {checkpoint_dir!r}. NLA critic "
            f"checkpoints ship this alongside config.json — it's the trained "
            f"reconstruction head, not derivable from the backbone."
        )
        self.value_head.load_state_dict(load_file(str(head_path)))

        self.backbone = backbone.to(device).eval()
        self.value_head = self.value_head.to(device).eval()
        self.device = device
        print(f"[NLACritic] {backbone.config.num_hidden_layers} layers  "
              f"d_model={d}  mse_scale={self.mse_scale:.2f}")

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        """Explanation text → predicted activation vector (raw, unnormalized)."""
        prompt = self.template.format(explanation=explanation)
        # add_special_tokens=True: Gemma critic was trained with BOS prefix
        # (critic_prompt_template is a raw string, not chat-template-processed).
        # Qwen has bos_token=None so this is a no-op there. Omitting BOS for
        # Gemma shifts position-0 meaning → degraded reconstruction everywhere.
        ids = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=True)["input_ids"].to(self.device)
        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]  # last token
        return self.value_head(h).float().cpu()

    def score(self, explanation: str,
              original: np.ndarray | torch.Tensor) -> tuple[float, float]:
        """(direction-MSE, cos-sim). Both pred+gold L2-normalized to mse_scale
        before MSE → MSE = 2(1-cos), range [0, 4]. Orthogonal = 2."""
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        pred_n = pred / pred.norm().clamp_min(1e-12) * self.mse_scale
        gold_n = gold / gold.norm().clamp_min(1e-12) * self.mse_scale
        mse = ((pred_n - gold_n) ** 2).mean().item()
        cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
        return mse, cos


# ─── CLI ────────────────────────────────────────────────────────────────────

def _main() -> None:
    """Feed vectors from a parquet's activation_vector column, or smoke-test
    with one random vector. ALL outputs in CJK (or English describing a CJK
    char)? Injection likely failed — see README §Debugging."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="HF-format NLA actor dir (with nla_meta.yaml)")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--parquet", default=None,
                    help="Parquet with activation_vector column. Default: "
                         "smoke-test with one random vector.")
    ap.add_argument("--n", type=int, default=3, help="rows to sample from parquet")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--injection-scale", type=float, default=None,
                    help="Override sidecar value (OOD — only if sidecar is "
                         "wrong/missing)")
    ap.add_argument("--prompt", default=None,
                    help="Custom user content with <INJECT> marker. Default: "
                         "sidecar's actor template (recommended).")
    ap.add_argument("--raw", action="store_true",
                    help="Print raw output (no tag extraction)")
    args = ap.parse_args()

    client = NLAClient(
        args.checkpoint,
        sglang_url=args.sglang_url,
        injection_scale_override=args.injection_scale,
    )

    if args.parquet is None:
        print("[smoke] No parquet — generating for one random unit vector.")
        v = np.random.randn(client.cfg.d_model).astype(np.float32)
        out = client.generate(
            v, prompt=args.prompt,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            extract_explanation=not args.raw,
        )
        print(f"\n{out}\n")
        return

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet)
    batch = next(pf.iter_batches(batch_size=args.n, columns=["activation_vector"]))
    # flatten→reshape avoids to_pylist()'s O(n×d) Python-float creation
    flat = batch.column("activation_vector").flatten().to_numpy(
        zero_copy_only=False).astype(np.float32)
    vecs = flat.reshape(len(batch), -1)

    for i, v in enumerate(vecs):
        out = client.generate(
            v, prompt=args.prompt,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            extract_explanation=not args.raw,
        )
        print(f"─── [{i}]  ||v||={np.linalg.norm(v):.1f} ─────────────────────")
        print(out)
        print()


if __name__ == "__main__":
    _main()
