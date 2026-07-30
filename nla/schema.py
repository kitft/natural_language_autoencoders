"""Shared NLA sidecar schema — single source of truth for token metadata.

Imported by both nla/datagen/ (writes sidecars) and nla/config.py (reads + asserts).
Schema: docs/design.md §2.

Two sidecar conventions:
  - Dataset: {parquet_path}.nla_meta.yaml  (kind: nla_dataset)
  - Model:   {checkpoint_dir}/nla_meta.yaml (kind: nla_model)
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyarrow.parquet as pq
import torch

SIDECAR_SUFFIX = ".nla_meta.yaml"
SIDECAR_BASENAME = "nla_meta.yaml"

# Actor response payload tags. Datagen (stage3_build.py) wraps the AV-SFT
# response column with these; training (nla_generate, nla_rm) parses them
# back out. Single source of truth — if tags change, both sides update.
EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"
EXPLANATION_RE = re.compile(
    f"{re.escape(EXPLANATION_OPEN)}(.*?){re.escape(EXPLANATION_CLOSE)}",
    re.DOTALL,
)


def wrap_explanation(text: str) -> str:
    """Wrap text in explanation tags for the AV-SFT response column.

    Datagen's stage3_build.py should use this for the `response` column
    so the format is locked to what extract_explanation can parse.
    """
    return f"{EXPLANATION_OPEN}\n{text}\n{EXPLANATION_CLOSE}"


def extract_explanation(response: str) -> str | None:
    """Extract payload between explanation tags; None on miss.

    Strips whitespace so the result matches what datagen's stage3_build used
    to fill the critic template (the raw api_explanation, no \n wrapper).
    Without this, RL queries the critic with <text>\nfoo\n</text> but AR-SFT
    trained it on <text>foo</text> — different tokens.
    """
    m = EXPLANATION_RE.search(response)
    return m.group(1).strip() if m else None


# Parquet column name — datagen writes it, NLADataSource + rollouts read it.
ACTIVATION_COLUMN = "activation_vector"

# Placeholder in parquet prompt column — datagen writes <INJECT> literal,
# NLADataSource swaps it for the injection char at load time.
INJECT_PLACEHOLDER = "<INJECT>"

# multimodal_train_inputs dict keys — rollouts stash, train_actor + loss read.
# String typo here = silent KeyError deep in training.
MM_ACTIVATION_KEY = "nla_activation"
MM_CRITIC_TOKENS_KEY = "nla_critic_tokens"
MM_MSE_SCALE_KEY = "nla_mse_scale"

# Sentinel for extraction.{injection_scale, mse_scale} — resolve to sqrt(d_model)
# at load time. Lets sidecars say "use the default" without baking a float.
SCALE_SQRT_D = "sqrt_d_model"


def resolve_target_scale(raw: float | str | None, d_model: int) -> float | None:
    """Turn a scale value (from sidecar or CLI) into a concrete float or None.

    Accepts:
      - None / "raw" / "none"  → None → no normalization (use raw vectors)
      - "sqrt_d_model"         → sqrt(d_model) — ambient residual-stream scale
      - a float or float-string → that exact L2 norm

    Key-absent in sidecar is NOT None — config.py supplies "sqrt_d_model" as
    the default to .get(), so absent ⇒ normalize to sqrt(d). Explicit null
    (or "raw" from CLI) is the only way to opt out.
    """
    if raw is None or raw in ("raw", "none"):
        return None
    if raw == SCALE_SQRT_D:
        return math.sqrt(d_model)
    if isinstance(raw, str):
        return float(raw)  # ValueError on bad string — loud
    assert isinstance(raw, (int, float)), (
        f"scale must be None/'raw', {SCALE_SQRT_D!r}, or a number; got {raw!r}"
    )
    return float(raw)


@dataclass
class NLATokenMeta:
    """Token IDs pinned at dataset/model generation time.

    Training-side hook scans for injection_token_id + verifies neighbors.

    Critic extraction: position = last token of the prompt (before any padding
    or EOS the training-side tokenizer adds). `critic_suffix_ids` is the
    expected last-N tokens — training can verify the prompt ends with these
    as a one-time sanity check at dataset load, then just index `tokens[-1]`
    per-forward (no scanning, GPU-friendly).

    ALL IDs must match the live tokenizer — drift = silent wrong-position.
    """
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    critic_suffix_ids: list[int] | None = None


def sidecar_path_for(path: str | Path) -> Path:
    """Resolve sidecar path for either a parquet file or a checkpoint directory.

    Handles miles' path-slicing syntax (foo.parquet@[0:1000]) by stripping the
    slice suffix before appending SIDECAR_SUFFIX.

    Directory vs file detection uses is_dir() (exists + is-dir) OR a heuristic
    (no file extension) so write-paths to not-yet-created dirs work too.
    """
    p = Path(str(path).split("@[")[0])
    if p.is_dir() or (not p.exists() and p.suffix == ""):
        return p / SIDECAR_BASENAME
    return p.with_name(p.name + SIDECAR_SUFFIX)


def normalize_activation(v: torch.Tensor, target_scale: float | None) -> torch.Tensor:
    """Scale vectors to target_scale L2-norm, or pass through if None.

    Used for TWO distinct purposes:
      - Actor injection (train_actor.py): scale = cfg.injection_scale.
        A tunable HYPERPARAMETER — affects what the model learns.
      - Critic MSE (loss.py): scale = cfg.mse_scale. Applied to BOTH
        pred and gold symmetrically. Just loss numerical stability.

    target_scale=None → no-op pass-through for either purpose.
    Idempotent. Zero vectors stay zero. Norm computed in fp32 for precision;
    single division v / (||v||_fp32 / scale).
    """
    if target_scale is None:
        return v
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


def compute_predict_mean_baselines(
    vectors: torch.Tensor, mse_scale: float | None
) -> tuple[float, float]:
    """Two predict-the-mean baseline MSEs for FVE logging:

      fve_nrm_meannorm baseline: MSE(v_norm, normalize(μ)). The critic's
        best achievable constant pred — its output ALSO gets normalized.
        ≈ 0.94 for Qwen7B layer-20. FVE>0 ⇒ critic is learning.
        critic_rand (shuffled targets) lands near this.

      fve_nrm baseline: MSE(v_norm, μ). Raw per-element variance of the
        normalized distribution. ≈ 0.72 for Qwen7B. Tighter baseline
        (critic can't literally output short μ under normalization) but
        matches the classical "fraction of variance explained" definition.

    Returns (meannorm_baseline, raw_variance_baseline).
    """
    v_norm = normalize_activation(vectors.float(), mse_scale)
    mu = v_norm.mean(dim=0, keepdim=True)
    mu_normed = normalize_activation(mu, mse_scale)
    mse_meannorm = ((v_norm - mu_normed) ** 2).mean().item()
    mse_rawvar = ((v_norm - mu) ** 2).mean().item()
    return mse_meannorm, mse_rawvar


def load_predict_mean_baselines(
    parquet_source: BinaryIO | str, mse_scale: float | None, max_rows: int = 50_000
) -> tuple[float, float]:
    """Read activation_vector column from parquet, compute both baselines.

    max_rows caps memory — 50k × 3584 × fp32 ≈ 700MB. Sampling error on the
    variance estimate is O(1/√n) — tight at 50k.
    """
    pf = pq.ParquetFile(parquet_source)
    rows = []
    n = 0
    for batch in pf.iter_batches(batch_size=8192, columns=[ACTIVATION_COLUMN]):
        # ListArray → flat values → reshape. Avoids to_pylist() which creates
        # millions of PyFloat objects (same GC pressure we fixed in data_source).
        col = batch.column(ACTIVATION_COLUMN)
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        chunk = flat.reshape(len(col), -1)
        rows.append(chunk)
        n += chunk.shape[0]
        if n >= max_rows:
            break
    V = torch.from_numpy(np.concatenate(rows, axis=0)[:max_rows])
    return compute_predict_mean_baselines(V, mse_scale)


def compute_canonical_neighbors(
    tokenizer: Any,
    actor_template: str,
    injection_char: str,
    injection_token_id: int,
) -> tuple[int, int]:
    """Tokenize the canonical actor prompt, return token IDs at inj_pos ± 1.

    datagen calls this to POPULATE neighbor fields in the sidecar.
    config.py calls this to VERIFY them against the live tokenizer.
    Both must use identical tokenization — this function is that contract.

    add_generation_prompt=True matches RL's prompt format. AV-SFT training appends
    an assistant message instead, but the neighbors are inside the user content
    (<concept>㊗</concept>) so the trailing chat-template scaffolding is identical.
    """
    content = actor_template.format(injection_char=injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )

    # transformers>=5 compat: apply_chat_template(tokenize=True) no longer
    # reliably returns list[int]. Observed returning BatchEncoding on 5.12/5.14.
    # enumerate() over a BatchEncoding iterates dict KEYS, so the scan below
    # finds 0 matches and asserts as if the vocab had drifted.
    if isinstance(ids, str):
        ids = tokenizer.encode(ids, add_special_tokens=False)
    elif hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    matches = [i for i, tid in enumerate(ids) if tid == injection_token_id]
    assert len(matches) == 1, (
        f"injection token id {injection_token_id} ({injection_char!r}) appears "
        f"{len(matches)}× in canonical actor prompt (expected 1). Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1, (
        f"injection token at position {p} is at edge of sequence (len={len(ids)})"
    )
    return ids[p - 1], ids[p + 1]
