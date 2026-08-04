"""Transformers-backed NLA verbalizer — a drop-in for `nla_inference.NLAClient`.

WHY THIS EXISTS
  The documented inference path serves the AV via SGLang and posts `input_embeds`
  to /generate. This box runs NVIDIA driver 470 (CUDA 11.4 max), and `sgl-kernel`
  ships cu12-only wheels needing driver >=525 — SGLang cannot run here at all.

  So we keep everything except the transport. `LocalNLAClient` subclasses
  `NLAClient` and overrides exactly one method, `_sglang_generate`, swapping the
  HTTP POST for an in-process `model.generate(inputs_embeds=...)`. Prompt
  construction, the sidecar contract, injection-scale normalization and the
  neighbor-checked injection are all inherited unchanged from `_build_embeds`,
  which is the property we care about: the vector reaching the model is the same
  one SGLang would have received.

SAMPLING FIDELITY (subtle, matters)
  SGLang applies ONLY the sampling params in the request. HF `generate()` instead
  starts from the checkpoint's generation_config.json, and Qwen2.5's ships
  top_p=0.8, top_k=20, repetition_penalty=1.05. Silently inheriting those would
  make local decodes differ from served ones for reasons unrelated to injection.
  So the defaults below neutralize them (top_p=1.0, top_k=0, rep=1.0) to match
  SGLang, and any caller override is passed through explicitly.

USAGE
  python -m bench.av_local <checkpoint_dir>            # random-vector smoke test
  python -m bench.av_local <checkpoint_dir> --parquet acts.parquet --n 3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Repo root on sys.path so `nla_inference` imports under both `python bench/av_local.py`
# and `python -m bench.av_local`.
_ROOT = Path(__file__).resolve().parent.parent
if (_ROOT / "nla_inference.py").exists() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nla_inference import NLAClient  # noqa: E402

# Sampling keys we know how to translate SGLang -> HF. Anything else raises
# rather than being silently dropped: quietly ignoring e.g. `stop` would change
# results while looking like it worked.
_SUPPORTED_SAMPLING = frozenset({
    "temperature", "max_new_tokens", "top_p", "top_k",
    "repetition_penalty", "skip_special_tokens", "seed",
})


class LocalNLAClient(NLAClient):
    """`NLAClient` with the SGLang HTTP call replaced by a local forward pass."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        injection_scale_override: float | None = None,
        model: Any = None,
    ):
        """
        device / dtype: where the AV itself runs. bf16 on one A6000 is ~15 GB.
        model: an already-loaded AV, to avoid a second copy if the caller has one.
        """
        # Parent loads the tokenizer, validates the sidecar against it, and loads
        # a CPU copy of the embedding table for `_build_embeds`. The sglang_url is
        # inherited but never used — we override the only method that reads it.
        super().__init__(
            checkpoint_dir,
            sglang_url="http://localhost:0",
            injection_scale_override=injection_scale_override,
            device="cpu",
        )

        if model is None:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir), torch_dtype=dtype,
            )
            model = model.to(device)
        self.model = model.eval()

        # The AV's own embedding table must match the one `_build_embeds` injects
        # into, or we would be splicing a vector into a different vector space.
        assert self.model.get_input_embeddings().weight.shape[0] == self.embed.weight.shape[0], (
            "AV vocab size disagrees with the embedding table loaded from safetensors"
        )

        pad = self.tokenizer.pad_token_id
        self._pad_id = pad if pad is not None else self.tokenizer.eos_token_id

        print(f"[LocalNLAClient] AV on {self.model.device}, dtype={self.model.dtype}")

    def _sglang_generate(self, embeds_np: np.ndarray, **sampling: object) -> dict[str, Any]:
        """Run the injected embeddings through the AV locally.

        Signature and return shape match the parent's HTTP version so that
        `generate()` / `generate_batch()` work unmodified.
        """
        unknown = set(sampling) - _SUPPORTED_SAMPLING
        assert not unknown, (
            f"unsupported sampling params for the local backend: {sorted(unknown)}. "
            f"Supported: {sorted(_SUPPORTED_SAMPLING)}."
        )

        # Defaults mirror SGLang's, NOT the checkpoint's generation_config.json.
        sp: dict[str, Any] = {
            "temperature": 1.0, "max_new_tokens": 200, "top_p": 1.0, "top_k": 0,
            "repetition_penalty": 1.0, "skip_special_tokens": False,
        }
        sp.update(sampling)

        seed = sp.pop("seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))

        max_new = int(sp.pop("max_new_tokens"))
        temperature = float(sp.pop("temperature"))
        skip_special = bool(sp.pop("skip_special_tokens"))
        top_k = int(sp.pop("top_k"))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "repetition_penalty": float(sp.pop("repetition_penalty")),
            "pad_token_id": self._pad_id,
        }
        if temperature > 0.0:
            # top_k=0 means "no truncation" in SGLang; HF spells that as None.
            gen_kwargs.update(
                do_sample=True, temperature=temperature,
                top_p=float(sp.pop("top_p")), top_k=(top_k or None),
            )
        else:
            gen_kwargs.update(do_sample=False)
            sp.pop("top_p", None)

        embeds = (torch.from_numpy(embeds_np)
                  .unsqueeze(0)
                  .to(self.model.device, self.model.dtype))
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)

        with torch.no_grad():
            out = self.model.generate(inputs_embeds=embeds, attention_mask=attn, **gen_kwargs)

        # With `inputs_embeds` and no `input_ids`, HF returns ONLY the new tokens.
        # Assert rather than trust it: a future version that prepends the prompt
        # would otherwise have us silently "verbalize" the prompt back.
        assert out.shape[1] <= max_new, (
            f"expected <={max_new} generated tokens, got {out.shape[1]} — "
            f"generate() appears to have prepended the prompt."
        )
        return {"text": self.tokenizer.decode(out[0], skip_special_tokens=skip_special)}

    def unload(self) -> None:
        """Free the AV's GPU memory (the base model needs the card for capture)."""
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()


def _main() -> None:
    """Verbalize vectors from a parquet's activation_vector column, or smoke-test
    with one random vector. Mirrors `nla_inference._main` minus --sglang-url.

    ALL outputs in CJK (or English describing a CJK char)? Injection failed —
    see docs/inference.md §Debugging."""
    import argparse

    ap = argparse.ArgumentParser(description=_main.__doc__)
    ap.add_argument("checkpoint", help="HF-format NLA actor dir (with nla_meta.yaml)")
    ap.add_argument("--parquet", default=None,
                    help="Parquet with activation_vector column. Default: "
                         "smoke-test with one random vector.")
    ap.add_argument("--n", type=int, default=3, help="rows to sample from parquet")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--injection-scale", type=float, default=None,
                    help="Override sidecar value (OOD — only if sidecar is wrong)")
    ap.add_argument("--prompt", default=None,
                    help="Custom user content with <INJECT> marker. Default: "
                         "sidecar's actor template (recommended).")
    ap.add_argument("--raw", action="store_true", help="Print raw output (no tag extraction)")
    args = ap.parse_args()

    client = LocalNLAClient(
        args.checkpoint, device=args.device,
        injection_scale_override=args.injection_scale,
    )
    sampling = dict(temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    if args.seed is not None:
        sampling["seed"] = args.seed

    if args.parquet is None:
        print("[smoke] No parquet — generating for one random unit vector.")
        rng = np.random.default_rng(args.seed)
        v = rng.standard_normal(client.cfg.d_model).astype(np.float32)
        out = client.generate(v, prompt=args.prompt,
                              extract_explanation=not args.raw, **sampling)
        print(f"\n{out}\n")
        return

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet)
    batch = next(pf.iter_batches(batch_size=args.n, columns=["activation_vector"]))
    flat = batch.column("activation_vector").flatten().to_numpy(
        zero_copy_only=False).astype(np.float32)
    vecs = flat.reshape(len(batch), -1)

    for i, v in enumerate(vecs):
        out = client.generate(v, prompt=args.prompt,
                              extract_explanation=not args.raw, **sampling)
        print(f"─── [{i}]  ||v||={np.linalg.norm(v):.1f} ─────────────────────")
        print(out)
        print()


if __name__ == "__main__":
    _main()
