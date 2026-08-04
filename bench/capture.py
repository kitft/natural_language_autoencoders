"""Capture residual-stream activations from a base model and dump them to parquet.

LAYER INDEX CONVENTION — READ THIS
  `block_index=K` means the OUTPUT of decoder block K (post-MLP, post-residual-add),
  which is HF's `hidden_states[K+1]` because `hidden_states[0]` is the embedding
  output. This matches `nla/datagen/extractors.py`, which is what produced the
  training data for the released NLA, and it is what the sidecar's
  `extraction_layer_index: 20` refers to.

  The task brief's phrase "read hidden_states[20]" is off by one — that is the
  output of block 19. Use block_index=20 (== hidden_states[21]).
  `verify_layer_convention.py` settles this empirically.

The parquet stores RAW vectors — no normalization, matching the repo-wide invariant
in CLAUDE.md ("Data-gen NEVER normalizes"). The AV rescales to injection_scale at
injection time.

USAGE
  python -m bench.capture --texts-file passages.txt --out acts.parquet
  python -m bench.capture --text "I am furious about this." --out one.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BLOCK_INDEX = 20        # output of block 20 == hidden_states[21]; see sidecar
D_MODEL = 3584


def load_model(model_name: str = BASE_MODEL, device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16):
    """Load the base model whose residual stream we read."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Right-padding is mandatory: we index the last REAL token via the attention
    # mask, and left-padding would put pad tokens there. Mirrors extractors.py.
    tok.padding_side = "right"
    tok.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device).eval(), tok


def _decoder_layers(model) -> Any:
    """Use the repo's own arch adapter rather than assuming model.model.layers.

    That assumption holds for Qwen/Llama but breaks on Gemma-3, whose config nests
    everything under text_config and whose decoder lives at a different path.
    nla.arch_adapters already resolves both, and is what the upstream extractor uses.
    """
    from nla.arch_adapters import resolve_decoder_layers
    return resolve_decoder_layers(model)


def capture_activations(
    model, tok, texts: Sequence[str], *,
    block_index: int = BLOCK_INDEX,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Return [N, d_model] fp32 activations at the last real token of each text.

    Uses a forward hook on `model.model.layers[block_index]` rather than
    `output_hidden_states=True` — same tensor, but without materializing all
    N_layers activations. Identical to the upstream extractor's method.
    """
    layers = _decoder_layers(model)
    assert 0 <= block_index < len(layers), (
        f"block_index={block_index} out of range for {len(layers)} layers"
    )

    grabbed: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = h.detach().clone()

    handle = layers[block_index].register_forward_hook(hook)
    from nla.arch_adapters import resolve_text_config
    d_model = resolve_text_config(model.config).hidden_size
    out = np.empty((len(texts), d_model), dtype=np.float32)
    try:
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(model.device)
            with torch.no_grad():
                model(**enc)
            h = grabbed["h"]                                  # [B, T, d]
            last = enc["attention_mask"].sum(dim=1) - 1       # last real token
            picked = h[torch.arange(h.shape[0], device=h.device), last]
            out[start:start + len(chunk)] = picked.float().cpu().numpy()
    finally:
        handle.remove()
    return out


def write_parquet(path: str | Path, vectors: np.ndarray, texts: Sequence[str],
                  *, block_index: int = BLOCK_INDEX, model_name: str = BASE_MODEL,
                  labels: Sequence[str] | None = None) -> Path:
    """Write an `activation_vector` parquet the AV client can consume directly."""
    assert vectors.ndim == 2 and len(vectors) == len(texts)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    cols: dict[str, Any] = {
        "activation_vector": pa.array(list(vectors), type=pa.list_(pa.float32())),
        "text": pa.array(list(texts), type=pa.string()),
        "norm": pa.array([float(np.linalg.norm(v)) for v in vectors], type=pa.float32()),
    }
    if labels is not None:
        assert len(labels) == len(texts)
        cols["label"] = pa.array(list(labels), type=pa.string())

    table = pa.table(cols, metadata={
        b"base_model": model_name.encode(),
        b"block_index": str(block_index).encode(),
        b"hidden_states_index": str(block_index + 1).encode(),
        # Matches the repo-wide invariant: parquets hold raw, unnormalized vectors.
        b"norm_convention": b"none",
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_parquet(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Inverse of write_parquet: returns ([N, d] fp32, texts)."""
    t = pq.read_table(path)
    flat = t.column("activation_vector").combine_chunks().flatten().to_numpy(
        zero_copy_only=False).astype(np.float32)
    vecs = flat.reshape(t.num_rows, -1)
    return vecs, t.column("text").to_pylist()


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="single text to capture")
    src.add_argument("--texts-file", help="one text per line")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--block-index", type=int, default=BLOCK_INDEX,
                    help=f"decoder block whose OUTPUT to read (default {BLOCK_INDEX}; "
                         f"== hidden_states[block_index+1])")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    texts = ([args.text] if args.text
             else [ln.strip() for ln in Path(args.texts_file).read_text().splitlines()
                   if ln.strip()])

    model, tok = load_model(args.model)
    vecs = capture_activations(model, tok, texts, block_index=args.block_index,
                               batch_size=args.batch_size)
    out = write_parquet(args.out, vecs, texts,
                        block_index=args.block_index, model_name=args.model)
    norms = np.linalg.norm(vecs, axis=1)
    print(f"wrote {out}  [{len(texts)} x {vecs.shape[1]}]  "
          f"block {args.block_index} (= hidden_states[{args.block_index + 1}])  "
          f"||v||: min {norms.min():.1f} / mean {norms.mean():.1f} / max {norms.max():.1f}")


if __name__ == "__main__":
    _main()
