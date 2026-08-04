"""GoEmotions -> emotion passages -> mean-difference steering vectors.

WHY PASSAGES AND NOT RAW COMMENTS
  The released NLA was trained on activations extracted at token position >= 50
  (CLAUDE.md, stage-0 `_MIN_POSITION = 50`: "need enough left-context for the
  activation to be meaningful. Earlier positions decode to noise"). Raw
  GoEmotions comments are far below that — measured over our selected pool,
  median 15 Qwen tokens, p99 = 35, and 0.0% reach 50. Reading last-token
  activations off raw comments would put every capture out of distribution.

  So we concatenate several same-emotion comments into one passage of >=
  MIN_TOKENS. The left context is then both long enough AND emotionally
  consistent, which is exactly what a mean-difference direction wants. The
  source text stays real and public, so the benchmark remains reproducible.

EMOTION SELECTION
  12 emotions, chosen for (a) >= ~400 single-label training examples and
  (b) low pairwise confusability. Near-synonym pairs present in GoEmotions
  (anger/annoyance, joy/excitement, admiration/approval) are deliberately NOT
  both included: the benchmark asks "did the reader name THIS emotion", which is
  ill-posed when two labels are near-interchangeable.

  Only single-label rows are used — multi-label rows have, by construction,
  ambiguous emotional content.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Sequence

import numpy as np

EMOTIONS = [
    # Ekman six
    "anger", "joy", "sadness", "fear", "disgust", "surprise",
    # distinct positives / social
    "gratitude", "love", "amusement", "admiration",
    # epistemic
    "curiosity", "confusion",
]
NEUTRAL = "neutral"

MIN_TOKENS = 60          # comfortably past stage-0's _MIN_POSITION = 50
N_PER_EMOTION = 100      # brief asks for 50-100
N_NEUTRAL = 200          # the neutral mean is subtracted from every emotion,
                         # so it is worth estimating more precisely
_ARTIFACTS = ("[NAME]", "[RELIGION]")


def _clean_pool(split="train") -> dict[str, list[str]]:
    """Single-label, artifact-free comments grouped by emotion name."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified")[split]
    names = ds.features["labels"].feature.names
    pool: dict[str, list[str]] = collections.defaultdict(list)
    for row in ds:
        if len(row["labels"]) != 1:
            continue
        text = row["text"].strip()
        if any(a in text for a in _ARTIFACTS) or len(text) < 10:
            continue
        pool[names[row["labels"][0]]].append(text)
    return pool


def build_passages(
    n_per_emotion: int = N_PER_EMOTION,
    n_neutral: int = N_NEUTRAL,
    min_tokens: int = MIN_TOKENS,
    seed: int = 0,
    tokenizer=None,
) -> dict[str, list[str]]:
    """Concatenate same-emotion comments into >= min_tokens passages.

    Sampling is without replacement WITHIN a passage (no duplicated sentences)
    but with replacement across passages, since rare emotions (fear, disgust)
    have only a few hundred usable comments. Seeded, so the released benchmark
    is byte-reproducible.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    pool = _clean_pool()
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {}

    for emotion in EMOTIONS + [NEUTRAL]:
        comments = pool[emotion]
        assert len(comments) >= 50, f"{emotion}: only {len(comments)} usable comments"
        target = n_neutral if emotion == NEUTRAL else n_per_emotion
        passages: list[str] = []
        while len(passages) < target:
            order = rng.permutation(len(comments))
            parts, n_tok, i = [], 0, 0
            while i < len(order) and n_tok < min_tokens:
                c = comments[order[i]]
                parts.append(c if c.endswith((".", "!", "?")) else c + ".")
                n_tok = len(tokenizer(" ".join(parts))["input_ids"])
                i += 1
            if n_tok >= min_tokens:
                passages.append(" ".join(parts))
        out[emotion] = passages[:target]
    return out


def build_raw(
    n_per_emotion: int = N_PER_EMOTION,
    n_neutral: int = N_NEUTRAL,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Single GoEmotions comments, no concatenation — the brief's original design.

    Round-trip MSE says the AV reads these BETTER than concatenated passages
    (0.219 vs 0.319), so the min-position worry that motivated concatenation does
    not hold at inference time. Kept alongside `build_passages` so the two
    constructions can be compared functionally rather than argued about.
    """
    pool = _clean_pool()
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {}
    for emotion in EMOTIONS + [NEUTRAL]:
        comments = pool[emotion]
        target = n_neutral if emotion == NEUTRAL else n_per_emotion
        idx = rng.choice(len(comments), size=min(target, len(comments)), replace=False)
        out[emotion] = [comments[i] for i in idx]
    return out


def build_emotion_vector(
    model, tok, emotion_texts: Sequence[str], neutral_texts: Sequence[str],
    *, block_index: int = 20, batch_size: int = 8,
) -> np.ndarray:
    """v = mean(act | emotion) - mean(act | neutral). One direction per emotion."""
    from bench.capture import capture_activations

    emo = capture_activations(model, tok, emotion_texts,
                              block_index=block_index, batch_size=batch_size)
    neu = capture_activations(model, tok, neutral_texts,
                              block_index=block_index, batch_size=batch_size)
    return emo.mean(axis=0) - neu.mean(axis=0)


def _main() -> None:
    import argparse
    import json

    from bench.capture import capture_activations, load_model

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--block-index", type=int, default=20)
    ap.add_argument("--n-per-emotion", type=int, default=N_PER_EMOTION)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building passages from GoEmotions ...")
    passages = build_passages(n_per_emotion=args.n_per_emotion, seed=args.seed)
    (out_dir / "emotion_passages.json").write_text(json.dumps(passages, indent=2))
    ntok = {e: len(v) for e, v in passages.items()}
    print(f"  {len(passages)} groups: " + ", ".join(f"{e}={n}" for e, n in ntok.items()))

    model, tok = load_model()

    # Capture everything ONCE, then take means. Capturing per-emotion would
    # re-run the neutral set 12 times, and the probe baseline (T5) needs the
    # per-passage activations anyway.
    acts: dict[str, np.ndarray] = {}
    for name, texts in passages.items():
        acts[name] = capture_activations(model, tok, texts, block_index=args.block_index)
        print(f"  captured {name:<12} {acts[name].shape}  "
              f"mean||v||={np.linalg.norm(acts[name], axis=1).mean():.1f}")

    neutral_mean = acts[NEUTRAL].mean(axis=0)
    vectors = {e: acts[e].mean(axis=0) - neutral_mean for e in EMOTIONS}

    np.savez(out_dir / "emotion_vectors.npz",
             **vectors, _neutral_mean=neutral_mean, _block_index=args.block_index)
    np.savez(out_dir / "emotion_activations.npz", **acts)

    print(f"\n{'emotion':<14}{'||v||':>9}{'cos to nearest other emotion':>32}")
    keys = list(vectors)
    M = np.stack([vectors[k] / np.linalg.norm(vectors[k]) for k in keys])
    sim = M @ M.T
    np.fill_diagonal(sim, -np.inf)
    for i, k in enumerate(keys):
        j = int(sim[i].argmax())
        print(f"{k:<14}{np.linalg.norm(vectors[k]):>9.1f}"
              f"{f'{keys[j]} ({sim[i, j]:.2f})':>32}")
    print(f"\nwrote {out_dir/'emotion_vectors.npz'}, {out_dir/'emotion_activations.npz'}, "
          f"{out_dir/'emotion_passages.json'}")


if __name__ == "__main__":
    _main()
