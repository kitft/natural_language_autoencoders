"""AR round-trip fidelity: activation -> AV explanation -> AR reconstruction -> MSE.

This is the grader-free faithfulness signal. `NLACritic.score` returns
(mse, cos) with mse = 2(1-cos) after both vectors are L2-normalized to
mse_scale (= sqrt(d_model)), so mse is bounded [0, 4] and 2.0 means orthogonal.
Reference points from nla_inference.py: cos 0.9 -> mse 0.2 is a good decode,
cos 0.5 -> mse 1.0 is mediocre.

Memory discipline: the base model (~15 GB), AV (~15 GB) and AR (~11 GB) do not
comfortably coexist on one 48 GB A6000 alongside activations and KV cache, so
callers should run capture -> verbalize -> reconstruct as three sequential
phases, freeing between them. `free_model` is the helper for that.
"""

from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import torch


def free_model(*objs: Any) -> None:
    """Drop references and reclaim VRAM between phases."""
    for o in objs:
        for attr in ("model", "backbone", "value_head"):
            if hasattr(o, attr):
                setattr(o, attr, None)
    gc.collect()
    torch.cuda.empty_cache()


def verbalize_all(av, vectors: Sequence[np.ndarray], *, max_new_tokens: int = 220,
                  temperature: float = 0.0, progress: bool = True,
                  prompt: str | None = None) -> list[str]:
    """Explanation text for each activation. Greedy by default so reruns match.

    prompt=None uses the sidecar's canonical template, which is what the AV was
    trained on. A custom prompt must contain the <INJECT> marker.
    """
    from tqdm import tqdm

    it = tqdm(vectors, desc="verbalize", disable=not progress)
    return [av.generate(v, temperature=temperature, max_new_tokens=max_new_tokens,
                        prompt=prompt)
            for v in it]


def score_all(ar, explanations: Sequence[str], originals: Sequence[np.ndarray],
              *, progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(mse[N], cos[N]) for each (explanation, original activation) pair."""
    from tqdm import tqdm

    assert len(explanations) == len(originals)
    mses, coss = [], []
    it = tqdm(range(len(explanations)), desc="reconstruct", disable=not progress)
    for i in it:
        mse, cos = ar.score(explanations[i], originals[i])
        mses.append(mse)
        coss.append(cos)
    return np.asarray(mses, dtype=np.float64), np.asarray(coss, dtype=np.float64)


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired comparison of two MSE arrays over the SAME items (a vs b).

    Paired matters: per-text variance in how "decodable" an activation is dwarfs
    the between-layer difference, so an unpaired comparison would be far noisier.
    Wilcoxon rather than a t-test — n is small and MSE is not normal.
    """
    from scipy.stats import wilcoxon

    d = a - b
    out = {
        "mean_diff": float(d.mean()),
        "a_lower_frac": float((d < 0).mean()),   # fraction where a beats b
    }
    try:
        out["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    except ValueError:      # all differences zero
        out["wilcoxon_p"] = 1.0
    return out


def summarize(mse: np.ndarray, cos: np.ndarray) -> dict[str, float]:
    n = len(mse)
    return {
        "n": n,
        "mse_mean": float(mse.mean()),
        "mse_sem": float(mse.std(ddof=1) / np.sqrt(n)),
        "cos_mean": float(cos.mean()),
        "cos_sem": float(cos.std(ddof=1) / np.sqrt(n)),
        "mse_median": float(np.median(mse)),
    }
