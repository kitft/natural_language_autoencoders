"""Build injected activations: neutral_act + alpha * v, with alpha made comparable.

THE NORMALIZATION FIX
  Raw mean-difference vectors have very different norms across emotions — measured
  on our GoEmotions vectors, fear 17.2 up to gratitude 33.0, a 1.9x spread. Scaling
  each by the same alpha therefore injects a 1.9x different dose depending on which
  emotion you picked, so a cross-emotion recovery comparison would partly be
  measuring vector norm. EmoSteer-TTS (arXiv 2508.03543) L2-normalizes its
  difference-in-means vector before applying a steering weight for the same reason.

WHY ALPHA IS A RATIO HERE
  The AV rescales whatever vector it receives to `injection_scale` (150 for Qwen)
  before injecting, so the magnitude of `neutral_act + alpha*v` is discarded
  entirely — only its DIRECTION reaches the model. What actually varies with alpha
  is the angle between the injected vector and v, and that depends on the ratio
  alpha*||v|| / ||neutral_act||, not on either norm alone.

  So alpha is defined dimensionlessly: alpha = 1 adds a vector whose norm equals
  the carrier activation's own norm. This makes alpha comparable across emotions
  AND across carrier passages, and it is the quantity the readout actually
  responds to.

  Mapping to the brief's grid: with ||v|| ~ 22 and ||neutral_act|| ~ 103, the
  brief's alpha in {0,2,4,6,8,12} corresponds to ratios {0, .43, .87, 1.3, 1.7, 2.6}.
  The default grid below spans the same range.
"""

from __future__ import annotations

import numpy as np

# Dimensionless. Spans "barely perturbed" to "v dominates the direction".
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    assert n > 1e-9, "cannot normalize a zero vector"
    return v / n


def build_injected(neutral_act: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """neutral_act + alpha * ||neutral_act|| * v_hat.

    alpha is a dimensionless ratio of injected norm to carrier norm.
    """
    return neutral_act + alpha * np.linalg.norm(neutral_act) * unit(v)


def cos_to(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def expected_cos(neutral_act: np.ndarray, v: np.ndarray, alpha: float) -> float:
    """cos(injected, v) — the quantity that actually saturates.

    Reported alongside alpha so a flattening recovery curve can be read as
    "the direction stopped changing" rather than misread as "the reader hit a
    ceiling". For neutral_act roughly orthogonal to v this is ~alpha/sqrt(1+alpha^2),
    which is already 0.89 at alpha=2 and 0.97 at alpha=4.
    """
    return cos_to(build_injected(neutral_act, v, alpha), v)
