# How much signal does an LLM monitor need before it finds anything?

Our positive control has one rung — name the influence and the judge reaches 100% — which rules out incompetence but adds information that was not there. A control whose treatment adds information cannot measure sensitivity to information already present. This plants a marker of known strength instead and varies it.

| planted p | marked | available (classifier) | recovered (judge) |
|---|---|---|---|
| 1 | 100% | 0.956 | 55.0% |
| 0.6 | 61% | 0.942 | 50.0% |
| 0.3 | 30% | 0.931 | 75.0% |
| 0 | 0% | 0.920 | 40.0% |

**Verdict:** THE REAL FAILURE IS NOT ABOUT SIGNAL STRENGTH. The judge detects a planted cue once a classifier reaches 0.931, which is at or below where the real readouts already sit (0.920) — yet on those it scores 40%. So the information in the real readouts is present in a form this monitor cannot use, rather than merely being too faint.

> The marker is one ordinary word ("moreover") inserted after the first sentence. It carries no hint about influence — a marker that meant something would test comprehension rather than sensitivity — and it is exactly the kind of distributional cue a bag-of-words model thrives on and a reader skims past.

> Pairs are answer-matched, and the judge gets no hint in any row.

> p=0 is the unmodified readout, so its row is the real result and the rest of the curve says where that sits.
