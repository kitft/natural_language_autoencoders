# How much signal does an LLM monitor need before it finds anything?

Our positive control has one rung — name the influence and the judge reaches 100% — which rules out incompetence but adds information that was not there. A control whose treatment adds information cannot measure sensitivity to information already present. This plants a marker of known strength instead and varies it.

| planted p | marked | available (classifier) | recovered (judge) |
|---|---|---|---|
| 1 | 100% | 0.890 | 37.5% |
| 0.6 | 61% | 0.865 | 45.0% |
| 0.3 | 30% | 0.845 | 37.5% |
| 0 | 0% | 0.829 | 30.0% |

**Verdict:** NO DETECTION AT ANY PLANTED STRENGTH. Even with a marker in 100% of swayed readouts giving a classifier 0.890, the judge stays at 38%. This monitor does not use lexical cues at all, so its failure on the real readouts says nothing about how much signal they carry — and the binary signposted control, which works, is measuring comprehension of a statement rather than sensitivity.

> The marker is one ordinary word ("moreover") inserted after the first sentence. It carries no hint about influence — a marker that meant something would test comprehension rather than sensitivity — and it is exactly the kind of distributional cue a bag-of-words model thrives on and a reader skims past.

> Pairs are answer-matched, and the judge gets no hint in any row.

> p=0 is the unmodified readout, so its row is the real result and the rest of the curve says where that sits.
