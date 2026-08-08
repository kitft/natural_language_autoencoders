# Is the probe winning only because it is a wider channel?

The headline negative compares a probe on thousands of continuous dimensions against a bag-of-words model on a paragraph. If the influence genuinely occupies many dimensions, that comparison is unfair and the finding is an artefact. This measures how much of the activation the influence actually occupies.

| probe dimensions | AUROC |
|---|---|
| 1 | 0.490 |
| 2 | 0.567 |
| 5 | 0.579 |
| 10 | 0.584 |
| 25 | 0.652 |
| 100 | 0.670 |
| 500 | 0.656 |
| 3584 | 0.683 |
| readout text (for comparison) | 0.601 |

**Verdict:** THE OBJECTION HAS FORCE. One dimension captures only 72% of the full probe, so the influence is genuinely distributed and comparing thousands of dimensions against a paragraph is not obviously fair. The claim should be restated in terms of capacity.

> Dimensions are selected INSIDE each fold. Ranking on the full dataset and then cross-validating on it leaks labels and inflates every small-k point, which would manufacture the conclusion this control exists to test.

> The yardstick is deliberate: the same readouts carry the model's chosen option at 0.98 and the pushed option at 0.93-0.96, both categorical facts. If sway is a scalar, the readout transmits harder things than the one it drops.
