# Is the probe winning only because it is a wider channel?

The headline negative compares a probe on thousands of continuous dimensions against a bag-of-words model on a paragraph. If the influence genuinely occupies many dimensions, that comparison is unfair and the finding is an artefact. This measures how much of the activation the influence actually occupies.

| probe dimensions | AUROC |
|---|---|
| 1 | 0.847 |
| 2 | 0.856 |
| 5 | 0.871 |
| 10 | 0.872 |
| 25 | 0.887 |
| 100 | 0.894 |
| 500 | 0.892 |
| 3584 | 0.893 |
| readout text (for comparison) | 0.829 |

**Verdict:** BANDWIDTH IS NOT THE EXPLANATION. ONE dimension captures 95% of what all 3584 dimensions achieve (0.847 against 0.893), so the influence is essentially a scalar. A readout that transmits the model's chosen option at 0.98 has room for a scalar many times over; here it carries the influence at 0.829. What it drops is not too big to carry.

> Dimensions are selected INSIDE each fold. Ranking on the full dataset and then cross-validating on it leaks labels and inflates every small-k point, which would manufacture the conclusion this control exists to test.

> The yardstick is deliberate: the same readouts carry the model's chosen option at 0.98 and the pushed option at 0.93-0.96, both categorical facts. If sway is a scalar, the readout transmits harder things than the one it drops.
