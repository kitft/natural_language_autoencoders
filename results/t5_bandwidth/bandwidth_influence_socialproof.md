# Is the probe winning only because it is a wider channel?

The headline negative compares a probe on thousands of continuous dimensions against a bag-of-words model on a paragraph. If the influence genuinely occupies many dimensions, that comparison is unfair and the finding is an artefact. This measures how much of the activation the influence actually occupies.

| probe dimensions | AUROC |
|---|---|
| 1 | 0.765 |
| 2 | 0.782 |
| 5 | 0.812 |
| 10 | 0.796 |
| 25 | 0.821 |
| 100 | 0.844 |
| 500 | 0.848 |
| 3584 | 0.859 |
| readout text (for comparison) | 0.577 |

**Verdict:** BANDWIDTH IS NOT THE EXPLANATION. A ONE-dimensional projection of the activation predicts sway at 0.765, above the whole readout's 0.577. The influence is essentially a scalar, and a readout that transmits the model's chosen option at 0.98 plainly has room for a scalar. What it drops is not too big to carry.

> Dimensions are selected INSIDE each fold. Ranking on the full dataset and then cross-validating on it leaks labels and inflates every small-k point, which would manufacture the conclusion this control exists to test.

> The yardstick is deliberate: the same readouts carry the model's chosen option at 0.98 and the pushed option at 0.93-0.96, both categorical facts. If sway is a scalar, the readout transmits harder things than the one it drops.
