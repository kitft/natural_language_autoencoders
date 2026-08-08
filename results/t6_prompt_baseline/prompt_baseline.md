# Does the readout carry anything the prompt does not?

2509.13316 shows verbalization benchmarks are often solvable without the target model's internals. If a classifier does as well on the raw prompt as on the readout, then "the information is in the readout" is a statement about the prompt template rather than about the activation.

| substrate | claim-held-out | + answer-stratified |
|---|---|---|
| readout | 0.829 | 0.842 |
| prompt | 0.649 | 0.930 |
| reasoning | 0.766 | 0.846 |
| shuffled readout | 0.425 | 0.480 |
| answer alone | 0.375 | in-sample 0.510 |

readout − prompt = **+0.180**; readout − reasoning = **+0.062**; readout − answer alone = **+0.454**.

**Verdict:** PREMISE HOLDS. The readout (0.829) beats the raw prompt (0.649) by +0.180 on identical labels and claim-held-out folds, so the classifier is not simply reading the prompt through the readout, and the sway signal is genuinely carried by the description of the activation.

> Same labels, same claim-held-out folds for every arm. Trials sharing a claim share nearly all their wording, so a random split lets a classifier memorise the claim — the defect that inflated an earlier figure here to 0.933 against a 0.931 shortcut baseline.

> The reasoning arm is the practical bound: it is the substrate an auditor already has without any NLA at all.
