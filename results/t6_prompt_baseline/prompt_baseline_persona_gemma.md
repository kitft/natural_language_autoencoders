# Does the readout carry anything the prompt does not?

2509.13316 shows verbalization benchmarks are often solvable without the target model's internals. If a classifier does as well on the raw prompt as on the readout, then "the information is in the readout" is a statement about the prompt template rather than about the activation.

| substrate | claim-held-out | + answer-stratified |
|---|---|---|
| readout | 0.626 | nan |
| prompt | 0.510 | nan |
| shuffled readout | 0.557 | nan |
| answer alone | 0.608 | in-sample 0.943 |

readout − prompt = **+nan**; readout − reasoning = **+nan**; readout − answer alone = **+0.018**.

**Verdict:** UNTESTABLE ON THIS READER. The answer letter alone predicts sway at 0.943, so the comparison needs answer stratification — but within an answer value one class nearly vanishes, so no AUROC is defined. Answer, claim and sway are mutually entangled here and this dataset cannot separate them.

> Same labels, same claim-held-out folds for every arm. Trials sharing a claim share nearly all their wording, so a random split lets a classifier memorise the claim — the defect that inflated an earlier figure here to 0.933 against a 0.931 shortcut baseline.

> The reasoning arm is the practical bound: it is the substrate an auditor already has without any NLA at all.
