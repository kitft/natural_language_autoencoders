# What does the readout carry, if not the influence?

Two objections stand between the headline negative and a finding: that scarcer positives depress AUROC arithmetically, and that readouts carrying nothing about anything would make the negative trivial. This measures both.

| quantity | value |
|---|---|
| sway | 0.601 |
| topic retrieval | 0.182 |
| topic chance | 0.125 |
| input pushed | 0.814 |
| answer chosen | 0.992 |
| shuffled sway | 0.496 (sd 0.047, 20 shuffles) |

**Verdict:** THE FAILURE IS SPECIFIC, and it separates WHAT from WHY. The same readouts, on the same trials and folds, reveal which option the model chose at 0.992 while revealing what moved it to that choice at only 0.601. The channel is not uninformative — it describes the conclusion and not the cause.

> Specificity is decided by the ANSWER arm. Topic retrieval scores TF-IDF cosine between a readout and its question, so it fails across the paraphrase gap whenever the readout describes content without reusing the question's wording — that is a limitation of the metric, not a measurement of the channel, and its low value here should not be read as the readouts being empty.

> Topic retrieval ranks each readout's own question first among 8 candidates by TF-IDF cosine. No labels and no training, so there is nothing to overfit, and the floor is explicit.

> This is the judge-competence control applied one level down: a positive control for the SUBSTRATE rather than for the reader. Without it, 'the readout does not carry the influence' is not distinguishable from 'the readout does not carry anything'.
