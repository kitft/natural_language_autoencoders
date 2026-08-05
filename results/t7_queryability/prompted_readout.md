# Can't see it, or wasn't asked?

The default readout did not let a judge separate swayed from unswayed runs that produced the same answer, even though a probe reaches 0.824 AUROC on the same activations. Patchscopes and LatentQA get their results by ASKING, and the released AV accepts a custom prompt, so the same vector can be queried directly.

| measure | value |
|---|---|
| trials (balanced) | 60 |
| control: topic query overlaps the DEFAULT readout | 0.43 |
| control: influence query overlaps the DEFAULT readout | 0.38 |
| floor: default vs a DIFFERENT vector's default | 0.38 |
| usable yes/no answers on the influence query | 0 |
| says YES when the model WAS swayed | **nan%** |
| says YES when it was NOT swayed | **nan%** |
| accuracy | nan% (chance 50%, Fisher p=1) |
| probe on the same activations | 0.824 AUROC |

**Verdict: the AV is not queryable, so this question cannot be settled here.**

It never answered the question — 0 usable yes/no answers out of 60 — and under a
custom prompt its output falls to the floor of vector-specific content. The influence
query shares no more with the default readout of the SAME vector (0.38) than
two unrelated readouts share with each other (0.38). The topic query is barely above it
at 0.43.

So the prompt does not redirect the AV, it degrades it. RL training on a single template
has left this checkpoint able to emit descriptions and nothing else — it is not a
queryable decoder in the Patchscopes or LatentQA sense. The can't-see-it versus
wasn't-asked question stays open, and answering it needs a checkpoint that accepts
queries.

> The control row is the precondition. The AV was RL-trained on one template, so any custom prompt is off-distribution and a null on the influence question could just mean the prompt broke the model. The topic query asks for something the default readout demonstrably does carry; if that fails, nothing below it is interpretable.

> Swayed and unswayed trials are balanced, so the yes-rate contrast is not driven by prevalence.
