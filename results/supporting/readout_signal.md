# Is the influence absent from the readout, or present but unusable?

A judge cannot separate swayed from unswayed readouts on answer-matched pairs. That is consistent with the readout carrying nothing, and with it carrying the signal in a form no reader would use. A bag-of-words classifier settles which.

| model | text probe (answer-strat.) | shortcut | text probe (claim-strat.) | shortcut | judge |
|---|---|---|---|---|---|
| qwen | **0.880** | 0.287 | **0.876** | 0.441 | 34% |
| gemma | **0.717** | 0.447 | **0.933** | 0.931 | 52% |

**Verdict:** mixed across readers; read the table.

> The headline text-probe column is answer-stratified: fit across all trials but scored within groups that gave the same final answer, where answer and sway are decoupled. The unstratified score is reported in the JSON and is confounded, because being swayed means answering differently from baseline.

> The two stratified columns remove the two confounds separately: the answer, because being swayed means answering differently from baseline, and the claim, because sway rates differ by claim and readouts describe content. Removing both at once is not possible here — claim-by-answer cells hold about five trials, below the minimum per class — so each is controlled singly.

> The last column is the direct check that the judge did not simply miss an explicit mention. On MMLU swayed readouts mentioned an opinion LESS often than unswayed ones, 1 of 16 against 27 of 224.
