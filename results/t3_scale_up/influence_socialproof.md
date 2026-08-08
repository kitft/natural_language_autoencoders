# Does a `socialproof` influence hide the same way a persona does?

The hidden-influence result rests on one socially-loaded manipulation, and the register cues found are exactly the vocabulary a persona introduces. This applies a different influence to the same kind of no-correct-answer claims.

| quantity | value |
|---|---|
| claims pooled | 200 |
| trials | 400 (2 draws per claim) |
| swayed | 92 (23%) |
| probe on activations | 0.867 |
| bag-of-words on readout text | 0.577 |
| persona-trained classifier applied here | 0.516 |

**Verdict:** BOUNDED. The information is in the activations (0.87) but does not reach the readout text (0.58), so the persona result is about socially-framed influence rather than the channel in general.

> Swayed is the SEMANTIC answer changing from the no-influence control AND changing toward what the influence pushes at. Answering the way the influence points is not evidence of influence if the model holds that view anyway.

> Folds are held out by CLAIM. Trials from one claim share wording, so a random split would let a classifier memorise the claim rather than the influence.

> Each Anthropic sycophancy file has only 32 unique claims — the ~10k rows are persona variants — so nlp_survey and philpapers are pooled and the influence is instantiated several times per claim.
