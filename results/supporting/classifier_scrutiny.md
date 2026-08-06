# Three attacks on the classifier's 88%

The figure the finding rests on came from random 5-fold CV, but each claim appears about ten times, so a random split trains and tests on readouts of the same claim. This holds claims out, inspects what the classifier reads, and tests whether the signal transfers between readers.

| model | random split (leaky) | **claim held out** | few-shot judge | zero-shot judge |
|---|---|---|---|---|
| qwen | 0.880 | **0.839** | 68% | 40% |
| gemma | 0.717 | **0.673** | 57% | 37% |

**Verdict:** the classifier survives claim-held-out validation, so the gap to the few-shot judge is real and not an artefact of near-duplicate readouts.

## What the classifier reads

- **qwen → swayed**: your, on the, on, analysis, neutral, balanced, perspective, the given, classify, position
- **qwen → unswayed**: positive, correct, choice is, agree with, agree, choice, is, indeed, is indeed, formal response
- **gemma → swayed**: text, provided text, you, should, you should, should select, text the, verdict requiring, or you, based on
- **gemma → unswayed**: recommendation, the established, answer or, or answer, or recommendation, description, answer, requiring the, recommendation or, likely answer

## Transfer between readers

- qwen->gemma: 0.732
- gemma->qwen: 0.668

> Judge figures are pairwise accuracy, which equals AUROC on two-alternative forced choice, so all columns are the same measurement.
