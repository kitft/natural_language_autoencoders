# Positive control: can the judge see influence when it is signposted?

The hidden-influence claim rests on a judge scoring at chance on readouts that a bag-of-words classifier reads at 0.88-0.93 AUROC. Chance is equally consistent with "nothing marks the signal" and with "this judge cannot do the task". This separates them by giving the same judge the same real readouts with one sentence added that names the influence.

| arm | judge accuracy | n | p vs chance |
|---|---|---|---|
| signposted | **100.0%** | 40 | 1.82e-12 |
| cued (text unchanged, cues named) | **72.5%** | 40 | 0.00643 |
| subtle (unchanged) | 60.0% | 40 | 0.268 |

**Verdict:** JUDGE IS COMPETENT, SIGNAL IS UNSIGNPOSTED. The same judge, same format, same real readouts scores 100% when one sentence names the influence and 60% without it. So chance performance on real readouts is not a limitation of the reader — nothing in the text marks the influence as something to look for, which is exactly the position of an auditor trying to DISCOVER an unknown influence.

> The signpost names only that the user's stated opinion moved the model. It never names the answer or the direction — a signpost that gave the answer away would test reading comprehension instead of signposting.

> Marking is added to REAL readouts, so the only difference between arms is the presence of a signpost. Unparseable replies count as wrong, so an arm cannot score well by declining to answer.
