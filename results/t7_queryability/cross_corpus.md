# Cross-corpus generalisation

The probe is trained on GoEmotions and never refitted. Both readers are then
scored on EmpatheticDialogues speaker text (180 passages, 9 emotions). In-distribution figures are the held-out
GoEmotions numbers from natural_comparison.md and prompt_elicitation.md.

| reader | GoEmotions (in-dist) | EmpatheticDialogues (shifted) | delta |
|---|---|---|---|
| linear probe | 64% | 26% | -38% |
| NLA (canonical) | 41% | 37% | -4% |
| NLA (emotion_focus) | 43% | 39% | -4% |
| NLA (forced_choice) | 45% | 42% | -3% |

> Only the speaker's own words are used; listener turns are empathetic
> responses from someone who does not hold the emotion. amusement, curiosity
> and confusion have no counterpart in this corpus and are absent.
