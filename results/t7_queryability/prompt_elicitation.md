# Targeted-prompt elicitation

Does the NLA carry the emotion but not volunteer it under the sidecar's
generic 'describe the semantic content' framing? Canonical rows are the
previously measured baseline on the same held-out split.

Reference: linear probe 64% on this split (supervised, forced 12-way).

| prompt | task | recovery | dominant==target |
|---|---|---|---|
| canonical | natural | 38% | 23% |
| canonical | injected (a=4) | 93% | 83% |
| emotion_focus | injected | 92% | 89% |
| emotion_focus | natural | 43% | 26% |
| forced_choice | injected | 100% | 97% |
| forced_choice | natural | 45% | 27% |

> All declared variants are reported, including any below canonical. A custom
> prompt is off-distribution for the RL-trained AV, so a drop is a real
> possibility and not a reason to drop the variant from the table.
