# Manipulating redundancy inside one attribute

The three-attribute result was suggestive but confounded: comparing across attributes mixes redundancy with attribute identity, which is why two near-zero-redundancy points differed by 26 points. Here redundancy is manipulated instead of observed, with one attribute, one reader, one set of activations and 12 values. Only the content description changes.

| content level | redundancy | mentioning it | being correct | share | p |
|---|---|---|---|---|---|
| none | 3% | +0.0721 | **+0.0148** | +17% | 2.3e-12 |
| truncated | 58% | +0.0551 | **+0.0073** | +12% | 5.3e-08 |
| redacted | 83% | +0.0063 | **+0.0027** | +30% | 0.0004 |
| full | 85% | +0.0041 | **-0.0005** | -15% | 0.3 |

Spearman rho = **-1.00** (n=4) across 4 levels.

**Verdict:** the information-gain account HOLDS within a single attribute: as the content gives more of the answer away, being correct is worth less, with everything else held constant.

> Redundancy is measured at each level by giving a pinned judge the content alone and asking it to name the emotion. The redacted level masks emotion words so tone survives but the label cannot be read off directly.

> Four levels is few for a rank correlation; the ordering and the sign of the correctness column carry more than the p-value does.
