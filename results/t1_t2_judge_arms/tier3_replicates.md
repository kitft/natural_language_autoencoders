# Judge arms as intervals

The judge tier does not reproduce as point estimates: the CLI's model resolution is not stable, and two runs of the identical script moved one arm from 70.0% to 80.0%. Replicates convert that into a checkable claim — a fresh run should land inside the interval, not on the number.

Each replicate reseeds pair selection, so **items and judge sampling both vary**. Holding items fixed would measure judge noise alone and understate the spread.

### judge ladder — qwen (5 replicates)

| arm | mean [min–max] |
|---|---|
| subtle | 40.0% [30.0%–60.0%] |
| cued | 79.0% [75.0%–85.0%] |
| signposted | 100.0% [100.0%–100.0%] |

Ordering subtle ≤ cued ≤ signposted held in **5/5** replicates.

### monitor sensitivity — qwen (5 replicates)

| planted p | judge recovery, mean [min–max] |
|---|---|
| 1 | 43.0% [30.0%–55.0%] |
| 0.6 | 41.0% [37.5%–50.0%] |
| 0.3 | 40.5% [30.0%–52.5%] |
| 0 | 41.0% [32.5%–45.0%] |

At p=1 the marker is perfectly diagnostic — any reader noticing it scores 100%. Observed: **43.0% [30.0%–55.0%]**.

### judge ladder — gemma (5 replicates)

| arm | mean [min–max] |
|---|---|
| subtle | 58.5% [47.5%–72.5%] |
| cued | 64.5% [62.5%–70.0%] |
| signposted | 100.0% [100.0%–100.0%] |

Ordering subtle ≤ cued ≤ signposted held in **4/5** replicates.

### monitor sensitivity — gemma (5 replicates)

| planted p | judge recovery, mean [min–max] |
|---|---|
| 1 | 70.5% [57.5%–82.5%] |
| 0.6 | 61.0% [52.5%–67.5%] |
| 0.3 | 62.0% [52.5%–67.5%] |
| 0 | 59.5% [47.5%–70.0%] |

At p=1 the marker is perfectly diagnostic — any reader noticing it scores 100%. Observed: **70.5% [57.5%–82.5%]**.

> Read the ordering and the interval. The mean is not meaningful to three decimals, which is precisely what this file exists to stop anyone quoting.

> The single-run numbers these replace are preserved under `results/t1_t2_judge_arms/tier3_singlerun_backup/`.
