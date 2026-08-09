# Reproducing the results

```bash
bash reproduce.sh 1            # analysis only — no GPU, no API key
bash reproduce.sh 2            # + activations (GPU + released checkpoints)
bash reproduce.sh 3            # + judge arms (authenticated `claude` CLI)
FULL=1 bash reproduce.sh all   # also regenerates the trials themselves (hours)

python compare_repro.py results_repro
```

`compare_repro.py` diffs every numeric leaf against the committed `results/`, with a
per-tier tolerance, and names anything that moved more than it should.

## Three tiers, because they do not reproduce equally

| tier | needs | determinism |
|---|---|---|
| **1 — analysis** | nothing beyond the Python env | **exact.** Pure computation over committed trial JSONs; any drift is a real difference |
| **2 — activations** | GPU + the four NLA checkpoints | **near-exact.** Recaptures activations, so expect movement in the last decimal or two. The arms that regenerate readouts seed their decode and reproduce exactly |
| **3 — judge** | authenticated `claude` CLI | **weak.** See below — this tier does not reproduce, and pretending otherwise would be dishonest |

The write-up's central claim — endpoints 0.93–0.996 vs link 0.58–0.60, and its base-rate and
prompt controls — is **entirely tier 1**. The mechanism (`locate_loss`) and the bandwidth
control are tier 2. Only the methods contribution depends on tier 3.

## Where the inputs live

`results/` is one folder per test, numbered in the order the tests were run:

| folder | what produced it |
|---|---|
| `trials/` | **inputs only** — the trial sets every test reads, plus the cue source for the cued arm |
| `t1_t2_judge_arms/` | `run_judge_competence`, `run_monitor_sensitivity`, pooled by `aggregate_tier3` |
| `t3_scale_up/` | `run_influence_battery` |
| `t4_endpoints_vs_link/` | `run_readout_capacity` |
| `t5_bandwidth/` | `run_bandwidth_control` |
| `t6_prompt_baseline/` | `run_prompt_baseline` |
| `t7_queryability/` | `run_cross_corpus`, `run_prompt_elicitation`, `run_prompted_readout` |
| `t8_locate_loss/` | `run_locate_loss` |
| `supporting/` | `run_classifier_scrutiny`, `run_readout_signal`, `run_redundancy_sweep` |

Everything a script *reads* (trial JSONs, the cue source, the cached emotion activations) sits
under `results/trials/`; everything a script *writes* goes to the matching folder above.
That separation is load-bearing rather than cosmetic: `reproduce.sh` writes to
`results_repro/`, so any script resolving an input against its own output directory would go
looking for trials in a folder that has none.

## The Test 7 arms regenerate readouts, and still reproduce

`run_cross_corpus`, `run_prompt_elicitation` and `run_prompted_readout` differ from the rest
of tier 2: they regenerate readouts through the AV rather than analysing committed ones. That
might sound non-reproducible, but `verbalize_all` decodes **greedily** by default
(`temperature=0.0`, and `av_local.py` only samples above zero). A rerun reproduced all 552
`prompt_elicitation` rows byte-for-byte, readout text included, and `cross_corpus` matched
exactly.

They still cost far more wall-clock than the analysis steps, roughly ten minutes per
verbalisation arm, which is the real reason to run them separately.

## Tier 3 does not reproduce, and here is the specific reason

`bench/judge.py` pins `MODEL = "claude-opus-4-8"`, but the CLI's resolution of a model name
is **not stable**: its own header records a probe of `claude-opus-5` returning
`claude-haiku-4-5-20251001`, evidently a transient capacity fallback, which later resolved
differently. So the judge behind a given run is not fully determined by the code.

Observed consequence, same script and same seed, two runs: Qwen's cued arm read **70.0%**
then **80.0%**, and its subtle arm **50.0%** then **37.5%**, while Gemma reproduced exactly.
Judge arms are n=40.

**What to take from a tier-3 run:** the *ordering* of the arms (subtle < cued < signposted),
which was stable across every run we did, and the p=1.0 result in `monitor_sensitivity`
(a perfectly diagnostic planted word leaving both judges near chance). Do not expect the
point estimates to land where the committed files have them, and do not read a few points of
movement as a discrepancy.

## Pinned environment

The numbers in `results/` were produced with:

```
python 3.11.15
torch          2.7.1+cu118      # cu118 build; this box is on driver 470
transformers   4.57.6
scikit-learn   1.9.0            # AUROC / GroupKFold — a version change can move digits
numpy          2.4.6
scipy          1.17.1
datasets       5.0.1
accelerate     1.14.0
safetensors    0.8.0
claude CLI     2.1.205
GPU            NVIDIA RTX A6000 (48 GB)
```

Install (torch must come from the cu118 index on this hardware, or you get an unusable cu12
build):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e . -r requirements-bench.txt
```

`scikit-learn` matters more than it looks: every headline number is a cross-validated AUROC,
so a version change can move the last digits without anything being wrong.

## Pinned checkpoints

Tier 2 needs these from the Hugging Face hub. The SHAs are the snapshots the committed
results were produced with:

| checkpoint | snapshot |
|---|---|
| `kitft/nla-qwen2.5-7b-L20-av` | `b88469162777ae6553bc14208eb0cb579336f8f4` |
| `kitft/nla-qwen2.5-7b-L20-ar` | `e2c9e57eac213d37a31612087f645ab6332c1bb6` |
| `kitft/nla-gemma3-12b-L32-av` | `7aec22599e8a9cd533564868999b443fcc963cf4` |
| `kitft/nla-gemma3-12b-L32-ar` | `3d6901d8243182d642af9dea452ca91549a94615` |

Base models for activation capture: `Qwen/Qwen2.5-7B-Instruct` (layer 20) and
`unsloth/gemma-3-12b-it` (layer 32).

```bash
huggingface-cli download kitft/nla-qwen2.5-7b-L20-av --revision b8846916...
```

> **Known sharp edge.** `run_controlled_claims.py` resolves the AV directory with
> `glob(~/.cache/huggingface/hub/<name>/snapshots/*/)[0]`. With one snapshot cached that is
> deterministic; with several it silently picks whichever sorts first. If you have pulled
> more than one revision, check which one you are on.

## Regenerating the trials themselves

`FULL=1` goes all the way back to the checkpoints. It regenerates the sycophancy trials
(`run_sycophancy_influence`, both checkpoints) and the GlobalOpinionQA battery
(`run_influence_battery`) — prompts, model answers, activations and readouts. Tier 3 then
relabels sway on them with `run_sycophancy_judge`.

Every generation step in that chain is deterministic: trial selection is seeded (`SEED = 0`),
the base model answers with `do_sample=False`, and the AV verbalises greedily. So a from-
scratch run should match the committed trials rather than merely resemble them. The caveat is
hardware, not method — greedy decoding takes an argmax, and GPU float nondeterminism can flip
a near-tied token, so expect close agreement rather than a guarantee of byte-equality across
different cards or library versions.

The pipeline is still split this way because the generation steps cost hours of GPU time and
the analysis tiers cost seconds. The committed trials let a reviewer check every claim without
paying for regeneration first.

## Two CLI edges worth knowing

- **Pass the input flag explicitly.** Scripts take `--raw` (a trial JSON) or `--acts-dir`
  (cached activations). The defaults point at `results/trials/`, but passing them makes the
  dependency visible instead of implicit — and every place this project conflated the input
  directory with `--out-dir`, a reproduction run silently looked for its trials in the folder
  it was writing to.
- `reproduce.sh` writes to `results_repro/` by default and never overwrites `results/`, so a
  failed reproduction cannot damage the baseline it is being compared against.
