# Natural Language Autoencoders (NLA) — faithfulness audit

**Write-up: [Natural Language Autoencoders can verbalize influence but not whether it changed the outcome](https://jaisood.substack.com/p/natural-language-autoencoders-can)**

This branch is an audit of two released NLA checkpoints. We influence a model, verbalize the
activation behind its answer, and ask whether the readout reports three things: which way the
influence pushed, which option the model chose, and whether the influence changed that choice.
The readouts answer the first two and not the third.

Everything in the write-up is reproduced by the code and artifacts on this branch.
[REPRODUCE.md](REPRODUCE.md) covers the environment, pinned checkpoint SHAs, and what each tier
does or does not reproduce exactly.

## Reproduce everything

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-bench.txt
bash reproduce.sh 1                # analysis only, no GPU and no API key
bash reproduce.sh 2                # + activations: GPU and the released checkpoints
bash reproduce.sh 3                # + judge arms: an authenticated `claude` CLI
python compare_repro.py results_repro
```

`compare_repro.py` diffs every number produced against the committed `results/` and names
anything that moved further than its tier allows. Tier 1 should be exact.

## Reproduce one result at a time

Each row regenerates the numbers behind one section of the write-up. All of them write to
`results_repro/` and leave the committed `results/` untouched, so you can diff afterwards.
`$V` is `.venv/bin/python`.

| Result in the write-up | Tier | Command |
|---|---|---|
| Both models moved by influence | 1 | rates are counted from the committed trial sets in `results/trials/` |
| The readout carries influence and answer but not cause | 1 | `$V -m bench.run_readout_capacity --raw results/trials/sycophancy_influence_raw.json --out-dir results_repro` (repeat for the other four trial files) |
| What the readout adds over knowing the answer | 1 | `$V -m bench.run_prompt_baseline --model qwen --raw results/trials/influence_persona_raw.json --out-dir results_repro --label prompt_baseline_persona` |
| The activation held what the readout left out | 2 | `$V -m bench.run_bandwidth_control --model qwen --raw results/trials/influence_socialproof_raw.json --out-dir results_repro` |
| What the readout carries that the question does not | 1 | same `run_prompt_baseline` call as above, once per trial file and checkpoint |
| Why the information is missing | 2 | `$V -m bench.run_locate_loss --model qwen --raw results/trials/sycophancy_influence_raw.json --out-dir results_repro` |
| The judge misses a signal we planted | 3 | `$V -m bench.run_judge_competence --model qwen --raw results/trials/sycophancy_influence_raw.json --out-dir results_repro` then `$V -m bench.run_monitor_sensitivity --model qwen --raw <same> --out-dir results_repro` |

Two supporting controls, both tier 1:

```bash
$V -m bench.run_classifier_scrutiny --trials-dir results/trials --out-dir results_repro
$V -m bench.run_readout_signal      --trials-dir results/trials --out-dir results_repro
```

Every script takes `--model {qwen,gemma}`. Pass `--raw` explicitly: without it `--out-dir`
doubles as the input directory and the run fails looking for trials it has not written yet.

## Reproduce from the checkpoints, not from the committed trials

The commands above read the trial sets in `results/trials/`. To regenerate those from the
checkpoints themselves, which takes hours and needs a GPU:

```bash
FULL=1 bash reproduce.sh all
```

This re-runs both models over both corpora, recaptures activations, and regenerates every
readout before any analysis. The sycophancy trials reproduce field-for-field because decoding
is greedy and seeded. The GlobalOpinionQA battery samples its claims, so those trials will
differ in composition while the reported statistics hold.

## Where the outputs live

`results/` is one folder per test, numbered in the order the tests were run. Everything a
script *reads* lives in `results/trials/`; everything it *writes* goes to the matching folder.
[REPRODUCE.md](REPRODUCE.md) has the full map.

---


Open-source library accompanying the Anthropic Transformer Circuits post
**[Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html)**.

📄 [Blog post](https://www.anthropic.com/research/natural-language-autoencoders) · ▶ [Video walkthrough](https://www.youtube.com/watch?v=j2knrqAzYVY) · 🔬 [Try the released NLAs on Neuronpedia](https://www.neuronpedia.org/nla)

---

A Natural Language Autoencoder is a pair of fine-tuned LMs that map
residual-stream activation vectors to natural language and back:

| | direction | mechanism |
|---|---|---|
| **AV** (activation verbalizer) | `vector → text` | inject the vector as a single token embedding into a fixed prompt, autoregress a description |
| **AR** (activation reconstructor) | `text → vector` | truncated K+1-layer LM + `Linear(d, d)` head, extract at the final token |

Both vectors are L2-normalised before comparison, so the round-trip
`MSE(reconstructed, original) = 2(1 − cos)` measures direction agreement only.
Low MSE means the AR could recover the original direction from the AV's words
alone, which implies the explanation captures the information in the vector.

This is the **full training repo** — data generation, SFT, GRPO RL, and
checkpoint conversion. For a lightweight inference-only package (just
`NLAClient` + `NLACritic`, no training deps), see
[`kitft/nla-inference`](https://github.com/kitft/nla-inference).

> **A note on naming.** Public-facing names are **AV** / **AR**. Inside the
> `nla/` package you will see **actor** / **critic** — those are the same two
> models, named to map directly onto Miles' RL primitives (the AV *is* the
> policy actor; the AR *is* the value critic). The codebase keeps actor/critic
> so the Miles extension points read naturally; everywhere user-facing we use
> AV/AR.

---

## Released checkpoints

All eight checkpoints are gathered in the
**[`kitft/nla-models` collection](https://huggingface.co/collections/kitft/nla-models)**
on the HF Hub — four base-model families, each with an AV and an AR. We extract
from a layer roughly **two-thirds of the way through the model** in each case
— deep enough that the residual stream carries rich semantic content, shallow
enough that it hasn't yet collapsed toward the unembedding.

| base model | layer | d_model | AV | AR |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 20 / 28 | 3584 | [`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av) | [`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar) |
| Gemma-3-12B-IT | 32 / 48 | 3840 | [`kitft/nla-gemma3-12b-L32-av`](https://huggingface.co/kitft/nla-gemma3-12b-L32-av) | [`kitft/nla-gemma3-12b-L32-ar`](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar) |
| Gemma-3-27B-IT | 41 / 62 | 5376 | [`kitft/nla-gemma3-27b-L41-av`](https://huggingface.co/kitft/nla-gemma3-27b-L41-av) | [`kitft/nla-gemma3-27b-L41-ar`](https://huggingface.co/kitft/nla-gemma3-27b-L41-ar) |
| Llama-3.3-70B-Instruct | 53 / 80 | 8192 | [`kitft/Llama-3.3-70B-NLA-L53-av`](https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-av) | [`kitft/Llama-3.3-70B-NLA-L53-ar`](https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-ar) |

Each checkpoint ships an `nla_meta.yaml` sidecar with the prompt template,
injection token IDs, and scale factors that the model was trained with — load
those, never hardcode them.

---

## How it fits together

NLA training is built as a thin extension on top of two open-source projects:

- **[Miles](https://github.com/radixark/miles)** — Ray-orchestrated RL training
  (FSDP2 / Megatron backends, GRPO, async rollout). We used the FSDP backend
  for the 7B/12B/27B runs and Megatron only for Llama-70B. NLA plugs in via Miles'
  upstream `--custom-rm-path`, `--data-source-path`, and
  `--custom-generate-function-path` extension points; the integration patch in
  `nla/miles_patches/` adds `--custom-actor-cls-path` and `--force-use-critic`
  on top (see [docs/design.md §2](docs/design.md)).
- **[SGLang](https://github.com/sgl-project/sglang)** — rollout serving. We
  send `input_embeds` (not `input_ids`) so the AV sees the injected vector;
  SGLang serves it like any other request. The embed sequence is built on the
  **trainer side** — we look up the prompt tokens in the actor's own embedding
  table, splice the activation vector in at the injection slot, and ship the
  finished `[seq, d]` tensor over HTTP. SGLang never needs to know what an
  injection is. We don't apply any learned map to the injected vector in this
  work — it goes in raw (after a fixed scalar `injection_scale`) — but this
  design means a future affine `W·v + b` adapter would be a trainer-side-only
  change: apply it before sending, no SGLang modification required. (vLLM also
  supports `input_embeds` and would work as a drop-in alternative.)

We chose this stack because it is **near-frontier training infrastructure**:
Miles + Megatron is what production-scale RL post-training looks like, and
hooking onto it cleanly is what let us scale to RL-ing a 70B-parameter AV — and
likely further. The `nla/` package never modifies Miles or SGLang in place; it
only subclasses and registers function-pointer hooks, so upstream updates pull
in cleanly.

---

## Quick start

### Inference (use a released checkpoint)

```bash
uv pip install torch transformers safetensors httpx orjson pyyaml numpy
uv pip install "sglang[all]>=0.5.6"

python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 --disable-radix-cache &

python nla_inference.py kitft/nla-qwen2.5-7b-L20-av \
    --sglang-url http://localhost:30000 \
    --parquet path/to/activations.parquet
```

Don't have a parquet yet? Any file with an `activation_vector` column of
`d_model`-wide float lists will do — here's a minimal one for Qwen layer 20:

```python
import torch, pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16, device_map="cuda")
ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to("cuda")
hs = m(**ids, output_hidden_states=True).hidden_states[20][0]  # [seq, 3584]
pq.write_table(pa.table({"activation_vector": hs.float().cpu().tolist()}), "demo.parquet")
```

(Or omit `--parquet` entirely for a smoke test on a random unit vector.)

`nla_inference.py` is a single self-contained file. The full recipe —
model-specific scale factors, the Gemma `√d` embed-scale gotcha, debugging the
"output is in Chinese" failure mode, AR scoring — is in
**[docs/inference.md](docs/inference.md)**. Worked transcripts in
[`examples/`](examples/).

### Training (reproduce a checkpoint)

Install Miles + SGLang + this package per **[docs/setup.md](docs/setup.md)**,
then run the three stages (Qwen7B reference: SFT on 2×H100-80GB; RL to ~75% FVE
on 2×8×H100 — see [`configs/TRAINING_NOTES.md`](configs/TRAINING_NOTES.md)):

```bash
# 0. Generate data (GPU + ANTHROPIC_API_KEY)
python -m nla.datagen.run_pipeline --config configs/datagen/qwen7b_fineweb_1M.yaml

# 1. AR SFT (MSE on raw activations)
bash configs/critic_sft.sh

# 2. AV SFT (next-token on API-generated explanations, with injection)
bash configs/actor_sft.sh

# 3. RL: simultaneous AV (GRPO) + AR (supervised); reward = -mse_nrm
bash configs/rl.sh
```

The full design — data transport through Miles' `multimodal_train_inputs`, the
injection forward-hook, simultaneous AV/AR scheduling, why `cp_size==1` — is in
**[docs/design.md](docs/design.md)**. Detailed profiling and hyperparameter
notes (Qwen7B case study; we reused those settings with only light adjustment
for the other models — a per-model sweep would likely do better):
[`configs/TRAINING_NOTES.md`](configs/TRAINING_NOTES.md).

---

## Repo layout

```
nla/                  core package
  schema.py, config.py, models.py     — sidecar contract, NLACriticModel (the AR)
  train_actor.py                      — NLAFSDPActor (Miles FSDP subclass)
  megatron/                           — NLAMegatronActor (TP+PP, CP=1 only)
  rollout/                            — SFT rollout, nla_generate (SGLang input_embeds)
  reward.py, loss.py                  — -mse_nrm reward, AR MSE loss
  datagen/                            — 4-stage activation → parquet pipeline
configs/              training shell configs + datagen YAMLs
scripts/              multi-GPU launch wrappers (datagen)
patches/              SGLang training patches (bf16 transport, chunked-prefill) + apply script
tools/                FSDP-DCP / Megatron-dist ↔ HF checkpoint converters
docs/                 design.md (training), inference.md (serving)
release/              HF model-card templates + sidecar sanitiser for releases
nla_inference.py      standalone single-file inference client
examples/             worked decode transcripts
```

---

## Citation

For attribution in academic contexts, please cite this work as

> Fraser-Taliente, Kantamneni, Ong et al., "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", Transformer Circuits, 2026.

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and Mossing, Dan and Lu, Christina and Bogdan, Paul C. and Ameisen, Emmanuel and Chen, James and Kishylau, Dzmitry and Pearce, Adam and Tarng, Julius and Wu, Alex and Wu, Jeff and Zhang, Yang and Ziegler, Daniel M. and Hubinger, Evan and Batson, Joshua and Lindsey, Jack and Zimmerman, Samuel and Marks, Samuel},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Released checkpoints additionally inherit the
license of their base model (Gemma, Llama-3.3) — see the NOTICE files in each
HF repo.
