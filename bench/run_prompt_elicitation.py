"""Does the NLA know the emotion but not say it? Targeted-prompt elicitation.

THE BIAS THIS CORRECTS
  The AV's sidecar prompt asks it to "describe the semantic content of that
  activation vector" in 2-3 snippets. It answers with topic, register, format and
  next-token expectations — emotion is one attribute among many, and often not the
  one it picks. We then score "did it say anger". The linear probe, meanwhile, is
  asked exactly the question we grade, which is a large part of why it wins 64% to
  38% on natural activations.

  `NLAClient.generate(prompt=...)` accepts a custom user message containing the
  <INJECT> marker, so the AV can be asked about emotion directly.

  This may also HURT. The AV was RL-trained on the canonical template, so any custom
  prompt is off-distribution; the sidecar warns that template drift degrades output.
  A negative result here is informative and gets reported the same as a positive one.

NO CHERRY-PICKING
  All variants below are declared up front and ALL are reported, including any that
  underperform the canonical prompt. Reporting only the best of several elicitations
  would make any improvement meaningless.

USAGE
  python -m bench.run_prompt_elicitation <av_checkpoint>
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.emotions import EMOTIONS, NEUTRAL, build_passages
from bench.grade import dominant_emotion, grade_readout
from bench.inject import build_injected
from bench.roundtrip import free_model, verbalize_all

SEED = 0
TEST_PER_EMOTION = 20
INJECTED_ALPHA = 4.0
N_CARRIERS = 3

# Declared in advance. "canonical" is not re-run — its numbers already exist in
# natural_comparison.json — but it is the reference every variant is compared to.
PROMPTS: dict[str, tuple[str, int]] = {
    # Same task framing as the sidecar, but pointed at affect instead of "semantic
    # content". Deliberately close to the original wording to stay near-distribution.
    "emotion_focus": (
        "You are a meticulous AI researcher investigating activation vectors from a "
        "language model. Your task is to describe the EMOTIONAL TONE or affective "
        "state carried by that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. You "
        "must then produce an explanation enclosed within <explanation> tags, naming "
        "the emotion the vector expresses and the evidence for it.\n\n"
        "Here is the vector:\n\n<concept><INJECT></concept>\n\n"
        "Please provide an explanation.",
        160,
    ),
    # Forced 12-way choice — the closest possible match to what the probe is scored
    # on, and therefore the fairest head-to-head.
    "forced_choice": (
        "You are analysing an activation vector from a language model, passed to you "
        "inside <concept> tags.\n\n"
        "Here is the vector:\n\n<concept><INJECT></concept>\n\n"
        "Which SINGLE emotion from this list does the vector most express?\n"
        + ", ".join(EMOTIONS) + "\n\n"
        "Answer inside <explanation> tags with just the one word.",
        160,   # 40 truncated before the closing tag; the AV ignores "answer in one word"
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("av_checkpoint", nargs="?", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--acts-dir", default="results/trials",
                    help="directory holding emotion_activations.npz / emotion_vectors.npz")
    args = ap.parse_args()

    av_dir = args.av_checkpoint or glob.glob(
        str(Path.home() / ".cache/huggingface/hub/"
            "models--kitft--nla-qwen2.5-7b-L20-av/snapshots/*/"))[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # --out-dir is where results are WRITTEN; the cached activations are an INPUT and
    # live with the other trial data. Reading them from out_dir meant a reproduction
    # run writing to results_repro/ looked for its inputs there and died — the same
    # input-vs-output confusion that broke run_locate_loss.
    acts_dir = Path(args.acts_dir)
    acts = dict(np.load(acts_dir / "emotion_activations.npz"))
    vectors = dict(np.load(acts_dir / "emotion_vectors.npz"))
    rng = np.random.default_rng(SEED)

    # Same held-out split as run_natural_comparison, so canonical numbers transfer.
    jobs, vecs = [], []
    for e in EMOTIONS:
        for i in rng.permutation(len(acts[e]))[:TEST_PER_EMOTION]:
            jobs.append(("natural", e))
            vecs.append(acts[e][int(i)])

    # A few injected cells too, to check a custom prompt does not break the strong
    # injected result (93% under the canonical prompt).
    neutral_pool = acts[NEUTRAL][:N_CARRIERS]
    for e in EMOTIONS:
        for act in neutral_pool:
            jobs.append(("injected", e))
            vecs.append(build_injected(act, vectors[e], INJECTED_ALPHA))
    print(f"{len(vecs)} vectors x {len(PROMPTS)} prompts")

    from bench.av_local import LocalNLAClient
    av = LocalNLAClient(av_dir)

    rows = []
    for name, (prompt, max_new) in PROMPTS.items():
        print(f"\n--- prompt: {name} (max_new_tokens={max_new}) ---")
        texts = verbalize_all(av, vecs, max_new_tokens=max_new, prompt=prompt)
        for (task, e), t in zip(jobs, texts):
            rows.append({"prompt": name, "task": task, "emotion": e,
                         "hit": bool(grade_readout(t, e, method="lexicon")),
                         "dominant_is_target": dominant_emotion(t) == e,
                         "readout": t})
    free_model(av)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["prompt"], r["task"])].append(r)

    lines = ["| prompt | task | recovery | dominant==target |", "|---|---|---|---|",
             "| canonical | natural | 38% | 23% |",
             "| canonical | injected (a=4) | 93% | 83% |"]
    for (p, task), sel in sorted(agg.items()):
        lines.append(f"| {p} | {task} | {np.mean([r['hit'] for r in sel]):.0%} | "
                     f"{np.mean([r['dominant_is_target'] for r in sel]):.0%} |")
    table = "\n".join(lines)
    print(f"\n{table}\n\nlinear probe on the same natural split: 64%")

    (out_dir / "prompt_elicitation.md").write_text(
        "# Targeted-prompt elicitation\n\n"
        "Does the NLA carry the emotion but not volunteer it under the sidecar's\n"
        "generic 'describe the semantic content' framing? Canonical rows are the\n"
        "previously measured baseline on the same held-out split.\n\n"
        "Reference: linear probe 64% on this split (supervised, forced 12-way).\n\n"
        + table + "\n\n"
        "> All declared variants are reported, including any below canonical. A custom\n"
        "> prompt is off-distribution for the RL-trained AV, so a drop is a real\n"
        "> possibility and not a reason to drop the variant from the table.\n"
    )
    (out_dir / "prompt_elicitation.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {out_dir / 'prompt_elicitation.md'}")


if __name__ == "__main__":
    main()
