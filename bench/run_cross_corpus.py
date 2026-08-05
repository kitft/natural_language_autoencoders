"""Does the probe's advantage survive distribution shift?

THE QUESTION
  On held-out GoEmotions the linear probe beats the NLA 64% to 41-45%. But the probe
  was trained on that exact corpus and construction, while the NLA is zero-shot. The
  interesting property of a reader is generalisation: a probe fitted to one
  distribution's class means is expected to be brittle, whereas a reader that
  genuinely decodes emotion should transfer.

  So: keep the probe trained on GoEmotions, and test BOTH readers on
  EmpatheticDialogues. Whichever degrades less is the more useful reader.

  If the NLA holds while the probe collapses, that is the contribution. If the probe
  transfers too, that is a real negative result and gets reported as one.

CORPUS
  Estwld/empathetic_dialogues_llm, grouped by conversation. Only the SPEAKER's own
  words are used — the `situation` field plus the `user` turns. The `assistant` turns
  are empathetic responses from a listener who does not hold the emotion, so
  including them would dilute the signal. Speaker-only text is shorter (p50 43
  tokens), which is why the situation line is prepended.

MAPPING
  9 of our 12 emotions have counterparts. amusement, curiosity and confusion have
  none, so they are absent from this test — the probe can still predict them, and a
  wrong prediction counts as wrong.

USAGE
  python -m bench.run_cross_corpus
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.capture import capture_activations, load_model
from bench.emotions import EMOTIONS
from bench.grade import dominant_emotion, grade_readout
from bench.run_prompt_elicitation import PROMPTS

SEED = 0
N_PER_EMOTION = 20
MIN_TOKENS = 40

# ED label -> our label. Conservative: only near-synonyms. `caring` -> love and
# `impressed` -> admiration are the loosest and are flagged in the write-up.
MAPPING = {
    "angry": "anger", "furious": "anger",
    "joyful": "joy",
    "sad": "sadness", "devastated": "sadness",
    "afraid": "fear", "terrified": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "grateful": "gratitude",
    "impressed": "admiration",
    "caring": "love",
}


def build_ed_passages(n_per_emotion: int = N_PER_EMOTION, seed: int = SEED):
    import warnings
    warnings.filterwarnings("ignore")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train")

    pool = defaultdict(list)
    for row in ds:
        ours = MAPPING.get(row["emotion"])
        if ours is None:
            continue
        speaker = " ".join(t["content"] for t in row["conversations"]
                           if t["role"] == "user")
        text = f"{row['situation'].strip()} {speaker}".strip()
        if len(tok(text)["input_ids"]) >= MIN_TOKENS:
            pool[ours].append(text)

    rng = np.random.default_rng(seed)
    out = {}
    for e, texts in pool.items():
        idx = rng.permutation(len(texts))[:n_per_emotion]
        out[e] = [texts[int(i)] for i in idx]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--block-index", type=int, default=20)
    ap.add_argument("--acts-dir", default="results/trials",
                    help="directory holding emotion_activations.npz")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # --out-dir is where results are WRITTEN; the cached activations are an INPUT and
    # live with the other trial data. Reading them from out_dir meant a reproduction
    # run writing to results_repro/ looked for its inputs there and died — the same
    # input-vs-output confusion that broke run_locate_loss.
    acts = dict(np.load(Path(args.acts_dir) / "emotion_activations.npz"))

    print("building EmpatheticDialogues passages ...")
    ed = build_ed_passages()
    print("  " + ", ".join(f"{e}={len(v)}" for e, v in sorted(ed.items())))
    covered = sorted(ed)

    # ── probe trained on GoEmotions ONLY ─────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([acts[e] for e in EMOTIONS])
    y = np.concatenate([[i] * len(acts[e]) for i, e in enumerate(EMOTIONS)])
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000)).fit(X, y)

    print("capturing EmpatheticDialogues activations ...")
    model, tok = load_model()
    texts, golds = [], []
    for e in covered:
        texts.extend(ed[e]); golds.extend([e] * len(ed[e]))
    ed_acts = capture_activations(model, tok, texts, block_index=args.block_index)

    pred = probe.predict(ed_acts)
    probe_hits = [EMOTIONS[int(p)] == g for p, g in zip(pred, golds)]
    print(f"  probe cross-corpus: {np.mean(probe_hits):.1%}")

    # ── NLA on the same activations, both prompts ────────────────────────────
    from bench.av_local import LocalNLAClient
    from bench.roundtrip import free_model, verbalize_all
    free_model(model); del model
    av_dir = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                           "models--kitft--nla-qwen2.5-7b-L20-av/snapshots/*/"))[0]
    av = LocalNLAClient(av_dir)

    rows = []
    for i, (t, g, ph) in enumerate(zip(texts, golds, probe_hits)):
        rows.append({"emotion": g, "text": t, "probe_hit": bool(ph)})
    for pname, (prompt, max_new) in [("canonical", (None, 220))] + \
            [(k, v) for k, v in PROMPTS.items()]:
        print(f"  NLA [{pname}] ...")
        outs = verbalize_all(av, list(ed_acts), max_new_tokens=max_new, prompt=prompt)
        for r, o in zip(rows, outs):
            r[f"nla_{pname}_hit"] = bool(grade_readout(o, r["emotion"], method="lexicon"))
            r[f"nla_{pname}_dom"] = dominant_emotion(o) == r["emotion"]
            r[f"nla_{pname}_readout"] = o
    free_model(av)

    # ── report: in-distribution vs cross-corpus ──────────────────────────────
    IN_DIST = {"probe": 0.637, "canonical": 0.41, "emotion_focus": 0.43,
               "forced_choice": 0.45}
    lines = ["| reader | GoEmotions (in-dist) | EmpatheticDialogues (shifted) | delta |",
             "|---|---|---|---|"]
    summary = {}
    cross = {"probe": float(np.mean(probe_hits))}
    for pname in ("canonical", "emotion_focus", "forced_choice"):
        cross[pname] = float(np.mean([r[f"nla_{pname}_hit"] for r in rows]))
    for k, label in (("probe", "linear probe"), ("canonical", "NLA (canonical)"),
                     ("emotion_focus", "NLA (emotion_focus)"),
                     ("forced_choice", "NLA (forced_choice)")):
        d = cross[k] - IN_DIST[k]
        summary[k] = {"in_dist": IN_DIST[k], "cross": cross[k], "delta": d}
        lines.append(f"| {label} | {IN_DIST[k]:.0%} | {cross[k]:.0%} | {d:+.0%} |")
    table = "\n".join(lines)
    print(f"\n{table}")
    print(f"\nn = {len(rows)} passages across {len(covered)} emotions; chance 8%")

    (out_dir / "cross_corpus.md").write_text(
        "# Cross-corpus generalisation\n\n"
        "The probe is trained on GoEmotions and never refitted. Both readers are then\n"
        f"scored on EmpatheticDialogues speaker text ({len(rows)} passages, "
        f"{len(covered)} emotions). In-distribution figures are the held-out\n"
        "GoEmotions numbers from natural_comparison.md and prompt_elicitation.md.\n\n"
        + table + "\n\n"
        "> Only the speaker's own words are used; listener turns are empathetic\n"
        "> responses from someone who does not hold the emotion. amusement, curiosity\n"
        "> and confusion have no counterpart in this corpus and are absent.\n"
    )
    (out_dir / "cross_corpus.json").write_text(json.dumps(
        {"summary": summary, "covered": covered, "rows": rows}, indent=2))
    print(f"\nwrote {out_dir / 'cross_corpus.md'}")


if __name__ == "__main__":
    main()
