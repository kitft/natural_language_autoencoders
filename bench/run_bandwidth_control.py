"""Is the probe beating the readout only because it is a much wider channel?

THE OBJECTION
  The headline negative compares a linear probe on 3584 continuous activation dimensions
  against a bag-of-words classifier on roughly 700 characters of text. A reviewer says: of
  course the wide channel wins, and "the readout drops the causal link" is really "text is
  a narrower pipe than a residual stream". If that is right, the finding is an artefact of
  the comparison rather than a fact about verbalisation.

THE TEST
  Measure how much of the activation the influence actually occupies. Sweep the number of
  dimensions available to the probe — selected on TRAINING folds only — and see where its
  AUROC saturates.

    a 1-D projection already predicts sway   the information is a single scalar, and 700
                                             characters has room for a scalar many times
                                             over, so bandwidth cannot be the explanation
    performance needs hundreds of dims       the influence is genuinely high-dimensional
                                             and the objection has force; the claim must be
                                             restated as being about capacity

  Feature selection happens INSIDE each fold. Ranking dimensions on the full dataset and
  then cross-validating on the same data leaks the labels and inflates every small-k point,
  which would manufacture exactly the conclusion this control is meant to test.

WHY A SCALAR IS THE RIGHT YARDSTICK
  The readout carries the model's chosen option at 0.98 and the pushed option at 0.93-0.96
  — both categorical facts, each worth at least a bit. If sway is also recoverable from a
  one-dimensional projection at comparable AUROC, then the readout demonstrably transmits
  harder things than the one it drops.

USAGE
  python -m bench.run_bandwidth_control --model qwen
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model", default="qwen", choices=["qwen", "gemma"])
    ap.add_argument("--raw", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[1, 2, 5, 10, 25, 100, 500])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sfx = "" if args.model == "qwen" else f"_{args.model}"
    raw = Path(args.raw) if args.raw else Path("results/trials") / f"sycophancy_influence_raw{sfx}.json"
    trials = json.loads(raw.read_text())["trials"]
    y = np.array([bool(t["swayed"]) for t in trials])
    groups = np.array([t["claim"] for t in trials])

    # activations are not stored with the trials; recapture them the same way the run did
    from bench.capture import capture_activations, load_model
    from bench.roundtrip import free_model
    from bench.run_controlled_claims import MODELS
    cfg = MODELS[args.model]
    model, tok = load_model(cfg["base"])
    prompts = [tok.apply_chat_template([{"role": "user", "content": t["prompt"]}],
                                       add_generation_prompt=True, tokenize=False)
               for t in trials]
    acts = capture_activations(model, tok, prompts, block_index=cfg["block"],
                               max_length=1024).astype(np.float64)
    free_model(model); del model
    print(f"{args.model}: {acts.shape[0]} trials x {acts.shape[1]} dims, "
          f"{y.mean():.0%} swayed")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = GroupKFold(n_splits=5)
    res = {}
    for k in args.dims + [acts.shape[1]]:
        if k > acts.shape[1]:
            continue
        # SelectKBest inside the pipeline => refit per fold, so no label leakage
        pipe = make_pipeline(StandardScaler(),
                             SelectKBest(f_classif, k=min(k, acts.shape[1])),
                             LogisticRegression(max_iter=3000))
        auc = float(cross_val_score(pipe, acts, y, cv=cv, groups=groups,
                                    scoring="roc_auc").mean())
        res[str(k)] = auc
        print(f"  probe on {k:>5} dims   {auc:.3f}")

    txt = [" ".join((t.get("readout") or "").split()) for t in trials]
    readout = float(cross_val_score(
        make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                      LogisticRegression(max_iter=2000)),
        txt, y, cv=cv, groups=groups, scoring="roc_auc").mean())
    print(f"  readout text          {readout:.3f}")

    one = res.get("1", float("nan"))
    full = res[str(acts.shape[1])]
    print(f"\n  1-D probe {one:.3f} vs full probe {full:.3f} vs readout {readout:.3f}")

    # The objection is about CAPACITY, so the test is whether the influence occupies many
    # dimensions — not how the 1-D probe compares to the readout. If one dimension already
    # captures nearly all of what the full probe gets, the signal is a scalar and no
    # capacity argument can explain a paragraph failing to carry it.
    share = one / full if full else float("nan")
    print(f"  1-D captures {share:.0%} of the full probe")
    if share > 0.9:
        verdict = (f"BANDWIDTH IS NOT THE EXPLANATION. ONE dimension captures {share:.0%} "
                   f"of what all {acts.shape[1]} dimensions achieve ({one:.3f} against "
                   f"{full:.3f}), so the influence is essentially a scalar. A readout that "
                   "transmits the model's chosen option at 0.98 has room for a scalar many "
                   f"times over; here it carries the influence at {readout:.3f}. What it "
                   "drops is not too big to carry")
    elif share < 0.75:
        verdict = (f"THE OBJECTION HAS FORCE. One dimension captures only {share:.0%} of "
                   f"the full probe, so the influence is genuinely distributed and "
                   "comparing thousands of dimensions against a paragraph is not obviously "
                   "fair. The claim should be restated in terms of capacity")
    elif one >= readout + 0.05:
        verdict = (f"BANDWIDTH IS NOT THE EXPLANATION. A ONE-dimensional projection of the "
                   f"activation predicts sway at {one:.3f}, above the whole readout's "
                   f"{readout:.3f}. The influence is essentially a scalar, and a readout "
                   "that transmits the model's chosen option at 0.98 plainly has room for a "
                   "scalar. What it drops is not too big to carry")
    elif one < readout:
        verdict = (f"THE OBJECTION HAS FORCE. A 1-D projection reaches only {one:.3f}, "
                   f"below the readout's {readout:.3f}, so the influence is not a scalar "
                   "and comparing 3584 dimensions against a paragraph is not obviously "
                   "fair. The claim should be restated in terms of capacity")
    else:
        verdict = (f"PARTIAL: 1-D captures {share:.0%} of the full probe; between the "
                   "thresholds, so read the table")
    print(f"\nverdict: {verdict}")

    lines = "\n".join(f"| {k} | {v:.3f} |" for k, v in res.items())
    # each corpus needs its own file: three runs previously overwrote one
    lab = args.label or (f"bandwidth_{Path(raw).stem.replace('_raw','')}{sfx}")
    (out_dir / f"{lab}.md").write_text(
        "# Is the probe winning only because it is a wider channel?\n\n"
        "The headline negative compares a probe on thousands of continuous dimensions "
        "against a bag-of-words model on a paragraph. If the influence genuinely occupies "
        "many dimensions, that comparison is unfair and the finding is an artefact. This "
        "measures how much of the activation the influence actually occupies.\n\n"
        f"| probe dimensions | AUROC |\n|---|---|\n{lines}\n"
        f"| readout text (for comparison) | {readout:.3f} |\n\n"
        f"**Verdict:** {verdict}.\n\n"
        "> Dimensions are selected INSIDE each fold. Ranking on the full dataset and then "
        "cross-validating on it leaks labels and inflates every small-k point, which would "
        "manufacture the conclusion this control exists to test.\n\n"
        "> The yardstick is deliberate: the same readouts carry the model's chosen option "
        "at 0.98 and the pushed option at 0.93-0.96, both categorical facts. If sway is a "
        "scalar, the readout transmits harder things than the one it drops.\n")
    (out_dir / f"{lab}.json").write_text(json.dumps(
        {"probe_by_dims": res, "readout": readout}, indent=2))
    print(f"\nwrote {out_dir / f'{lab}.md'}")


if __name__ == "__main__":
    main()
