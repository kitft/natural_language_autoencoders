"""Three attacks on the 88%: leakage, interpretability, and transfer.

WHY THE 88% IS SUSPECT
  The text classifier's 0.880 came from random 5-fold CV over 318 readouts with up to
  5000 TF-IDF features. Each claim appears about ten times, so a random split trains and
  tests on readouts of the SAME claim — near-duplicates. That is leakage, and it inflates
  the number the whole finding rests on.

  If the honest figure is near the few-shot judge's 68%, then classifier and reader agree
  and there is no residual gap to explain. The finding would reduce to "you need labels",
  which is much weaker than what has been claimed.

THE THREE CHECKS
  leakage    split by CLAIM (GroupKFold), so train and test share no claim, while still
             scoring within answer strata. This is the number that should be reported
  features   what is the classifier actually reading? We know it is not explicit
             mentions — Qwen's swayed readouts reference an opinion LESS often. If the
             top features are interpretable the finding gains a mechanism; if they are
             arbitrary function words it is a stylistic fingerprint and weaker
  transfer   train on one reader's readouts, test on the other's. A shared signal is a
             property of NLA readouts; no transfer means another checkpoint-specific
             artefact, which is what everything else here turned out to be

USAGE
  python -m bench.run_classifier_scrutiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 0


def load(trials_dir: Path, sfx: str):
    t = json.loads((trials_dir / f"sycophancy_influence_raw{sfx}.json").read_text())["trials"]
    return ([" ".join((x["readout"] or "").split()) for x in t],
            np.array([bool(x["swayed"]) for x in t]),
            np.array([x["answer"] or "?" for x in t]),
            np.array([x["claim"] for x in t]))


def pipe():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=2000, class_weight="balanced"))


def cv_scored_within(texts, y, score_groups, split_groups=None) -> dict:
    """CV AUROC scored within score_groups; split by split_groups if given."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    if y.sum() < 8 or (~y).sum() < 8:
        return {"auroc": float("nan"), "n": 0}
    pred = np.zeros(len(y))
    if split_groups is None:
        splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(texts, y)
    else:
        splits = GroupKFold(n_splits=5).split(texts, y, groups=split_groups)
    for tr, te in splits:
        clf = pipe().fit([texts[i] for i in tr], y[tr])
        pred[te] = clf.predict_proba([texts[i] for i in te])[:, 1]
    aucs, ns = [], []
    for g in np.unique(score_groups):
        s = score_groups == g
        if y[s].sum() < 4 or (~y[s]).sum() < 4:
            continue
        aucs.append(roc_auc_score(y[s], pred[s])); ns.append(int(s.sum()))
    if not aucs:
        return {"auroc": float("nan"), "n": 0}
    return {"auroc": float(np.average(aucs, weights=ns)), "n": int(sum(ns)),
            "strata": len(aucs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--trials-dir", default="results/trials",
                    help="directory holding the sycophancy_influence_raw*.json trials")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # --out-dir is WRITTEN to; the trials are an INPUT and live with the other trial
    # data. Reading them from out_dir meant a reproduction run writing to
    # results_repro/ looked for its inputs there and died before producing anything.
    trials_dir = Path(args.trials_dir)

    data = {}
    for m, sfx in (("qwen", ""), ("gemma", "_gemma")):
        p = trials_dir / f"sycophancy_influence_raw{sfx}.json"
        if p.exists():
            data[m] = load(trials_dir, sfx)

    res = {}
    for m, (texts, y, ans, claim) in data.items():
        print(f"\n=== {m} ===")
        leaky = cv_scored_within(texts, y, ans)                       # random split
        honest = cv_scored_within(texts, y, ans, split_groups=claim)  # claim held out
        res[m] = {"random_split": leaky, "claim_held_out": honest}
        print(f"  random split (leaky)   {leaky['auroc']:.3f}")
        print(f"  CLAIM HELD OUT         {honest['auroc']:.3f}  "
              f"({honest['strata']} answer strata)")

        # ── what is it reading? ────────────────────────────────────────────
        clf = pipe().fit(texts, y)
        vec, lr = clf.steps[0][1], clf.steps[1][1]
        names = np.array(vec.get_feature_names_out())
        coef = lr.coef_[0]
        top_s = names[np.argsort(coef)[-15:]][::-1]
        top_u = names[np.argsort(coef)[:15]]
        res[m]["top_swayed_features"] = list(top_s)
        res[m]["top_unswayed_features"] = list(top_u)
        print(f"  top 'swayed' terms  : {', '.join(top_s[:10])}")
        print(f"  top 'unswayed' terms: {', '.join(top_u[:10])}")

    # ── transfer between readers ──────────────────────────────────────────
    if len(data) == 2:
        from sklearn.metrics import roc_auc_score
        for src, dst in (("qwen", "gemma"), ("gemma", "qwen")):
            xs, ys, _, _ = data[src]
            xd, yd, ad, _ = data[dst]
            clf = pipe().fit(xs, ys)
            pr = clf.predict_proba(xd)[:, 1]
            aucs, ns = [], []
            for g in np.unique(ad):
                s = ad == g
                if yd[s].sum() < 4 or (~yd[s]).sum() < 4:
                    continue
                aucs.append(roc_auc_score(yd[s], pr[s])); ns.append(int(s.sum()))
            a = float(np.average(aucs, weights=ns)) if aucs else float("nan")
            res.setdefault("transfer", {})[f"{src}->{dst}"] = a
            print(f"\n  transfer {src} -> {dst}: {a:.3f} (answer-stratified)")

    lines = ["| model | random split (leaky) | **claim held out** | few-shot judge | "
             "zero-shot judge |", "|---|---|---|---|---|"]
    JUDGE = {"qwen": (0.68, 0.40), "gemma": (0.57, 0.37)}
    for m in data:
        f, z = JUDGE[m]
        lines.append(f"| {m} | {res[m]['random_split']['auroc']:.3f} | "
                     f"**{res[m]['claim_held_out']['auroc']:.3f}** | {f:.0%} | {z:.0%} |")
    table = "\n".join(lines)
    print(f"\n{table}")

    honest_q = res.get("qwen", {}).get("claim_held_out", {}).get("auroc", float("nan"))
    if honest_q == honest_q and honest_q < 0.72:
        verdict = ("the 88% was LEAKAGE. With claims held out the classifier lands near "
                   "the few-shot judge, so there is no residual gap: a supervised "
                   "classifier and a supervised reader do about equally well, and the "
                   "finding is that neither works without labels")
    elif honest_q == honest_q and honest_q > 0.80:
        verdict = ("the classifier survives claim-held-out validation, so the gap to the "
                   "few-shot judge is real and not an artefact of near-duplicate readouts")
    else:
        verdict = "partial shrinkage; read the table"
    print(f"\nverdict: {verdict}")

    (out_dir / "classifier_scrutiny.md").write_text(
        "# Three attacks on the classifier's 88%\n\n"
        "The figure the finding rests on came from random 5-fold CV, but each claim "
        "appears about ten times, so a random split trains and tests on readouts of the "
        "same claim. This holds claims out, inspects what the classifier reads, and tests "
        "whether the signal transfers between readers.\n\n" + table + "\n\n"
        f"**Verdict:** {verdict}.\n\n"
        "## What the classifier reads\n\n"
        + "\n".join(f"- **{m} → swayed**: {', '.join(res[m]['top_swayed_features'][:10])}\n"
                    f"- **{m} → unswayed**: {', '.join(res[m]['top_unswayed_features'][:10])}"
                    for m in data) + "\n\n"
        + ("## Transfer between readers\n\n"
           + "\n".join(f"- {k}: {v:.3f}" for k, v in res.get("transfer", {}).items())
           + "\n\n" if "transfer" in res else "")
        + "> Judge figures are pairwise accuracy, which equals AUROC on two-alternative "
        "forced choice, so all columns are the same measurement.\n"
    )
    (out_dir / "classifier_scrutiny.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out_dir / 'classifier_scrutiny.md'}")


if __name__ == "__main__":
    main()
