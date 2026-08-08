"""Does the READOUT carry anything the prompt does not already carry?

THE OBJECTION, WHICH ATTACKS THE PREMISE RATHER THAN THE FRAMING
  Our claim is that "was swayed" survives into the NLA readout text — a bag-of-words
  classifier reads it at 0.88 AUROC — while a judge reading that same text is at chance.
  [2509.13316](https://arxiv.org/abs/2509.13316) shows verbalization benchmarks are often
  solvable WITHOUT the target model's internals: the verbalizer's output reflects the
  prompt it was implicitly describing rather than anything read off the activation.

  If a classifier does as well on the raw PROMPT as on the readout, then "the information
  is in the readout" is a statement about the prompt template, not about the activation or
  the sway, and the whole dissociation is uninteresting.

THE ARMS, all with the same labels and the same folds
    readout    the NLA's description of the activation — the claim's substrate
    prompt     the text the model was answering, persona bio included
    reasoning  the model's own chain of thought for the same trial, which is the
               substrate an auditor already has without any NLA
    shuffled   readouts permuted against labels, so the pipeline's floor is visible

  Folds are held out by CLAIM. Trials sharing a claim share almost all their wording, so a
  random split lets a classifier memorise the claim instead of the influence — the same
  defect that inflated an earlier number in this project to 0.933 against a 0.931 shortcut.

WHAT EACH OUTCOME MEANS
  readout > prompt        the readout carries sway-specific signal the prompt does not, and
                          the premise holds
  readout ~ prompt        the classifier is reading the prompt through the readout; the
                          claim reduces to a property of the template and must be withdrawn
  reasoning >= readout    an auditor gains nothing from the NLA over the model's own words,
                          which bounds the practical claim regardless of the above

USAGE
  python -m bench.run_prompt_baseline --model qwen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model", default="qwen", choices=["qwen", "gemma"])
    ap.add_argument("--raw", default=None, help="trials json; defaults to sycophancy")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    sfx = "" if args.model == "qwen" else f"_{args.model}"
    raw = Path(args.raw) if args.raw else Path("results/trials") / f"sycophancy_influence_raw{sfx}.json"
    trials = json.loads(raw.read_text())["trials"]
    lab = args.label or f"prompt_baseline{sfx}"
    print(f"trials from {raw.name}")
    y = np.array([bool(t["swayed"]) for t in trials])
    groups = np.array([t["claim"] for t in trials])

    def txt(field):
        return [" ".join((t.get(field) or "").split()) for t in trials]

    arms = {"readout": txt("readout"), "prompt": txt("prompt"),
            "reasoning": txt("reasoning")}
    arms["shuffled readout"] = [arms["readout"][i] for i in rng.permutation(len(trials))]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    print(f"{args.model}: {len(trials)} trials, {int(y.sum())} swayed, "
          f"{len(set(groups))} claims; folds held out by claim")
    # ANSWER-STRATIFIED as well as claim-held-out. Gemma's readouts leak the final answer
    # (swayed answer "B" 91% of the time, unswayed "A" 84%), and the prompt does not contain
    # the answer at all — so an unstratified readout-vs-prompt gap can be entirely that
    # leak. Scoring within each answer value and averaging removes it. readout_signal.md
    # established this shortcut for exactly this model; not applying it here would repeat
    # the error the shortcut baseline was built to catch.
    # battery runs store the chosen OPTION text rather than a letter
    ans = np.array([str(t.get("answer") or t.get("answer_opt")) for t in trials])
    def scored(X):
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                             LogisticRegression(max_iter=2000))
        overall = float(cross_val_score(pipe, X, y, cv=GroupKFold(n_splits=5),
                                        groups=groups, scoring="roc_auc").mean())
        per = []
        for a in sorted(set(ans)):
            m = ans == a
            if m.sum() < 40 or len(set(y[m])) < 2 or len(set(groups[m])) < 5:
                continue
            per.append(float(cross_val_score(
                pipe, [x for x, k in zip(X, m) if k], y[m],
                cv=GroupKFold(n_splits=5), groups=groups[m],
                scoring="roc_auc").mean()))
        return overall, (float(np.mean(per)) if per else float("nan"))

    res, res_strat = {}, {}
    for name, X in arms.items():
        if not any(x.strip() for x in X):
            print(f"  {name:18s} EMPTY — field missing from this run")
            continue
        auc, strat = scored(X)
        res[name], res_strat[name] = auc, strat
        print(f"  {name:18s} claim-held-out {auc:.3f}   "
              f"+answer-stratified {strat:.3f}")

    # WHICH figure decides depends on how entangled the answer is with sway, and the two
    # available controls fight each other. Holding claims out leaves the answer free;
    # stratifying on the answer makes CLAIM near-deterministic for sway (this project has
    # already verified that every claim has a constant sway label within an answer), and the
    # prompt names the claim — which is why the prompt "improves" to 0.930 under
    # stratification. That is the claim shortcut, not prompt signal.
    #
    # So: use the answer oracle to decide. If the answer alone cannot predict sway, the
    # claim-held-out number needs no answer stratification and is the honest one.
    from sklearn.metrics import roc_auc_score
    try:
        # generalised beyond A/B: score each answer value by its sway rate
        rate = {a: float(y[ans == a].mean()) for a in set(ans)}
        oracle = float(roc_auc_score(y, np.array([rate[a] for a in ans])))
        oracle = max(oracle, 1 - oracle)
    except Exception:
        oracle = float("nan")
    print(f"\n  answer-letter-alone oracle for 'was swayed': {oracle:.3f}")

    # THE ANSWER AS A BASELINE, NOT AS A DIAGNOSTIC.
    #   The figure above fits the per-answer sway rate on the very rows it scores, so it says
    #   how entangled answer and sway are in this sample and nothing more. It cannot be
    #   subtracted from the readout: the readout is scored on held-out claims and is therefore
    #   answering a strictly harder question, so the difference would flatter the readout by
    #   however much of the entanglement was claim-specific.
    #   Scoring the answer as its own arm, on the same folds, is the comparison that means
    #   something. readout minus answer is then a like-for-like gap.
    #
    #   The answer is CATEGORICAL, so it is one-hot encoded rather than run through the
    #   word-level vectorizer the text arms use. Two reasons. The sycophancy answers are bare
    #   letters, which the default token pattern discards entirely, leaving an empty vocabulary
    #   and no score at all. And on the datasets whose answers are full option strings, a word
    #   model reads the option's WORDING, which is more than "the auditor knows what the model
    #   answered" — it would inflate the baseline with content the answer alone does not give.
    try:
        pipe = make_pipeline(
            TfidfVectorizer(analyzer=lambda s: [s]),  # whole answer as one categorical token
            LogisticRegression(max_iter=2000))
        oracle_ho = float(cross_val_score(pipe, list(ans), y, cv=GroupKFold(n_splits=5),
                                          groups=groups, scoring="roc_auc").mean())
    except Exception:
        oracle_ho = float("nan")
    print(f"  the answer as an arm, claim-held-out:        {oracle_ho:.3f}")
    entangled = oracle > 0.6
    if entangled:
        print("  answer and sway are entangled on this reader; answer-stratified figures "
              "are the relevant ones, but stratifying leaves too few of one class to score")
    ro = res_strat.get("readout") if entangled else res.get("readout", float("nan"))
    pr = res_strat.get("prompt") if entangled else res.get("prompt", float("nan"))
    rs = res_strat.get("reasoning") if entangled else res.get("reasoning", float("nan"))
    ro, pr, rs = (float("nan") if v is None else v for v in (ro, pr, rs))
    print(f"\n  readout minus prompt:    {ro - pr:+.3f}")
    print(f"  readout minus reasoning: {ro - rs:+.3f}")

    if entangled and not (np.isfinite(ro) and np.isfinite(pr)):
        verdict = (f"UNTESTABLE ON THIS READER. The answer letter alone predicts sway at "
                   f"{oracle:.3f}, so the comparison needs answer stratification — but "
                   "within an answer value one class nearly vanishes, so no AUROC is "
                   "defined. Answer, claim and sway are mutually entangled here and this "
                   "dataset cannot separate them")
    elif not np.isfinite(ro) or not np.isfinite(pr):
        verdict = "INCOMPLETE: an arm was missing from this run"
    elif ro - pr > 0.05:
        verdict = (f"PREMISE HOLDS. The readout ({ro:.3f}) beats the raw prompt "
                   f"({pr:.3f}) by {ro - pr:+.3f} on identical labels and claim-held-out "
                   "folds, so the classifier is not simply reading the prompt through the "
                   "readout, and the sway signal is genuinely carried by the description "
                   "of the activation")
    elif abs(ro - pr) <= 0.05:
        verdict = (f"WITHDRAW. The prompt alone scores {pr:.3f} against the readout's "
                   f"{ro:.3f}, so nothing shows the readout carries sway-specific "
                   "information. The dissociation would then be a fact about the prompt "
                   "template, which is exactly the failure mode 2509.13316 describes")
    else:
        verdict = (f"WITHDRAW, more sharply: the prompt ({pr:.3f}) beats the readout "
                   f"({ro:.3f})")
    print(f"\nverdict: {verdict}")

    lines = "\n".join(f"| {k} | {v:.3f} | {res_strat.get(k, float('nan')):.3f} |"
                      for k, v in res.items())
    (out_dir / f"{lab}.md").write_text(
        "# Does the readout carry anything the prompt does not?\n\n"
        "2509.13316 shows verbalization benchmarks are often solvable without the target "
        "model's internals. If a classifier does as well on the raw prompt as on the "
        "readout, then \"the information is in the readout\" is a statement about the "
        "prompt template rather than about the activation.\n\n"
        f"| substrate | claim-held-out | + answer-stratified |\n|---|---|---|\n{lines}\n"
        f"| answer alone | {oracle_ho:.3f} | in-sample {oracle:.3f} |\n\n"
        f"readout − prompt = **{ro - pr:+.3f}**; readout − reasoning = "
        f"**{ro - rs:+.3f}**; readout − answer alone = "
        f"**{res.get('readout', float('nan')) - oracle_ho:+.3f}**.\n\n"
        f"**Verdict:** {verdict}.\n\n"
        "> Same labels, same claim-held-out folds for every arm. Trials sharing a claim "
        "share nearly all their wording, so a random split lets a classifier memorise the "
        "claim — the defect that inflated an earlier figure here to 0.933 against a 0.931 "
        "shortcut baseline.\n\n"
        "> The reasoning arm is the practical bound: it is the substrate an auditor "
        "already has without any NLA at all.\n")
    (out_dir / f"{lab}.json").write_text(json.dumps(
        {"claim_held_out": res, "answer_stratified": res_strat,
         "answer_only": {"in_sample": oracle, "claim_held_out": oracle_ho}}, indent=2))
    print(f"\nwrote {out_dir / f'{lab}.md'}")


if __name__ == "__main__":
    main()
