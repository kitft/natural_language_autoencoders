"""Manipulate redundancy directly, within one attribute, and watch what truth is worth.

WHY THIS DESIGN RATHER THAN MORE ATTRIBUTES
  Three attributes gave a suggestive relationship — correctness is worth 47% of an
  assertion at 7% redundancy, 21% at ~0%, and -25% at 83% — but comparing ACROSS
  attributes confounds redundancy with attribute identity. That is exactly why the two
  low-redundancy points fail to separate: language at 21% and injected emotion at 47%
  are both near-zero redundancy yet differ by 26 points, so something other than
  redundancy is moving.

  Here redundancy is MANIPULATED instead of observed, with everything else held fixed:
  one attribute, one reader, one set of activations, twelve values. Only how much the
  content description gives away changes.

THE FOUR LEVELS
    full      the whole passage, so the emotion is plainly there
    redacted  the passage with emotion words masked, so tone survives but naming does not
    truncated the first few words only
    none      no content at all — "An activation from a language model"

  Redundancy is measured at each level by giving a pinned judge the content alone.

PRE-DECLARED
  If the value of being correct falls monotonically as redundancy rises, within a single
  attribute with everything else constant, the information-gain account holds and the
  earlier three-point result was not an artefact of comparing unlike attributes. If it
  is flat, redundancy is not the variable and the cross-attribute pattern was
  coincidence.

USAGE
  python -m bench.run_redundancy_sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from bench.grade import EMOTION_LEXICON
from bench.run_controlled_claims import MODELS

SEED = 0
TRUNC_WORDS = 8
_ALL_EMOTION_WORDS = sorted({w for ws in EMOTION_LEXICON.values() for w in ws},
                            key=len, reverse=True)
_MASK = re.compile(r"\b(" + "|".join(re.escape(w) for w in _ALL_EMOTION_WORDS) + r")\b",
                   re.IGNORECASE)

LEVELS = ("full", "redacted", "truncated", "none")
# language levels, graded by how much of the target-language text is shown. The English
# parallel text is the zero-redundancy floor: same content, no clue to the language.
LANG_LEVELS = ("full", "six_words", "two_words", "translated", "none")


def content_at(level: str, passage: str) -> str:
    stem = "The activation comes from a passage"
    if level == "none":
        return "An activation from a language model."
    if level == "truncated":
        return f"{stem} beginning: {' '.join(passage.split()[:TRUNC_WORDS])}"
    if level == "redacted":
        # mask emotion NAMES so tone survives but the label cannot simply be read off
        return f"{stem} that says: {_MASK.sub('___', passage)}"
    return f"{stem} that says: {passage}"


def lang_content_at(level: str, target: str, english: str) -> str:
    stem = "The activation comes from a passage"
    if level == "none":
        return "An activation from a language model."
    if level == "translated":
        return f"{stem} that says, in translation: {english}"
    if level == "two_words":
        return f"{stem} beginning: {' '.join(target.split()[:2])}"
    if level == "six_words":
        return f"{stem} beginning: {' '.join(target.split()[:6])}"
    return f"{stem} that says: {target}"


def language_rows(n: int, rng):
    from bench.flores_corpus import LANGUAGES, SENTENCES_PER_PASSAGE
    from datasets import load_dataset
    d = load_dataset("mteb/flores", "default", split="devtest")
    values = [v for v in LANGUAGES.values() if v != "english"]
    rows = []
    for col, name in LANGUAGES.items():
        if name == "english":
            continue
        for k in range(n // len(values) + 1):
            j = k * SENTENCES_PER_PASSAGE
            tgt = [x for x in d[col][j:j + SENTENCES_PER_PASSAGE] if x]
            eng = [x for x in d["eng_Latn"][j:j + SENTENCES_PER_PASSAGE] if x]
            if len(tgt) < SENTENCES_PER_PASSAGE or len(eng) < SENTENCES_PER_PASSAGE:
                break
            rows.append({"target": " ".join(tgt), "english": " ".join(eng),
                         "value": name})
    rng.shuffle(rows)
    return rows[:n], values


def run_language(args, out_dir, cfg, rng):
    rows, values = language_rows(args.n, rng)
    from bench.capture import capture_activations, load_model
    from bench.roundtrip import free_model
    model, tok = load_model(cfg["base"])
    acts = capture_activations(model, tok, [r["target"] for r in rows],
                               block_index=cfg["block"])
    free_model(model); del model
    for r in rows:
        others = [v for v in values if v != r["value"]]
        r["wrong"] = others[int(rng.integers(len(others)))]
    print(f"{len(rows)} passages, {len(values)} languages, chance {1/len(values):.0%}")

    import glob as _glob
    from nla_inference import NLACritic
    ar_dir = _glob.glob(str(Path.home() / ".cache/huggingface/hub" /
                            cfg["av"].replace("-av", "-ar") / "snapshots/*/"))[0]
    ar = NLACritic(ar_dir, device="cuda:0")
    stmt = lambda v: f"The text is written in {str(v).capitalize()}."

    res = {}
    for level in LANG_LEVELS:
        S = {"content": [], "assert": [], "wrong": []}
        for i, r in enumerate(rows):
            base = lang_content_at(level, r["target"], r["english"])
            for a, txt in (("content", base),
                           ("assert", f"{base} {stmt(r['value'])}"),
                           ("wrong", f"{base} {stmt(r['wrong'])}")):
                S[a].append(float(ar.score(txt, acts[i])[1]))
        S = {a: np.array(x) for a, x in S.items()}
        corr = float((S["assert"] - S["wrong"]).mean())
        pres = float((S["wrong"] - S["content"]).mean())
        p = float("nan")
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(S["assert"], S["wrong"]).pvalue)
        except Exception:
            pass
        res[level] = {"presence": pres, "correctness": corr, "p_correctness": p}
        print(f"  [{level:<11}] presence {pres:+.4f}  correctness {corr:+.4f}  p={p:.2g}")
    free_model(ar); del ar

    from bench.run_code_legibility import judge_language
    sub = list(rng.choice(len(rows), min(args.judge_n, len(rows)), replace=False))
    for level in LANG_LEVELS:
        g = judge_language([lang_content_at(level, rows[i]["target"],
                                            rows[i]["english"]) for i in sub], values)
        red = float(np.mean([x == rows[i]["value"] for x, i in zip(g, sub)]))
        res[level]["redundancy"] = red
        print(f"  [{level:<11}] redundancy {red:.0%}")

    order = sorted(LANG_LEVELS, key=lambda L: res[L]["redundancy"])
    lines = ["| content level | redundancy | mentioning it | being correct | p |",
             "|---|---|---|---|---|"]
    for L in order:
        r = res[L]
        lines.append(f"| {L} | {r['redundancy']:.0%} | {r['presence']:+.4f} | "
                     f"**{r['correctness']:+.4f}** | {r['p_correctness']:.2g} |")
    table = "\n".join(lines)
    x = np.array([res[L]["redundancy"] for L in LANG_LEVELS])
    y = np.array([res[L]["correctness"] for L in LANG_LEVELS])
    rho = float("nan")
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(x, y).statistic)
    except Exception:
        pass
    print(f"\n{table}\nrho={rho:+.2f}")
    (out_dir / "redundancy_sweep_language.md").write_text(
        "# Redundancy manipulated inside LANGUAGE\n\n"
        "The emotion sweep showed the value of being correct falling monotonically as "
        "the content gives more away. This repeats it on a second attribute in the same "
        "reader, using the English parallel text as the zero-redundancy floor: identical "
        f"content, no clue to the target language.\n\n{table}\n\n"
        f"Spearman rho = **{rho:+.2f}** across {len(LANG_LEVELS)} levels. Four or five "
        "points is very few for a rank correlation; the ordering and the per-level "
        "p-values carry the claim.\n")
    (out_dir / "redundancy_sweep_language.json").write_text(
        json.dumps({"levels": res, "spearman_rho": rho, "n": len(rows)}, indent=2))
    print(f"\nwrote {out_dir / 'redundancy_sweep_language.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--acts-dir", default="results/trials",
                    help="directory holding emotion_activations.npz / emotion_passages.json")
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=168)
    ap.add_argument("--judge-n", type=int, default=60)
    ap.add_argument("--attribute", default="emotion", choices=["emotion", "language"])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = MODELS[args.model]
    rng = np.random.default_rng(SEED)

    if args.attribute == "language":
        return run_language(args, out_dir, cfg, rng)

    az = np.load(Path(args.acts_dir) / "emotion_activations.npz", allow_pickle=True)
    passages = json.loads((Path(args.acts_dir) / "emotion_passages.json").read_text())
    emotions = [e for e in az.files if e != "neutral"]
    rows = []
    for i in range(args.n):
        e = emotions[i % len(emotions)]
        k = int(rng.integers(min(len(az[e]), len(passages[e]))))
        others = [x for x in emotions if x != e]
        rows.append({"act": az[e][k],
                     "passage": " ".join(str(passages[e][k]).split())[:400],
                     "value": e,
                     "wrong": others[int(rng.integers(len(others)))]})
    acts = np.stack([r["act"] for r in rows]).astype(np.float32)
    print(f"{len(rows)} natural-emotion activations, {len(emotions)} values, "
          f"chance {1/len(emotions):.0%}")

    from nla_inference import NLACritic
    from bench.roundtrip import free_model
    ar_dir = glob.glob(str(Path.home() / ".cache/huggingface/hub" /
                           cfg["av"].replace("-av", "-ar") / "snapshots/*/"))[0]
    ar = NLACritic(ar_dir, device="cuda:0")

    res = {}
    for level in LEVELS:
        S = {"content": [], "assert": [], "wrong": []}
        for i, r in enumerate(rows):
            base = content_at(level, r["passage"])
            v = {"content": base,
                 "assert": f"{base} The passage expresses {r['value']}.",
                 "wrong": f"{base} The passage expresses {r['wrong']}."}
            for a in S:
                _, cos = ar.score(v[a], acts[i])
                S[a].append(float(cos))
        S = {a: np.array(x) for a, x in S.items()}
        presence = float((S["wrong"] - S["content"]).mean())
        correctness = float((S["assert"] - S["wrong"]).mean())
        total = presence + correctness
        p = float("nan")
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(S["assert"], S["wrong"]).pvalue)
        except Exception:
            pass
        res[level] = {"means": {a: float(S[a].mean()) for a in S},
                      "presence": presence, "correctness": correctness,
                      "total": total,
                      "correctness_share": correctness / total if total else float("nan"),
                      "p_correctness": p}
        print(f"  [{level:<9}] presence {presence:+.4f}  correctness {correctness:+.4f}"
              f"  ({res[level]['correctness_share']:+.0%})  p={p:.2g}")
    free_model(ar); del ar

    # ── redundancy at each level, measured not assumed ──────────────────────
    from bench.run_code_legibility import judge_language
    sub = list(rng.choice(len(rows), min(args.judge_n, len(rows)), replace=False))
    for level in LEVELS:
        guess = judge_language([content_at(level, rows[i]["passage"]) for i in sub],
                               emotions)
        red = float(np.mean([g == rows[i]["value"] for g, i in zip(guess, sub)]))
        res[level]["redundancy"] = red
        print(f"  [{level:<9}] redundancy {red:.0%}")

    order = sorted(LEVELS, key=lambda L: res[L]["redundancy"])
    lines = ["| content level | redundancy | mentioning it | being correct | share | p |",
             "|---|---|---|---|---|---|"]
    for L in order:
        r = res[L]
        lines.append(f"| {L} | {r['redundancy']:.0%} | {r['presence']:+.4f} | "
                     f"**{r['correctness']:+.4f}** | {r['correctness_share']:+.0%} | "
                     f"{r['p_correctness']:.2g} |")
    table = "\n".join(lines)
    print(f"\n{table}")

    x = np.array([res[L]["redundancy"] for L in LEVELS])
    y = np.array([res[L]["correctness"] for L in LEVELS])
    rho = p_rho = float("nan")
    try:
        from scipy.stats import spearmanr
        s = spearmanr(x, y)
        rho, p_rho = float(s.statistic), float(s.pvalue)
    except Exception:
        pass
    print(f"\nredundancy vs value-of-being-correct: rho={rho:+.2f} (p={p_rho:.2g}, "
          f"{len(LEVELS)} levels)")

    verdict = ("the information-gain account HOLDS within a single attribute: as the "
               "content gives more of the answer away, being correct is worth less, with "
               "everything else held constant"
               if rho < -0.5 else
               "redundancy does NOT drive it within an attribute, so the earlier "
               "cross-attribute pattern was not a redundancy effect")
    print(f"verdict: {verdict}")

    (out_dir / "redundancy_sweep.md").write_text(
        "# Manipulating redundancy inside one attribute\n\n"
        "The three-attribute result was suggestive but confounded: comparing across "
        "attributes mixes redundancy with attribute identity, which is why two "
        "near-zero-redundancy points differed by 26 points. Here redundancy is "
        "manipulated instead of observed, with one attribute, one reader, one set of "
        f"activations and {len(emotions)} values. Only the content description changes.\n\n"
        + table + f"\n\nSpearman rho = **{rho:+.2f}** (p={p_rho:.2g}) across "
        f"{len(LEVELS)} levels.\n\n**Verdict:** {verdict}.\n\n"
        "> Redundancy is measured at each level by giving a pinned judge the content "
        "alone and asking it to name the emotion. The redacted level masks emotion words "
        "so tone survives but the label cannot be read off directly.\n\n"
        "> Four levels is few for a rank correlation; the ordering and the sign of the "
        "correctness column carry more than the p-value does.\n"
    )
    (out_dir / "redundancy_sweep.json").write_text(json.dumps(
        {"levels": res, "spearman_rho": rho, "spearman_p": p_rho,
         "n": len(rows), "emotions": emotions}, indent=2))
    print(f"\nwrote {out_dir / 'redundancy_sweep.md'}")


if __name__ == "__main__":
    main()
