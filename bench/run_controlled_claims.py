"""Claim audit on a constructed corpus, across both released NLAs.

Every attribute is exact and balanced by construction (see controlled_corpus), so no
threshold is guessed and no base rate assumed — the two failures that undermined
earlier claim audits here.

For each attribute the readout is scored on:
  recall     does the readout name the true value at all
  precision  when it names a value, is that value correct
  shuffled   the same scoring against a DIFFERENT passage's attributes; a matcher
             that fires indiscriminately shows up here rather than in the headline

Run for Qwen2.5-7B L20 and Gemma-3-12B L32 so any finding is checked against two
independently trained NLAs rather than one checkpoint.

USAGE
  python -m bench.run_controlled_claims
  python -m bench.run_controlled_claims --models qwen
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from bench.controlled_corpus import ATTRS, CUES, build, names

MODELS = {
    "qwen": {"base": "Qwen/Qwen2.5-7B-Instruct", "block": 20,
             "av": "models--kitft--nla-qwen2.5-7b-L20-av"},
    "gemma": {"base": "unsloth/gemma-3-12b-it", "block": 32,
              "av": "models--kitft--nla-gemma3-12b-L32-av"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    texts, labels = build(reps=args.reps)
    n = len(texts)
    print(f"{n} constructed passages, {len(ATTRS)} attributes, balanced by construction")
    for a in ATTRS:
        vals = [str(l[a]) for l in labels]
        print(f"  {a:<10} {dict((v, vals.count(v)) for v in sorted(set(vals)))}")

    all_res = {}
    for mname in args.models:
        cfg = MODELS[mname]
        print(f"\n=== {mname} ===")
        from bench.capture import capture_activations, load_model
        from bench.roundtrip import free_model, verbalize_all
        model, tok = load_model(cfg["base"])
        acts = capture_activations(model, tok, texts, block_index=cfg["block"])
        free_model(model); del model

        from bench.av_local import LocalNLAClient
        av_dir = glob.glob(str(Path.home() / ".cache/huggingface/hub" /
                               cfg["av"] / "snapshots/*/"))[0]
        av = LocalNLAClient(av_dir)
        readouts = verbalize_all(av, list(acts), max_new_tokens=args.max_new_tokens)
        free_model(av)

        res = {}
        for a in ATTRS:
            vals = sorted({str(l[a]) for l in labels})
            recall, prec_num, prec_den, shuf = 0, 0, 0, 0
            for i, r in enumerate(readouts):
                true_v = labels[i][a]
                other_v = labels[(i + 1) % n][a]
                if names(r, a, true_v):
                    recall += 1
                if names(r, a, other_v):
                    shuf += 1
                # precision: over every value the readout names, how many are correct
                for v in CUES[a]:
                    if names(r, a, v):
                        prec_den += 1
                        prec_num += int(v == true_v)
            res[a] = {"recall": recall / n,
                      "precision": (prec_num / prec_den) if prec_den else float("nan"),
                      "shuffled_recall": shuf / n, "n_values": len(vals)}
            p = res[a]["precision"]
            print(f"  {a:<10} recall {res[a]['recall']:.0%}  "
                  + (f"precision {p:.0%}  " if p == p else "precision n/a  ")
                  + f"shuffled {res[a]['shuffled_recall']:.0%}")
        all_res[mname] = {"scores": res, "readouts": readouts}

    hdr = "| attribute | " + " | ".join(
        f"{m} recall | {m} prec | {m} shuf" for m in args.models) + " |"
    lines = [hdr, "|" + "---|" * (1 + 3 * len(args.models))]
    for a in ATTRS:
        cells = []
        for m in args.models:
            s = all_res[m]["scores"][a]
            cells += [f"{s['recall']:.0%}",
                      f"{s['precision']:.0%}" if s["precision"] == s["precision"] else "n/a",
                      f"{s['shuffled_recall']:.0%}"]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(f"\n{table}")

    (out_dir / "controlled_claims.md").write_text(
        "# Claim audit on a constructed corpus, across two NLAs\n\n"
        f"{n} passages assembled from per-attribute clauses, so every attribute is\n"
        "exact and balanced by construction — no guessed thresholds, no assumed base\n"
        "rates. Both released NLAs are scored identically.\n\n" + table + "\n\n"
        "> recall = names the true value. precision = of the values it names, how many\n"
        "> are right. shuffled = names a DIFFERENT passage's value, which is where an\n"
        "> indiscriminate matcher shows up.\n"
    )
    (out_dir / "controlled_claims.json").write_text(json.dumps(
        {"labels": labels, "texts": texts,
         "results": {m: all_res[m] for m in args.models}}, indent=2))
    print(f"\nwrote {out_dir / 'controlled_claims.md'}")


if __name__ == "__main__":
    main()
