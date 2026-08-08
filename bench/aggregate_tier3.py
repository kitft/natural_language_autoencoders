"""Pool the judge replicates into intervals, because a point estimate does not reproduce.

THE PROBLEM THIS SOLVES
  The judge arms are the one tier of this project that cannot be re-derived. `bench/judge.py`
  pins a model name, but the CLI's resolution of that name is not stable — its own header
  records a probe of one model returning another, a transient capacity fallback. Two runs of
  the identical script, same seed, moved one arm from 70.0% to 80.0%.

  Regenerating once does not fix that; it produces another single sample with the same
  property. Replicates do: with a distribution, "it reproduces" becomes the checkable claim
  that a fresh run lands inside the interval, rather than the false claim that it lands on
  the number.

WHAT VARIES BETWEEN REPLICATES
  Both the ITEMS and the judge's sampling. `--seed` reseeds pair selection, so each replicate
  scores a different set of answer-matched pairs and gets an independent draw from the judge.
  Holding items fixed would measure judge noise alone and understate the spread a reproducer
  faces.

WHAT TO READ
  The ORDERING of the arms, which was stable across every run, and the interval. Not the
  mean to three decimals — that is the thing this file exists to stop anyone quoting.

USAGE
  python -m bench.aggregate_tier3
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def load(pattern: str, keys: list[str]) -> dict:
    """Collect one metric per arm per replicate from files matching a glob."""
    out: dict[str, list[float]] = {k: [] for k in keys}
    seeds = []
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        m = re.search(r"_s(\d+)\.json$", f)
        seeds.append(int(m.group(1)) if m else -1)
        for k in keys:
            v = d.get(k)
            if isinstance(v, list):
                v = float(np.mean(v)) if v and isinstance(v[0], bool) else None
            if isinstance(v, (int, float)):
                out[k].append(float(v))
    return {"arms": out, "n": len(seeds), "seeds": seeds}


def fmt(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    a = np.array(vals)
    return f"{a.mean():.1%} [{a.min():.1%}–{a.max():.1%}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/t1_t2_judge_arms")
    ap.add_argument("--rep-dir", default="results/t1_t2_judge_arms/rep")
    args = ap.parse_args()
    out_dir, rep = Path(args.out_dir), Path(args.rep_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report, summary = [], {}
    for model, sfx in (("qwen", ""), ("gemma", "_gemma")):
        jc = load(str(rep / f"judge_competence{sfx}_s*.json"),
                  ["signposted", "cued", "subtle"])
        if jc["n"]:
            summary[f"judge_{model}"] = {k: v for k, v in jc["arms"].items()}
            report.append(f"### judge ladder — {model} ({jc['n']} replicates)\n")
            report.append("| arm | mean [min–max] |\n|---|---|")
            for arm in ("subtle", "cued", "signposted"):
                report.append(f"| {arm} | {fmt(jc['arms'][arm])} |")
            # the ordering is the claim; check it held in every replicate
            trip = list(zip(jc["arms"]["subtle"], jc["arms"]["cued"],
                            jc["arms"]["signposted"]))
            held = sum(s <= c <= g for s, c, g in trip)
            report.append(f"\nOrdering subtle ≤ cued ≤ signposted held in "
                          f"**{held}/{len(trip)}** replicates.\n")
            print(f"{model} judge: subtle {fmt(jc['arms']['subtle'])}, "
                  f"cued {fmt(jc['arms']['cued'])}, "
                  f"signposted {fmt(jc['arms']['signposted'])}, ordering {held}/{len(trip)}")

        ms = sorted(glob.glob(str(rep / f"monitor_sensitivity{sfx}_s*.json")))
        if ms:
            per_p: dict[str, list[float]] = {}
            for f in ms:
                for k, v in json.loads(Path(f).read_text()).items():
                    if isinstance(v, dict) and "recovered" in v:
                        per_p.setdefault(k, []).append(float(v["recovered"]))
            summary[f"sensitivity_{model}"] = per_p
            report.append(f"### monitor sensitivity — {model} ({len(ms)} replicates)\n")
            report.append("| planted p | judge recovery, mean [min–max] |\n|---|---|")
            for k in sorted(per_p, key=lambda x: -float(x)):
                report.append(f"| {k} | {fmt(per_p[k])} |")
            top = per_p.get("1", [])
            if top:
                report.append(f"\nAt p=1 the marker is perfectly diagnostic — any reader "
                              f"noticing it scores 100%. Observed: **{fmt(top)}**.\n")
            print(f"{model} sensitivity p=1: {fmt(per_p.get('1', []))}")

    if not report:
        raise SystemExit(f"no replicate files under {rep}")

    (out_dir / "tier3_replicates.md").write_text(
        "# Judge arms as intervals\n\n"
        "The judge tier does not reproduce as point estimates: the CLI's model resolution "
        "is not stable, and two runs of the identical script moved one arm from 70.0% to "
        "80.0%. Replicates convert that into a checkable claim — a fresh run should land "
        "inside the interval, not on the number.\n\n"
        "Each replicate reseeds pair selection, so **items and judge sampling both vary**. "
        "Holding items fixed would measure judge noise alone and understate the spread.\n\n"
        + "\n".join(report) +
        "\n> Read the ordering and the interval. The mean is not meaningful to three "
        "decimals, which is precisely what this file exists to stop anyone quoting.\n\n"
        "> The single-run numbers these replace are preserved under "
        "`results/t1_t2_judge_arms/tier3_singlerun_backup/`.\n")
    (out_dir / "tier3_replicates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'tier3_replicates.md'}")


if __name__ == "__main__":
    main()
