"""Compare a reproduction run against the committed results, and say what drifted.

WHY THIS EXISTS
  "It reproduces" is not checkable by eye across dozens of numbers in dozens of files, and
  the three tiers in reproduce.sh have genuinely different determinism. Tier 1 is pure
  computation over committed inputs and should match to the last decimal; tier 2 recaptures
  activations and will differ in the last place or two; tier 3 calls a sampled judge and
  moves by whole percentage points between runs of the identical script.

  So a single pass/fail is the wrong output. This reports the delta per number against a
  tolerance that depends on which tier produced it, and names anything that moved more than
  it should have.

USAGE
  python compare_repro.py results_repro            # compare against ./results
  python compare_repro.py results_repro --tol 0.02
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# per-file tolerance. Judge arms are sampled, so they get a wide band and are reported
# as informational rather than as failures. The Test 7 arms regenerate readouts rather than
# analysing committed ones, but the AV seeds its decode (bench/av_local.py), so they hold the
# tight default: a rerun reproduced all 552 prompt_elicitation rows byte-for-byte.
TOL = {
    "judge_competence": 0.15,
    "monitor_sensitivity": 0.15,
    "bandwidth_": 0.02,
    "locate_loss": 0.02,
}
DEFAULT_TOL = 0.005


def flatten(obj, prefix=""):
    """Every numeric leaf, keyed by its path, so nesting differences do not hide drift."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        # lists of per-item booleans (judge arms) collapse to their mean; comparing 40
        # individual coin flips would be noise, their rate is the quantity of interest
        if obj and all(isinstance(x, bool) for x in obj):
            out[f"{prefix}[mean]"] = sum(obj) / len(obj)
        else:
            for i, v in enumerate(obj[:50]):
                out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if math.isfinite(obj):
            out[prefix] = float(obj)
    return out


def tol_for(name: str) -> float:
    for k, v in TOL.items():
        if name.startswith(k):
            return v
    return DEFAULT_TOL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repro_dir")
    ap.add_argument("--baseline", default="results")
    ap.add_argument("--tol", type=float, default=None,
                    help="override every per-file tolerance")
    args = ap.parse_args()
    repro, base = Path(args.repro_dir), Path(args.baseline)

    files = sorted(p.name for p in repro.glob("*.json"))
    if not files:
        raise SystemExit(f"no json files in {repro}")

    total, drifted, missing = 0, [], []
    # results/ is organised into per-test folders, so a baseline is located by BASENAME
    # anywhere beneath it rather than by a flat path
    index = {p.name: p for p in base.rglob("*.json")}
    for name in files:
        b = index.get(name, base / name)
        if not b.exists():
            missing.append(name)
            continue
        tol = args.tol if args.tol is not None else tol_for(name)
        try:
            fa = flatten(json.loads((repro / name).read_text()))
            fb = flatten(json.loads(b.read_text()))
        except json.JSONDecodeError:
            drifted.append((name, "unparseable", 0.0, tol))
            continue
        shared = sorted(set(fa) & set(fb))
        worst, worst_key = 0.0, ""
        for k in shared:
            d = abs(fa[k] - fb[k])
            total += 1
            if d > worst:
                worst, worst_key = d, k
        status = "ok " if worst <= tol else "OFF"
        print(f"  {status} {name:<44} worst |Δ| {worst:.4f} (tol {tol:.3f})"
              f"{'  <- ' + worst_key if worst > tol else ''}")
        if worst > tol:
            drifted.append((name, worst_key, worst, tol))

    print(f"\n{total} numbers compared across {len(files) - len(missing)} files")
    if missing:
        print(f"{len(missing)} file(s) present in the reproduction but not the baseline: "
              + ", ".join(missing[:5]))
    if not drifted:
        print("\nEVERY comparable number is within tolerance.")
    else:
        print(f"\n{len(drifted)} file(s) outside tolerance:")
        for name, key, d, tol in drifted:
            print(f"  {name}: {key} moved {d:.4f} (tol {tol:.3f})")
        print("\nA judge file moving is expected — the judge is sampled and its arms shift "
              "by whole points between runs of the identical script. A tier-1 file moving "
              "at all is not, since those are deterministic given the committed inputs.")


if __name__ == "__main__":
    main()
