"""LLM judge via the `claude` CLI — no API key required.

WHY THE CLI AND NOT THE SDK
  The Anthropic Console role on this account is "Claude Code User", which cannot
  create API keys, and the Claude Code seat is OAuth with no key to extract. But
  `claude -p "..."` runs headless against that existing auth, which is enough.

  Note this only works from a terminal with access to the user's credential store.
  It is NOT callable from a sandboxed subprocess (returns "Not logged in"), so
  this module is run by the user, not by an agent mid-session.

WHY IT IS NEVER THE ONLY GRADER
  A faithfulness claim resting on a single LLM judge is a soft claim, and the
  released benchmark should be reproducible by someone with no Anthropic access at
  all. `bench.grade`'s lexicon grader is primary; this corroborates it and catches
  the cases a fixed word list cannot (e.g. anger expressed as profanity, or an
  emotion described periphrastically).

BATCHING
  One CLI invocation per readout would mean ~500 process starts. Readouts are
  batched into one prompt with numbered items and a JSON reply.

USAGE
  python -m bench.judge --check                       # verify auth works
  python -m bench.judge --grade results/recovery.json # add judge verdicts
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from typing import Sequence

from bench.emotions import EMOTIONS

BATCH = 20
TIMEOUT_S = 600

# Pinned to an explicit RESOLVED id, never an alias and never the CLI default.
# Verified via --output-format json, which reports the id actually billed
# (modelUsage); asking a model to name itself is not reliable:
#   - no --model at all  -> claude-sonnet-4-6   (the CLI default here)
#   - --model opus       -> claude-opus-4-8     (a generation behind)
#   - --model opusplan   -> claude-sonnet-4-6
#   - --model claude-sonnet-5 -> claude-sonnet-4-6  (Sonnet 5 not available here)
#
# Measured cost per call: sonnet-4-6 $0.0549, opus-4-8 $0.0635 — 16% apart, not the
# multiple you would expect from list pricing, because the ~8k-token cached CLI
# system prompt dominates and is tier-independent. Opus 4.8 is therefore taken on
# quality at negligible extra cost.
#
# Resolution is NOT stable: one probe of claude-opus-5 came back as
# claude-haiku-4-5-20251001, evidently a transient capacity fallback, then resolved
# correctly twice after. A silent mid-run downgrade would corrupt grading invisibly,
# so every call reports the model it actually used and mismatches are surfaced.
MODEL = "claude-opus-4-8"

_PROMPT = """You are grading the output of an interpretability tool that tries to \
describe an emotion encoded in a neural network's internal state.

For each numbered readout below, decide which SINGLE emotion from this list it is \
describing, if any:
{candidates}

Rules:
- Judge the emotional content the readout ATTRIBUTES to the text it describes.
- The readout may express the emotion without naming it (e.g. profanity for anger, \
"couldn't stop giggling" for amusement). Count those.
- If the readout describes no clear emotion from the list, answer "none".
- Answer with exactly one label per item. Do not explain.

Reply with ONLY a JSON object mapping item number (as a string) to a label, e.g.
{{"1": "anger", "2": "none", "3": "joy"}}

Readouts:
{items}
"""


# Every model any call actually resolved to, for the provenance stamp. A run that
# silently fell back mid-way will show more than one entry here.
_RESOLVED: Counter[str] = Counter()


def provenance(model: str = MODEL) -> dict[str, object]:
    """What actually graded these rows, recorded INTO the results file.

    A pin in source only helps someone reading the source at the same commit. The
    artifact has to carry its own provenance, so every judged file records the models
    the CLI actually billed across the run — plural, because resolution is not stable
    and a transient fallback would otherwise be invisible.
    """
    ver = subprocess.run(["claude", "--version"], capture_output=True,
                         text=True, timeout=60).stdout.strip()
    return {"judge_model_requested": model,
            "judge_models_resolved": dict(_RESOLVED),
            "judge_model_consistent": list(_RESOLVED) in ([model], []),
            "claude_cli_version": ver}


def _run_claude(prompt: str, model: str = MODEL) -> str:
    """Run one prompt, recording which model the CLI actually billed."""
    proc = subprocess.run(
        ["claude", "-p", prompt, "--model", model, "--output-format", "json"],
        capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    out = (proc.stdout or "").strip()
    if "Not logged in" in out or proc.returncode != 0:
        raise RuntimeError(
            f"`claude -p` failed (rc={proc.returncode}): {out[:200]!r}\n"
            f"Run this from a terminal where `claude` is authenticated."
        )
    payload = json.loads(out)
    for m in payload.get("modelUsage", {}) or ["<unknown>"]:
        _RESOLVED[m] += 1
    return (payload.get("result") or "").strip()


def _parse(reply: str, n: int) -> list[str | None]:
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if m is None:
        raise ValueError(f"no JSON object in judge reply: {reply[:200]!r}")
    data = json.loads(m.group(0))
    valid = set(EMOTIONS)
    out: list[str | None] = []
    for i in range(1, n + 1):
        label = str(data.get(str(i), "none")).strip().lower()
        out.append(label if label in valid else None)
    return out


def judge_batch(readouts: Sequence[str],
                candidates: Sequence[str] = tuple(EMOTIONS)) -> list[str | None]:
    """Dominant emotion per readout, or None. Batched over `claude -p` calls."""
    results: list[str | None] = []
    for start in range(0, len(readouts), BATCH):
        chunk = list(readouts[start:start + BATCH])
        items = "\n\n".join(
            # Readouts are multi-paragraph; collapse so item boundaries stay clear.
            f"{i + 1}. {' '.join((t or '').split())[:600]}"
            for i, t in enumerate(chunk)
        )
        reply = _run_claude(_PROMPT.format(
            candidates=", ".join(candidates), items=items))
        results.extend(_parse(reply, len(chunk)))
        print(f"  judged {min(start + BATCH, len(readouts))}/{len(readouts)}", flush=True)
    return results


def _main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify `claude -p` auth")
    ap.add_argument("--grade", help="a results JSON with a 'rows' list of readouts")
    args = ap.parse_args()

    if args.check:
        # Report the model the CLI actually resolved to. Asking the model what it is
        # is unreliable; the response metadata is not.
        import json as _json
        raw = subprocess.run(
            ["claude", "-p", "say ok", "--model", MODEL, "--output-format", "json"],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        ).stdout
        used = list(_json.loads(raw).get("modelUsage", {})) or ["<unknown>"]
        print(f"requested --model {MODEL!r}; CLI resolved to: {used}")
        print(f"provenance stamped into results: {provenance()}")

        labels = judge_batch(["The text is furious and full of rage.",
                              "A calm description of quarterly accounting figures."])
        print(f"judge returned: {labels}")
        assert labels[0] == "anger", f"expected anger, got {labels[0]!r}"
        print("OK — `claude -p` judge works.")
        return

    assert args.grade, "pass --check or --grade <results.json>"
    path = Path(args.grade)
    data = json.loads(path.read_text())
    rows = data["rows"]

    prov = provenance()
    print(f"judging with {prov['judge_model_resolved']} "
          f"(CLI {prov['claude_cli_version']})")
    labels = judge_batch([r["readout"] for r in rows])

    # Each runner names the lexicon verdict differently — recovery uses "hit",
    # controls "names_injected", the natural comparison "nla_hit" — but all three
    # mean "the lexicon grader saw the target emotion".
    lex_key = next((k for k in ("hit", "names_injected", "nla_hit") if k in rows[0]), None)
    assert lex_key, (
        f"no lexicon-verdict column in these rows; expected one of hit / "
        f"names_injected / nla_hit, got {sorted(rows[0])}"
    )
    agree = 0
    for r, lab in zip(rows, labels):
        r["judge_label"] = lab
        r["judge_hit"] = (lab == r["emotion"])
        if r.get("decoy"):
            r["judge_names_decoy"] = (lab == r["decoy"])
        agree += int(bool(r["judge_hit"]) == bool(r[lex_key]))
    data["judge_lexicon_agreement"] = agree / len(rows)
    data.update(provenance())      # after grading: _RESOLVED is populated by then
    if not data["judge_model_consistent"]:
        print(f"WARNING: judge resolution was not stable across the run: "
              f"{data['judge_models_resolved']}")

    out = path.with_name(path.stem + "_judged.json")
    out.write_text(json.dumps(data, indent=2))
    print(f"\njudge/lexicon agreement: {agree / len(rows):.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
