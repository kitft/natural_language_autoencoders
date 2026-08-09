#!/usr/bin/env bash
# Reproduce every run the write-up rests on, in dependency order.
#
# Three tiers, because they need different things. Run what you can.
#
#   TIER 1  analysis only        no GPU, no API key. Reproduces from the committed
#                                trial JSONs, so it works on any machine
#   TIER 2  activations          needs a GPU and the released NLA checkpoints
#   TIER 3  judge                needs an authenticated `claude` CLI, and is the least
#                                reproducible tier — see REPRODUCE.md on model drift
#
# Usage:  bash reproduce.sh [1|2|3|all]     default: 1

set -uo pipefail
TIER="${1:-1}"
V="${PYTHON:-python}"
OUT="${OUT_DIR:-results_repro}"
mkdir -p "$OUT"

run () {
  echo; echo "########## $1 ##########"; shift
  if "$@"; then echo "-> ok"; else echo "-> FAILED (exit $?)"; fi
}

# Everything downstream reads the committed trial files rather than regenerating them,
# so a Tier-2 failure does not silently invalidate Tier-1 numbers.
SYC=results/trials/sycophancy_influence_raw.json
SYC_G=results/trials/sycophancy_influence_raw_gemma.json
PER=results/trials/influence_persona_raw.json
PER_G=results/trials/influence_persona_gemma_raw.json
SOC=results/trials/influence_socialproof_raw.json
SOC_G=results/trials/influence_socialproof_gemma_raw.json
FEW=results/trials/influence_fewshot_raw.json

# ── TIER 1 ────────────────────────────────────────────────────────────────────
if [[ "$TIER" == "1" || "$TIER" == "all" ]]; then
  echo "TIER 1 — analysis only, no GPU or API key required"

  # the cue source for the cued judge arm, and the shortcut baseline. Both read the
  # committed trials and write no activations, so they belong in the analysis tier.
  run "classifier scrutiny" \
    "$V" -m bench.run_classifier_scrutiny --trials-dir results/trials --out-dir "$OUT"
  run "readout signal" \
    "$V" -m bench.run_readout_signal --trials-dir results/trials --out-dir "$OUT"

  # the result: endpoints vs link, on every corpus that ships a capacity artifact. Running
  # only three of the five meant the other two were never regenerated, and the fewshot one
  # silently kept a result predating the top2() fix.
  for f in "$SYC" "$SYC_G" "$PER" "$SOC" "$FEW"; do
    run "capacity: $(basename "$f")" \
      "$V" -m bench.run_readout_capacity --raw "$f" --out-dir "$OUT"
  done

  # control: does the readout beat the raw prompt, and beat knowing the answer? Persona and
  # socialproof are here because results 4.2 quotes their answer-only baseline; running only
  # sycophancy left those two cells with no artifact behind them.
  while read -r m raw lab; do
    run "premise control: $lab" \
      "$V" -m bench.run_prompt_baseline --model "$m" --raw "$raw" --out-dir "$OUT" --label "$lab"
  done <<EOF
qwen  $SYC   prompt_baseline
gemma $SYC_G prompt_baseline_gemma
qwen  $PER   prompt_baseline_persona
gemma $PER_G prompt_baseline_persona_gemma
qwen  $SOC   prompt_baseline_socialproof
gemma $SOC_G prompt_baseline_socialproof_gemma
EOF
fi

# ── TIER 2 ────────────────────────────────────────────────────────────────────
if [[ "$TIER" == "2" || "$TIER" == "all" ]]; then
  echo; echo "TIER 2 — needs a GPU and the released checkpoints"

  # control: is the probe winning only because it is a wider channel?
  for f in "$SYC" "$PER" "$SOC"; do
    run "bandwidth: $(basename "$f")" \
      "$V" -m bench.run_bandwidth_control --model qwen --raw "$f" --out-dir "$OUT"
  done

  # the mechanism: is a true link worth more to the AR than a false one?
  for m in qwen gemma; do
    raw=$SYC; [[ "$m" == gemma ]] && raw=$SYC_G
    run "locate loss: $m" "$V" -m bench.run_locate_loss --model "$m" \
      --raw "$raw" --out-dir "$OUT"
  done

  # can the readout be asked for the missing information directly? The emotion arm reads
  # the cached activations under results/trials/; the influence arm needs the judge and so
  # lives in Tier 3.
  # the emotion corpus is ~20 MB and gitignored rather than shipped. It is fully seeded,
  # so it is rebuilt here — into "$OUT", never into the committed baseline.
  run "queryability: build emotion corpus" \
    "$V" -m bench.emotions --out-dir "$OUT"
  run "queryability: cross-corpus" \
    "$V" -m bench.run_cross_corpus --acts-dir "$OUT" --out-dir "$OUT"
  run "queryability: prompt elicitation" \
    "$V" -m bench.run_prompt_elicitation --acts-dir "$OUT" --out-dir "$OUT"

  # FULL=1 goes all the way back to the checkpoints. The sycophancy trials are the
  # substrate for most of the analysis tiers, so they are regenerated first.
  if [[ "${FULL:-0}" == "1" ]]; then
    for m in qwen gemma; do
      run "regenerate sycophancy trials: $m" \
        "$V" -m bench.run_sycophancy_influence --model "$m" --out-dir "$OUT"
    done
  fi

  # the scale-up itself. SLOW — regenerates trials, activations and readouts, and
  # will not byte-match the committed JSONs because generation is not pinned.
  if [[ "${FULL:-0}" == "1" ]]; then
    for inf in persona socialproof fewshot; do
      for m in qwen gemma; do
        run "battery: $inf/$m" "$V" -m bench.run_influence_battery \
          --model "$m" --influence "$inf" --corpus globalopinions \
          --claims 200 --draws 2 --out-dir "$OUT"
      done
    done
  else
    echo; echo "## skipping the battery regeneration (set FULL=1 to include it)"
  fi
fi

# ── TIER 3 ────────────────────────────────────────────────────────────────────
if [[ "$TIER" == "3" || "$TIER" == "all" ]]; then
  echo; echo "TIER 3 — needs an authenticated \`claude\` CLI"
  if ! command -v claude >/dev/null; then
    echo "## claude CLI not found; skipping tier 3"
  else
    for m in qwen gemma; do
      raw=$SYC; [[ "$m" == gemma ]] && raw=$SYC_G
      # --raw is required here for the same reason as tier 2: without it --out-dir
      # doubles as the INPUT directory and the run dies looking for its own trials
      run "judge ladder: $m" \
        "$V" -m bench.run_judge_competence --model "$m" --raw "$raw" --out-dir "$OUT"
      run "label sway: $m" \
        "$V" -m bench.run_sycophancy_judge --model "$m" --trials-dir results/trials --out-dir "$OUT"
      run "prompted readout: $m" \
        "$V" -m bench.run_prompted_readout --model "$m" --raw "$raw" --out-dir "$OUT"
      run "monitor sensitivity: $m" \
        "$V" -m bench.run_monitor_sensitivity --model "$m" --raw "$raw" --out-dir "$OUT"
    done
    # Not per-model, so it sits outside the loop. It reads the emotion corpus that TIER 2
    # builds into $OUT, so `reproduce.sh 3` alone has no input for it. Skip with a message
    # rather than dying: a reviewer running tier 3 on its own should be told what to run,
    # not handed a FileNotFoundError from inside numpy.
    if [[ -f "$OUT/emotion_activations.npz" ]]; then
      run "redundancy sweep" \
        "$V" -m bench.run_redundancy_sweep --acts-dir "$OUT" --out-dir "$OUT"
    else
      echo; echo "## skipping the redundancy sweep: no $OUT/emotion_activations.npz"
      echo "## tier 2 builds it — run 'bash reproduce.sh all', or tier 2 first"
    fi
  fi
fi

echo; echo "########## done — outputs in $OUT ##########"
echo "Compare against the committed results/ with:  python compare_repro.py $OUT"
