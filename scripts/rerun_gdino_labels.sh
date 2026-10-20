#!/usr/bin/env bash
# Re-detect every Grounding DINO cell that feeds a reported number, with
# span-based vocabulary recovery.
#
# Boxes, scores, thresholds and model input are unchanged by construction --
# they come straight from the same HuggingFace post-processing call as before.
# Only the label string changes. Output goes to a separate root so the previous
# runs survive for the bit-identity check.
set -u
PY="${PY:-python}"
DEV="${DEV:-cuda:0}"
OUT="${OUT:-outputs/runs_spanfix}"
LOG="outputs/gdino_relabel_logs"
LLMS7="gemma3-4b gemma4-e2b gemma4-e4b gemma4-12b qwen25-vl-7b qwen35-9b internvl3-8b"
LLMS3="gemma4-12b gemma4-e4b qwen35-9b"
GD="gdino-tiny gdino-base"
mkdir -p "$LOG" "$OUT"
echo "$$" > outputs/gdino_relabel.pid

step () {                       # step <name> <args...>
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$LOG/progress.log"
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $name (rc=$rc)" | tee -a "$LOG/progress.log"
  [ $rc -ne 0 ] && echo "FAILED: $name -- see $LOG/$name.log" | tee -a "$LOG/progress.log"
  return $rc
}

step coco "$PY" run_benchmark.py --dataset coco --split test --stage detect \
     --no_resume --llm $LLMS7 --detector $GD \
     --device "$DEV" --output_root "$OUT" || exit 1
step lvis "$PY" run_benchmark.py --dataset lvis --split test --stage detect \
     --no_resume --llm $LLMS3 --detector $GD \
     --device "$DEV" --output_root "$OUT" || exit 1
step coco_ood "$PY" run_benchmark.py --dataset coco_ood --split test --stage detect \
     --no_resume --llm $LLMS7 --detector $GD \
     --device "$DEV" --output_root "$OUT" || exit 1
# The fixed-vocabulary cells (appendix, REPRODUCE.md section 8) account for 2
# of the 36 rows in results/relabel_verification.csv. Guarded rather than
# assumed, so the script still completes if that experiment is removed.
if [ -f scripts/run_fixed_vocab.py ]; then
  step fixed80 "$PY" scripts/run_fixed_vocab.py --dataset coco --detector $GD \
       --split test --device "$DEV" --output_root "$OUT" || exit 1
fi

echo "[$(date +%H:%M:%S)] ALL STEPS DONE" | tee -a "$LOG/progress.log"
