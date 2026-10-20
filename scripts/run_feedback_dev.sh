#!/usr/bin/env bash
# 500-image development phase: 3 configs x 3 iterative conditions.
set -u
PY="${PY:-python}"
DEV="${DEV:-cuda:0}"
LOG=outputs/feedback_logs
mkdir -p "$LOG"; echo "$$" > outputs/feedback_dev.pid
step () { local n="$1"; shift
  echo "[$(date +%H:%M:%S)] START $n" | tee -a "$LOG/progress.log"
  "$@" > "$LOG/$n.log" 2>&1; local rc=$?
  echo "[$(date +%H:%M:%S)] END   $n (rc=$rc)" | tee -a "$LOG/progress.log"
  [ $rc -ne 0 ] && echo "FAILED: $n" | tee -a "$LOG/progress.log"; return $rc; }

# gemma4-e4b has only one detector, so its three conditions run together.
step e4b_yolo "$PY" scripts/run_vocab_feedback.py --split tune \
     --llm gemma4-e4b --detector yolo-world --device "$DEV" || exit 1
# qwen35-9b: the selfreflect reply cache is detector-independent, so the
# yolo-world pass fills it and the owlv2 pass reuses it verbatim.
step qwen_yolo "$PY" scripts/run_vocab_feedback.py --split tune \
     --llm qwen35-9b --detector yolo-world --device "$DEV" || exit 1
step qwen_owlv2 "$PY" scripts/run_vocab_feedback.py --split tune \
     --llm qwen35-9b --detector owlv2-base --device "$DEV" || exit 1
echo "[$(date +%H:%M:%S)] ALL RUNS DONE" | tee -a "$LOG/progress.log"
