#!/usr/bin/env bash
# Build the v2-preset Round-1 baselines, then redo only what the OWLv2
# threshold confound invalidated.
set -u
PY="${PY:-python}"
DEV="${DEV:-cuda:0}"
LOG=outputs/feedback_logs
mkdir -p "$LOG"; echo "$$" > outputs/feedback_fix.pid
step () { local n="$1"; shift
  echo "[$(date +%H:%M:%S)] START $n" | tee -a "$LOG/fix.log"
  "$@" > "$LOG/fix_$n.log" 2>&1; local rc=$?
  echo "[$(date +%H:%M:%S)] END   $n (rc=$rc)" | tee -a "$LOG/fix.log"
  [ $rc -ne 0 ] && echo "FAILED: $n" | tee -a "$LOG/fix.log"; return $rc; }

step base_qwen_yolo  "$PY" scripts/make_feedback_baseline.py --split tune --llm qwen35-9b  --detector yolo-world --device "$DEV" || exit 1
step base_e4b_yolo   "$PY" scripts/make_feedback_baseline.py --split tune --llm gemma4-e4b --detector yolo-world --device "$DEV" || exit 1
step base_qwen_owl   "$PY" scripts/make_feedback_baseline.py --split tune --llm qwen35-9b  --detector owlv2-base --device "$DEV" || exit 1

# OWLv2's feedback was built from a 0.15-threshold D0, so both grounded
# conditions must be regenerated. Self-reflection never sees D0 and its cached
# replies stay valid; its detection pass is redone only so every condition
# shares one baseline.
rm -f outputs/feedback_cache/coco_tune__qwen35-9b__feedback_refine__owlv2-base.jsonl \
      outputs/feedback_cache/coco_tune__qwen35-9b__feedback_full__owlv2-base.jsonl
step rerun_qwen_owl  "$PY" scripts/run_vocab_feedback.py --split tune --llm qwen35-9b --detector owlv2-base --device "$DEV" || exit 1

# YOLO-World rounds were already at the right threshold; only the detection
# pass is redone so all conditions read the same baseline object.
step rerun_qwen_yolo "$PY" scripts/run_vocab_feedback.py --split tune --llm qwen35-9b  --detector yolo-world --device "$DEV" || exit 1
step rerun_e4b_yolo  "$PY" scripts/run_vocab_feedback.py --split tune --llm gemma4-e4b --detector yolo-world --device "$DEV" || exit 1
echo "[$(date +%H:%M:%S)] FIX DONE" | tee -a "$LOG/fix.log"
