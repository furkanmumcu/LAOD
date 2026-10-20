#!/usr/bin/env bash
# Fixed-ontology reference: every image gets the complete benchmark ontology.
# No MLLM anywhere. Detector settings are the paper's v2 preset, untouched.
set -u
PY="${PY:-python}"
DEV="${DEV:-cuda:0}"
LOG=outputs/fixed_ontology_logs
mkdir -p "$LOG"; echo "$$" > outputs/fixed_ontology.pid
step () { local n="$1"; shift
  echo "[$(date +%H:%M:%S)] START $n" | tee -a "$LOG/progress.log"
  "$@" > "$LOG/$n.log" 2>&1; local rc=$?
  echo "[$(date +%H:%M:%S)] END   $n (rc=$rc)" | tee -a "$LOG/progress.log"
  [ $rc -ne 0 ] && echo "FAILED: $n" | tee -a "$LOG/progress.log"; return $rc; }
R () { "$PY" scripts/run_fixed_vocab.py --split test --store_top 500 --device "$DEV" "$@"; }

step coco_yolo  R --dataset coco --detector yolo-world  || exit 1
step coco_owlv2 R --dataset coco --detector owlv2-base  || exit 1
step lvis_yolo  R --dataset lvis --detector yolo-world  || exit 1
step lvis_owlv2 R --dataset lvis --detector owlv2-base  || exit 1
step lvis_gdt   R --dataset lvis --detector gdino-tiny  || exit 1
step lvis_gdb   R --dataset lvis --detector gdino-base  || exit 1
echo "[$(date +%H:%M:%S)] ALL FIXED-ONTOLOGY RUNS DONE" | tee -a "$LOG/progress.log"
