#!/usr/bin/env bash
# Prompt ablation on LVIS: 500 frozen images, YOLO-World, three prompts.
#
# Tests whether `coco-specific` -- which wins on COCO -- transfers. Its three
# instructions each contradict LVIS's scheme: "write only the general name"
# against 1,203 fine-grained categories, "do not list infrastructural objects"
# against fireplug/streetlight/telephone pole, "don't use plural" against 25
# plural categories. If it loses here, that is the evidence for using one
# prompt across datasets rather than the best prompt per dataset.
set -uo pipefail
PY="${PY:-python}"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
echo $$ > outputs/lvis_ablation.pid
LLMS="${LLMS:-gemma4-12b gemma4-e4b qwen35-9b}"
# Gemma4-12B needs ~22 GB, so the label pass wants a card to itself.
DEV="${DEV:-cuda:0}"
PROMPTS="${PROMPTS:-default minimal coco-specific}"
for PROMPT in $PROMPTS; do
    echo "=== $(date +%H:%M:%S) prompt=$PROMPT ==="
    $PY run_benchmark.py --dataset lvis --split test --subset ablation \
        --llm $LLMS --detector yolo-world --prompt "$PROMPT" \
        --detector_preset v2 --device "$DEV" --batch_size 8 \
        || { echo "FAILED on $PROMPT"; exit 1; }
done
echo "=== $(date +%H:%M:%S) scoring ==="
$PY scripts/report_stage.py --pattern "lvis-test-ablation__*" --stage stage2_lvis \
    --title "Prompt ablation on LVIS (500-image subset, YOLO-World)" \
    --dataset lvis --split test --subset ablation --snap || exit 1
echo "ALLDONE"
