# Reproducing the paper

Every number in the paper comes from the commands below. The pipeline splits
into two halves: **model passes**, which need a GPU and produce per-image
detections, and **scoring**, which reads those detections and is cheap. The
split matters — a metric can be changed and everything re-scored without
re-running a single model.

Our own output tables are in `results/`, so you can diff rather than compare by
eye.

---

## 0. Splits

A 500-image hyperparameter holdout is drawn once from COCO val2017 and frozen
to an explicit id list, `results/tuning_split.json`. It is the only data any
parameter is fitted on, and it is excluded from every reported number.

Because LVIS-minival and COCO-OOD are subsets of COCO val2017, that one
exclusion applies to all three benchmarks:

| dataset | full | reported | objects |
|---|---|---|---|
| COCO-Val | 5,000 | **4,500** | 32,814 |
| LVIS-minival | 4,809 | **4,327** | 45,063 |
| COCO-OOD | 504 | **438** | 1,386 |

The id lists ship with this repository, so the splits are reproducible without
re-drawing. Two further frozen 500-image subsets are used by the prompt
ablation, drawn from the respective test splits with the holdout removed first:
`results/ablation_subset.json` (COCO) and `results/ablation_subset_lvis.json`
(LVIS).

Nothing needs running for this step. To verify:

```bash
python -c "
from laod.data.splits import load_tuning_split, load_ablation_subset
from laod.data.splits import ABLATION_LVIS_PATH
t = load_tuning_split()
print(len(t), 'holdout ids')
print('COCO ablation overlap:', len(load_ablation_subset() & t))
print('LVIS ablation overlap:', len(load_ablation_subset(ABLATION_LVIS_PATH) & t))
"
```

## 1. Detector thresholds

Thresholds are fitted on the holdout, against the metric actually reported
(CAAP at maxDets=100; U-AP for COCO-OOD).

```bash
python scripts/tune_thresholds.py --dataset coco     --max_dets 100 --suffix _fixed100
python scripts/tune_thresholds.py --dataset lvis     --max_dets 100 --suffix _fixed100
python scripts/tune_thresholds.py --dataset coco_ood --max_dets 100 --metric uap --suffix _fixed100
python scripts/apply_thresholds.py --sweeps "results/tuning_sweep_*_fixed100.csv"
```

All 21 (dataset, MLLM) fits per detector select the same value — the grid
floor — so one threshold per detector is used:

| detector | parameter | value |
|---|---|---|
| YOLO-World | `conf` | 0.001 |
| Grounding DINO-T | `score_threshold` | 0.03 |
| Grounding DINO-B | `score_threshold` | 0.03 |
| OWLv2 | `score_threshold` | 0.01 |

Grounding DINO's `text_threshold` is fixed at 0.3 and was not tuned. The
unanimity follows from the cap: once it binds, the top 100 detections by score
are the same whether the cut is 0.01 or 0.05, so the cap does the thresholding.

**Skip this step** unless you want to re-derive the thresholds — the fitted
values are already the defaults in `laod/models/registry.py`.

## 2. Model passes

Vocabularies are cached per `(image, MLLM, prompt)`, so detectors never
re-query the MLLM and every detector sees the byte-identical label list. The
grid is therefore additive: 7 MLLM passes feed 28 detector passes.

```bash
LLMS="gemma3-4b gemma4-e2b gemma4-e4b gemma4-12b qwen25-vl-7b qwen35-9b internvl3-8b"
DETS="yolo-world gdino-tiny gdino-base owlv2-base"

python run_benchmark.py --dataset coco     --split test --llm $LLMS --detector $DETS
python run_benchmark.py --dataset lvis     --split test --llm gemma4-12b gemma4-e4b qwen35-9b --detector $DETS
python run_benchmark.py --dataset coco_ood --split test --llm $LLMS --detector $DETS
```

Add `--stage labels` or `--stage detect` to run one half. Runs resume by
default; pass `--no_resume` to force regeneration — necessary if you change a
detector threshold, since otherwise the stored detections are kept.

Each run writes `outputs/runs/<dataset>-<split>__<llm>__<detector>__<prompt>/`
containing the raw MLLM reply, the parsed vocabulary, every box, score and
label, and the full configuration with the annotation file's checksum. Any
later metric is an offline pass over these.

## 3. Scoring

```bash
python scripts/maxdets_compare.py --dataset coco     --snap --out results/main_coco.csv
python scripts/maxdets_compare.py --dataset lvis     --snap --out results/main_lvis.csv
python scripts/maxdets_compare.py --dataset coco_ood --snap --out results/main_coco_ood.csv
python scripts/snap_sbert_all.py                 # SNAP under Sentence-BERT, with the shuffle control
python scripts/uap_pipeline_ood.py               # U-AP / U-F1 / U-PRE / U-REC, 28 cells
```

SNAP is **not** reported on COCO-OOD: that dataset annotates every object under
one label, so there is nothing to name and the shuffle control returns a gain
of exactly zero.

## 4. MLLMs as direct detectors

The baseline the two-stage design is measured against — no detector in the
pipeline.

```bash
python scripts/run_vlm_detect.py --model Qwen/Qwen2.5-VL-7B-Instruct \
    --key qwen25-vl-7b --dataset coco --coord_mode grid
python scripts/run_vlm_detect.py --model OpenGVLab/InternVL3-8B \
    --key internvl3-8b --dataset coco --coord_mode norm1000
python scripts/rescore_vlm.py --dataset coco       # offline re-score
```

`--coord_mode` is not cosmetic: Qwen2.5-VL emits pixels in its own *resized*
frame, InternVL3 emits 0–1000 normalised values. Declaring it wrong yields
boxes that look plausible but sit uniformly mis-sized. The code raises rather
than silently defaulting.

A VLM emits no confidence, so detections are ranked by the mean token
log-probability of the generated coordinates. That is the only such value
recoverable from a stored reply.

## 5. Prompt ablation

```bash
# COCO, three prompts on the frozen 500-image subset
for P in default minimal coco-specific; do
  python run_benchmark.py --dataset coco --split test --subset ablation \
      --llm $LLMS --detector yolo-world --prompt $P
done
python scripts/report_stage.py --pattern "coco-test-ablation__*" --stage stage2 \
    --dataset coco --subset ablation --snap

# LVIS, the same three prompts
bash scripts/run_lvis_prompt_ablation.sh
```

## 6. Figures and reports

```bash
python scripts/pick_qualitative.py --dataset coco --n 10        # choose images
python scripts/render_detections.py --run outputs/runs/... --images <ids> --score 0.25
python scripts/render_ground_truth.py --dataset coco --images <ids>

python scripts/build_final_report.py        # reports/RESULTS.md
python scripts/build_intervals_report.py    # reports/INTERVALS.md  (LO/MI/HI)
python scripts/build_prompt_report.py       # reports/PROMPTS.md
python scripts/audit_reports.py             # verifies tables against their sources
python -m scripts.figures                   # all figures
```

`audit_reports.py` checks the generated reports against the CSVs they were
built from and the conventions against the code — evaluator defaults, grid
equality, every table value against its source. It exits non-zero on failure.

## Cost

Measured on a single 24 GB GPU.

| step | time |
|---|---|
| Vocabulary generation, 7 MLLMs x 4,500 images | ~8 h |
| Detection, 4 detectors x 7 vocabularies x 4,500 images | ~2.5 h |
| LVIS and COCO-OOD | ~1.5 h |
| Scoring, all datasets | ~1 h |

Vocabulary generation dominates, which is why it is cached: changing a detector
threshold or a metric costs minutes, not hours.
