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

Grounding DINO's `text_threshold` is fixed at 0.3 and was not tuned, and since
labels are now read from the logits rather than from the decoded phrase it no
longer influences any label — it affects only a rendered string nothing reads.
See the fourth convention in the README and [LABEL_AUDIT.md](LABEL_AUDIT.md).

The unanimity follows from the cap: once it binds, the top 100 detections by
score are the same whether the cut is 0.01 or 0.05, so the cap does the
thresholding.

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
python scripts/semloc_all.py                     # SemLoc-AP, full 10x10 grid per cell
python scripts/uap_pipeline_ood.py               # U-AP / U-F1 / U-PRE / U-REC, 28 cells
```

The third line passes `--snap` like the other two and will tell you it is
ignoring it. SNAP and SemLoc-AP are **not defined** on COCO-OOD: the dataset
annotates every object under the single category `unknow object`, so there is
no name to match against and the shuffle control returns a gain of exactly
zero. The scorers consult `has_semantic_labels()` in
`laod/data/annotations.py` and emit no `snap*` columns rather than writing a
column of zeros — `semloc_all.py` skips the dataset outright.

This is worth being explicit about because the failure is silent and the
output looks like a result. Scored against `unknow object` under **CLIP** text
embeddings, SNAP on COCO-OOD returns ~0.55 across all 28 cells — a number
produced entirely by CLIP's cone effect, and one its own label-shuffle control
reproduces to four decimals. Read COCO-OOD on U-AP.

The rule is enforced at every entry point that takes a dataset —
`maxdets_compare.py`, `semloc_all.py`, `report_stage.py`, `rescore_vlm.py`,
`run_vlm_detect.py`, `run_benchmark.py --evaluate` and
`eval_metrics_only.py` — not only in the report builders, which is where it
used to live and why a SNAP column was written anyway.

`snap_sbert_all.py` is the deliberate exception and still scores COCO-OOD. It
takes no `--dataset`; its COCO-OOD group exists to produce the shuffle control
that establishes the exclusion, reported with `chance_*` and `gain_*` columns
and reading 0.0000 throughout. No report displays those rows.

`semloc_all.py` writes `results/semloc.csv` (one row per cell) and
`results/semloc_matrices.json` (the full 10×10 matrix per cell, which is what
the threshold-marginal tables are built from). It needs no GPU beyond the
sentence encoder.

### Run the two sweeps again after steps 4 and 5

`snap_sbert_all.py` and `semloc_all.py` both sweep **every** stage, but at
this point in the protocol only the step-2 runs exist. Each is cheap, offline
and idempotent, so the simplest thing is to run both again at the end; they
are listed here because this is where the step-2 cells are scored.

| sweep | cells here | after step 4 | after step 5 | shipped |
|---|---:|---:|---:|---:|
| `snap_sbert_all.py` | 68 | 72 | 102 | 102 |
| `semloc_all.py` | 40 | 42 | 72 | 74 |

A group whose runs are absent is reported and skipped, so running early costs
nothing but an incomplete table.

The last two `semloc.csv` cells are the `fixed_vocab` group, which needs the
runs from `scripts/run_fixed_vocab.py` (§8). Run that first and all 74
reproduce; skip it and the group is reported as absent.

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

## 6. Reports

```bash
python scripts/build_final_report.py        # reports/RESULTS.md     (per-cell grid tables)
python scripts/build_intervals_report.py    # reports/INTERVALS.md   (LO/MI/HI + SemLoc marginals)
python scripts/build_semloc_report.py       # reports/SEMLOC.md      (per-cell 10x10 matrices)
python scripts/build_prompt_report.py       # reports/PROMPTS.md
python scripts/audit_reports.py             # verifies tables against their sources
```

**No plotting or image-rendering code ships here.** The paper's grid figures
are a rendering of one value per MLLM x detector cell; `build_final_report.py`
emits exactly those values as markdown grid tables, and `results/main_*.csv`
holds them one row per cell. The qualitative figures are rendered images rather
than measurements — the predictions behind them are in the run directories,
where every box, score and label is stored.

`audit_reports.py` checks the generated reports against the CSVs they were
built from and the conventions against the code — evaluator defaults, grid
equality, every table value against its source. It exits non-zero on failure.
Nine sections, 66 checks, including that SemLoc-AP's two axes are still the
CAAP and SNAP grids, that no `snap*` column survives on COCO-OOD, and that no
stale CLIP SNAP value appears in any report.

## 7. Verifying the Grounding DINO label fix

Only needed if you re-detect the Grounding DINO cells. The fix changes which
vocabulary entry a box is reported as grounding to, and must change nothing
else:

```bash
bash scripts/rerun_gdino_labels.sh     # re-detect into outputs/runs_spanfix
python scripts/verify_relabel.py       # compares it against outputs/runs
```

The re-detection writes to a **separate** directory rather than overwriting,
so the comparison has something to compare against; swap the directories only
once the verification passes.

`verify_relabel.py` checks three things per cell — bit identity of boxes and
scores, then CAAP and U-AP (neither of which reads a label), then what did
change: the empty-label rate and how many labels moved. It writes
`results/relabel_verification.csv` and exits non-zero if anything in the first
two moved.

Ours ships in `results/`: **36 cells, all boxes and scores bit-identical, CAAP
and U-AP identical on every one.** Two of the 36 are the fixed-vocabulary cells
from §8; `rerun_gdino_labels.sh` runs that step when `run_fixed_vocab.py` is
present, which it now is.

## 8. Appendix experiments

Four experiments reported in the appendix. All run in the same environment and
read the same cached runs; none needs a fresh detector pass except where noted.

### Fixed-ontology reference (Tables 25–28)

Every image queried with the **complete** benchmark ontology — all 80 COCO
categories, all 1,203 LVIS categories — instead of an MLLM-generated
vocabulary, across all four detectors.

```bash
bash scripts/run_fixed_ontology.sh          # detector pass, full ontology per image
python scripts/eval_fixed_ontology.py       # -> results/fixed_ontology/
```

This is a **reference point, not an upper bound.** Supplying the ontology
removes the naming problem but adds 80 or 1,203 competing phrases to every
query, which costs localisation. Table 27 reports the difference in both
directions.

### Vocabulary-level refinement, and whether it helps (Tables 19–20)

The detector's per-concept support is summarised back to the MLLM, which
revises the **vocabulary**; the detector then re-grounds the revised list. The
MLLM never sees or emits a coordinate. Implemented in `laod/feedback/`.

```bash
python scripts/make_feedback_baseline.py    # matched one-pass baseline
bash   scripts/run_feedback_dev.sh          # four conditions, 500-image holdout
python scripts/eval_vocab_feedback.py       # -> results/feedback_dev/
python scripts/build_feedback_report.py
```

Refinement applied unconditionally does not help. Gating it on first-pass
detector confidence — refine only the images the detector is least sure about —
is the selective variant:

```bash
python scripts/selective_refinement.py          # gate, rate curves
python scripts/selective_refinement_control.py  # 1,000 random gates at the same rate
python scripts/selective_refinement_analysis.py
python scripts/selective_refinement_appendix.py # -> results/selective_refinement/
```

The control matters more than the result. A random gate at the same refinement
rate is the thing the confidence gate has to beat, and `selective_refinement_
control.py` draws 1,000 of them: Table 20 reports where the confidence gate
falls in that distribution rather than asserting it is better.

The feedback loop runs the MLLM a second time, so unlike every other script in
this section it is **not** an offline pass.

### Fixed-vocabulary probe

`results/fixed_vocab_coco.csv` is an internal two-row probe — Grounding DINO
only, COCO-80 as the prompt — that preceded the fixed-ontology experiment above
and is superseded by it. It ships because `scripts/eval_fixed_vocab.py` reads
it, not because it backs a table in the paper.

## Cost

Measured on a single 24 GB GPU.

| step | time |
|---|---|
| Vocabulary generation, 7 MLLMs x 4,500 images | ~8 h |
| Detection, 4 detectors x 7 vocabularies x 4,500 images | ~2.5 h |
| LVIS and COCO-OOD | ~1.5 h |
| Scoring, all datasets | ~1 h |
| SemLoc-AP, 74 cells (step 3, re-run after step 5) | 43 min (measured) |
| Grounding DINO re-detection + verification (step 7, optional) | detection cost again, for the 2 G-DINO backends |

Vocabulary generation dominates, which is why it is cached: changing a detector
threshold or a metric costs minutes, not hours.
