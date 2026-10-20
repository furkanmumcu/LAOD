<h1 align="center">
  <img width="760"
       alt="MLLM-Guided Autonomous Object Detection for Open-World Understanding"
       src="assets/title.svg">
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.10844"><img alt="arXiv"
     src="https://img.shields.io/badge/Read_Our_Paper-arXiv-red"></a>
  <a href="https://huggingface.co/spaces/fumucu/LAOD"><img alt="HuggingFace Space"
     src="https://img.shields.io/badge/🤗-HuggingFace%20Demo-blue.svg"></a>
</p>

A multimodal LLM reads an image and writes an **image-specific detection
vocabulary**; an open-vocabulary detector then grounds those names. Nothing is
fine-tuned and no category list is supplied at inference time.

This repository contains the full implementation and the code to reproduce
every number in the paper. It evaluates **7 MLLMs x 4 open-vocabulary
detectors** on COCO-Val, LVIS-minival and COCO-OOD, and includes the three
metrics introduced in the paper: CAAP, SNAP and SemLoc-AP.

> Datasets and model weights are **not** included — they are downloaded from
> their original sources, see [Data](#data). Everything else needed to
> reproduce the paper is here, including the frozen evaluation splits.

## Install

```bash
git clone https://github.com/furkanmumcu/LAOD.git && cd LAOD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add HF_TOKEN only if you use gated models
pytest tests/ -q              # 145 pass; 3 skip until data is provisioned
```

A CUDA GPU is required for the model passes. Gemma4-12B needs ~22 GB; the rest
fit comfortably in 16 GB. Metric-only re-scoring runs on CPU.

## Data

Place the three annotation files and the COCO val2017 images as below. All
three benchmarks read from the **same image pool** — LVIS-minival and COCO-OOD
are subsets of COCO val2017, so the images are downloaded once.

```
datasets/
  images/val2017/                              # COCO val2017 images (~1 GB)
  annotations/
    instances_val2017.json                     # COCO
    lvis_v1_minival.json                       # LVIS-minival
    instances_val2017_coco_ood.json            # COCO-OOD
```

| file | source |
|---|---|
| `val2017` images, `instances_val2017.json` | [cocodataset.org/#download](https://cocodataset.org/#download) |
| `lvis_v1_minival.json` | [LVIS](https://www.lvisdataset.org/dataset) (minival split) |
| `instances_val2017_coco_ood.json` | [COCO-OOD](https://github.com/deeplearning-wisc/stud) |

```bash
mkdir -p datasets/images datasets/annotations
wget -c http://images.cocodataset.org/zips/val2017.zip -P datasets/images/
unzip -q datasets/images/val2017.zip -d datasets/images/
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P /tmp/
unzip -qj /tmp/annotations_trainval2017.zip 'annotations/instances_val2017.json' -d datasets/annotations/
```

Detector and LLM weights download automatically from HuggingFace on first use,
except YOLO-World, which Ultralytics fetches to `weights/`.

## Quickstart

One image, one configuration:

```bash
python run_benchmark.py --dataset coco --split test --split_size 8 \
    --llm qwen35-9b --detector yolo-world --evaluate
```

Each run writes every box, score and label to
`outputs/runs/<dataset>-<split>__<llm>__<detector>__<prompt>/`, so predictions
can be inspected or re-scored without re-running anything.

## Reproducing the paper

See **[REPRODUCE.md](REPRODUCE.md)** for the full protocol. Each paper result
maps to one command:

**This repository produces numbers, not plots.** No plotting or
image-rendering code ships here.

| paper | command | output |
|---|---|---|
| Fig. 3 — CAAP/SNAP, 28 cells on COCO-Val | `run_benchmark.py --dataset coco` then `scripts/maxdets_compare.py` | `results/main_coco.csv`, one row per cell |
| Fig. 4 — LVIS-minival | `--dataset lvis` | `results/main_lvis.csv` |
| Fig. 5 — COCO-OOD | `--dataset coco_ood` then `scripts/uap_pipeline_ood.py` | `results/main_coco_ood.csv`, `results/phase4_uap_coco_ood.csv` |
| Tables 6–7, 9 — threshold-wise CAAP/SNAP | `scripts/snap_sbert_all.py` | `results/snap_sbert.csv` |
| Tables 8, 10 — SemLoc-AP marginals | `scripts/semloc_all.py` then `scripts/build_semloc_report.py` | `results/semloc.csv`, `results/semloc_matrices.json` |
| Tables 4–5 — prompt sensitivity | `--prompt default\|minimal\|coco-specific`, and `scripts/run_lvis_prompt_ablation.sh` | `results/snap_sbert.csv` (`stage2`) |
| Table 2, 12 — unknown-object detection | `scripts/uap_pipeline_ood.py` | `results/phase4_uap_coco_ood.csv` |
| Fig. 6, Table 24 — MLLMs as direct detectors | `scripts/run_vlm_detect.py` then `scripts/rescore_vlm.py` | `results/phase3_vlm_*.csv` |
| Tables 19–20 — refinement loop and its controls | `scripts/run_vocab_feedback.py`, then `scripts/selective_refinement*.py` | `results/feedback_dev/`, `results/selective_refinement/` |
| Table 23 — confidence-threshold sweep | `scripts/threshold_cost.py` | `results/threshold_cost.csv` |
| Tables 25–28 — fixed-ontology reference | `scripts/run_fixed_ontology.sh` then `scripts/eval_fixed_ontology.py` | `results/fixed_ontology/` |
| Table 3 — GenerateU rows | external baseline, see [GENERATEU.md](GENERATEU.md) | — |
| Table 3 — <img alt="LAOD" src="assets/laod.svg" height="11"> and direct-MLLM rows | already produced by the rows above | see below |

Table 3 is three comparisons in one table, and only its GenerateU rows come
from outside this repository. The rest are specific cells of the main grid -
the same configuration Table 2 uses, not one picked for the comparison:

- <img alt="LAOD" src="assets/laod.svg" height="11"> (MLLM + YOLO-World) - one cell of `results/main_coco.csv` for CAAP,
  `results/snap_sbert.csv` for SNAP, `results/semloc.csv` for SemLoc-AP
- <img alt="LAOD" src="assets/laod.svg" height="11"> detector avg. range - each detector averaged over the MLLMs, then the
  min and max across the four detectors
- Direct MLLM - `results/phase3_vlm_coco.csv`, except SNAP, which comes from
  the `vlm` rows of `results/snap_sbert.csv`

Figures 3-5 are heatmaps over the MLLM x detector grid - one value per cell,
colour standing in for a number. The commands above write those numbers
directly, one row per cell:

- Fig. 3, CAAP and SNAP on COCO-Val - 28 cells - `results/main_coco.csv`
- Fig. 4, CAAP, SNAP and SemLoc-AP on LVIS-minival - 12 cells -
  `results/main_lvis.csv` and `results/semloc.csv`
- Fig. 5, CAAP on COCO-OOD - 28 cells - `results/main_coco_ood.csv`

`scripts/build_final_report.py` additionally lays each one out as a markdown
grid table, MLLMs down and detectors across, in the same shape as the figure.

The `results/` directory ships the frozen split id lists and our own output
tables, so you can diff your numbers against ours rather than only hoping they
match.

## Evaluation protocol

Four conventions apply to every metric and are the defaults in code:

- **maxDets = 100.** The top 100 detections per image by confidence, applied
  once to the prediction list so CAAP, SNAP, SemLoc-AP and U-AP see the
  identical set.
- **Matching visits predictions by descending confidence.** One-to-one; each
  prediction takes the best unclaimed ground truth.
- **SNAP uses Sentence-BERT** (`all-MiniLM-L6-v2`, bare labels). CLIP's text
  embeddings put unrelated COCO labels at a median cosine of 0.786, so a
  threshold of 0.50 admits every pair and the metric stops measuring naming;
  a label-shuffle control reproduces the score exactly. Sentence-BERT is
  trained for text-text similarity and does not have this failure.
- **Grounding DINO labels are read from the logits, not from the decoded
  phrase.** A box's confidence is the maximum over its (query, text-token)
  row, so the vocabulary entry owning that winning token is the entry the box
  grounds to. HuggingFace's post-processor instead rebuilds a phrase from the
  tokens above `text_threshold`, which at our operating point (score 0.03,
  text 0.30) returned an **empty** name for ~96% of retained boxes and
  fragments such as `'##et'` for `'tennis racket'` for most of the rest. We
  map each box to its owning entry — located by character offset, so an entry
  containing a period stays intact — and return the exact supplied string.
  Boxes, scores and thresholds are unchanged by construction: across all 36
  re-detected cells, boxes and scores are bit-identical and CAAP and U-AP
  move by 0.0000 (`results/relabel_verification.csv`). Only SNAP changes, and
  only on the two Grounding DINO backends; YOLO-World and OWLv2 associate a
  box to its entry by integer index and were never affected. See
  [LABEL_AUDIT.md](LABEL_AUDIT.md).

CAAP and SNAP share one interval structure: `LO` 0.50/0.55/0.60,
`MI` 0.65/0.70/0.75/0.80, `HI` 0.85/0.90/0.95, with the headline score the
unweighted mean over all ten thresholds.

### SemLoc-AP

CAAP asks whether an object was localised; SNAP asks whether the names were
appropriate. Neither asks whether those two successes happened on the **same
object**, and a box tightly on a dog but labelled `cat` satisfies both when a
cat exists elsewhere in the image. SemLoc-AP counts a prediction only when one
unmatched ground-truth instance satisfies both criteria at once:

```
IoU(box_i, box_j) >= theta   and   sim(label_i, label_j) >= tau
```

It is scored over the full 10×10 grid — the CAAP IoU thresholds against the
SNAP cosine thresholds — and reported as the mean over all 100 pairs. It is a
**pooled AP, not a per-class mAP**: detections are ranked once across the
dataset, so the numbers are not comparable in magnitude with published COCO
mAP, nor with CAAP or SNAP. `results/semloc_matrices.json` ships the full
matrix for every cell.

**SNAP and SemLoc-AP are not defined on COCO-OOD.** That dataset annotates
every object under one category, `unknow object`, so there is no name to match
against. Every entry point that accepts a dataset consults
`has_semantic_labels()` in `laod/data/annotations.py` and emits no columns
rather than filling them with zeros. Read COCO-OOD on U-AP, which is what it
was built for.

The one deliberate exception is `results/snap_sbert.csv`, which keeps 28
COCO-OOD rows. Those are not a result — they are the **shuffle control that
establishes the exclusion**, carried next to `chance_*` and `gain_*` columns
and reading `0.0000` throughout. No report displays them. They are the
evidence, so removing them would remove the proof.

Worth stating plainly, because the failure mode is silent and the output looks
like a measurement: scored against `unknow object` under **CLIP** text
embeddings, SNAP on COCO-OOD returns ~0.55 across all 28 cells. That number is
entirely CLIP's cone effect, and CLIP's own label-shuffle control reproduces
it to four decimals.

## Layout

```
laod/            the package: metrics, data, models, IO, visualisation
scripts/         tuning, scoring and report generation
tests/           148 tests
results/         frozen splits + our output tables
run_benchmark.py       generate vocabularies and ground them
eval_metrics_only.py   score an existing prediction file
LABEL_AUDIT.md         how each detector associates a box to a vocabulary entry
GENERATEU.md           rebuilding the external GenerateU baseline
```

`laod/feedback/` holds the vocabulary-revision loop: the detector's per-concept
support is summarised back to the MLLM, which rewrites the **vocabulary** and
the detector re-grounds it. The MLLM never sees or emits a coordinate. It is
reported in the appendix as a negative result — see
[REPRODUCE.md](REPRODUCE.md) §8.

## Citation

```bibtex
@article{mumcu2025laod,
  title   = {LLM-Guided Autonomous Object Detection for Open-World Understanding},
  author  = {Mumcu, Furkan and Jones, Michael J. and Cherian, Anoop and Yilmaz, Yasin},
  journal = {arXiv preprint arXiv:2507.10844},
  year    = {2025}
}
```
