# LAOD: LLM-Guided Autonomous Object Detection for Open-World Understanding

[![arXiv](https://img.shields.io/badge/Read_Our_Paper-arXiv-red)](https://arxiv.org/abs/2507.10844)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Demo-blue.svg)](https://huggingface.co/spaces/fumucu/LAOD)

A multimodal LLM reads an image and writes an **image-specific detection
vocabulary**; an open-vocabulary detector then grounds those names. Nothing is
fine-tuned and no category list is supplied at inference time.

This repository contains the full implementation and the code to reproduce
every number in the paper. It evaluates **7 MLLMs x 4 open-vocabulary
detectors** on COCO-Val, LVIS-minival and COCO-OOD, and includes the two
metrics introduced in the paper, CAAP and SNAP.

> Datasets and model weights are **not** included — they are downloaded from
> their original sources, see [Data](#data). Everything else needed to
> reproduce the paper is here, including the frozen evaluation splits.

## Install

```bash
git clone https://github.com/furkanmumcu/LAOD.git && cd LAOD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add HF_TOKEN only if you use gated models
pytest tests/ -q              # 106 pass; 5 skip until data is provisioned
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

Render predictions and ground truth side by side:

```bash
python scripts/render_detections.py \
    --run outputs/runs/coco-test__qwen35-9b__yolo-world__default \
    --images 413247 --score 0.25
python scripts/render_ground_truth.py --dataset coco --images 413247
```

## Reproducing the paper

See **[REPRODUCE.md](REPRODUCE.md)** for the full protocol. Each paper result
maps to one command:

| paper | command |
|---|---|
| Fig. 3 — CAAP/SNAP, 28 MLLM x detector cells on COCO-Val | `run_benchmark.py --dataset coco` then `scripts/maxdets_compare.py` |
| LVIS-minival | `--dataset lvis` |
| COCO-OOD, U-AP/U-F1/U-PRE/U-REC | `--dataset coco_ood` then `scripts/uap_pipeline_ood.py` |
| MLLMs as direct detectors | `scripts/run_vlm_detect.py` |
| Table 4 — prompt sensitivity | `--prompt default\|minimal\|coco-specific`, and `scripts/run_lvis_prompt_ablation.sh` |
| Fig. 6 — qualitative | `scripts/pick_qualitative.py`, then the two render scripts |

The `results/` directory ships the frozen split id lists and our own output
tables, so you can diff your numbers against ours rather than only hoping they
match.

## Evaluation protocol

Three conventions apply to every metric and are the defaults in code:

- **maxDets = 100.** The top 100 detections per image by confidence, applied
  once to the prediction list so CAAP, SNAP and U-AP see the identical set.
- **Matching visits predictions by descending confidence.** One-to-one; each
  prediction takes the best unclaimed ground truth.
- **SNAP uses Sentence-BERT** (`all-MiniLM-L6-v2`, bare labels). CLIP's text
  embeddings put unrelated COCO labels at a median cosine of 0.786, so a
  threshold of 0.50 admits every pair and the metric stops measuring naming;
  a label-shuffle control reproduces the score exactly. Sentence-BERT is
  trained for text-text similarity and does not have this failure.

CAAP and SNAP share one interval structure: `LO` 0.50/0.55/0.60,
`MI` 0.65/0.70/0.75/0.80, `HI` 0.85/0.90/0.95, with the headline score the
unweighted mean over all ten thresholds.

## Layout

```
laod/            the package: metrics, data, models, IO, visualisation
scripts/         tuning, scoring, report and figure generation
tests/           111 tests
results/         frozen splits + our output tables
run_benchmark.py       generate vocabularies and ground them
eval_metrics_only.py   score an existing prediction file
```

## Citation

```bibtex
@article{mumcu2025laod,
  title   = {LLM-Guided Autonomous Object Detection for Open-World Understanding},
  author  = {Mumcu, Furkan and Jones, Michael J. and Cherian, Anoop and Yilmaz, Yasin},
  journal = {arXiv preprint arXiv:2507.10844},
  year    = {2025}
}
```
