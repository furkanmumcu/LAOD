# Reproducing the GenerateU baseline

GenerateU is the task-specific open-ended detector LAOD is compared against in
the paper. It is **not** part of this repository: it has its own CUDA
extensions, pins `transformers==4.25`, and needs a Python and PyTorch pair that
the LAOD environment cannot provide. Installing it alongside LAOD will break
LAOD.

This document is the recipe we used, so the comparison can be rebuilt from
scratch. No GenerateU code, weights, predictions or split manifests are vendored
here.

## 1. Upstream

| | |
|---|---|
| repository | <https://github.com/FoundationVision/GenerateU> |
| commit | `904e5337758d554e44ad50c2afa567c171166cd9` (2025-03-29) |
| checkpoint | `vg_grit5m_swinL.pth` — 6,563,306,767 bytes |
| checkpoint source | <https://huggingface.co/clin1223/GenerateU/resolve/main/vg_grit5m_swinL.pth> |
| config | `projects/DDETRS/configs/vg_grit5m_swinL.yaml`, **unmodified** |

Swin-L backbone, trained on VG + GRIT-5M. No file in the GenerateU repository
was modified for our run.

## 2. Environment

A separate conda environment. Do not install any of this into the LAOD
environment.

| | |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.1.2+cu121 |
| transformers | 4.25.0 (pinned by GenerateU) |
| numpy / Pillow | **1.26.4** / **9.5.0** |
| detectron2 | 0.6, the version vendored in the repo, built from source |
| compiled ops | `detectron2._C`, `MultiScaleDeformableAttention` (MSDeformAttn) |
| GPU | RTX 4090 (24 GB); ~14.3 GB peak |

Both CUDA extensions must be compiled for your card. On Ada (sm_89):

```bash
TORCH_CUDA_ARCH_LIST="8.9" pip install -e detectron2
TORCH_CUDA_ARCH_LIST="8.9" python projects/DDETRS/ddetrs/models/deformable_detr/backbone/ops/setup.py build install
```

### Why each pin is there

Three constraints are not in GenerateU's own `requirements.txt` and each one
fails in a way that is slow to diagnose:

- **Pillow < 10.** The vendored detectron2 uses `Image.LINEAR`, which Pillow 10
  removed. Fails at import.
- **numpy < 2.** torch 2.1.2 is built against the numpy 1.x ABI. numpy 2.x
  breaks the torch import itself.
- **A newer torch than the repo asks for.** The obvious route — reusing an
  existing detectron2 environment — does not work on recent GPUs: torch
  1.10/cu113 ships `['sm_37' … 'sm_86']` and simply cannot execute on sm_89.
  torch 2.1.2+cu121 compiled for your architecture is the floor.

**The Python version has a knock-on effect.** GenerateU needs Python 3.9, and
the LAOD package uses `dataclass(slots=True)`, which is 3.10+. So the GenerateU
process cannot import `laod` to read the frozen evaluation splits. Export the
split id list to JSON from the LAOD environment first and have the GenerateU
process read that file — do not re-derive the splits inside it, or the two sides
will not be scoring the same images.

A second detail that costs an afternoon: LVIS annotation entries carry
`coco_url`, not `file_name`. Resolve image paths accordingly.

## 3. How we scored it

**Ontology-free, by necessity.** GenerateU's published confidence couples the
benchmark ontology into the score itself:

```python
similarity   = (100.0 * text_features @ self.lvis_embed).softmax(dim=-1)
sim_values, sim_indices = similarity.topk(self.topk_for_mapping)
compose_prob = compose_prob * sim_values   # score scaled by LVIS similarity
scores_per_image = topk_values             # published score
labels_per_image = sim_indices             # label is an LVIS category id
```

So the published score is `objectness × IoU-quality × CLIP-similarity-to-an-LVIS-category`,
and the published label is an LVIS category rather than the generated name. **A
score that is both native and free of benchmark remapping does not exist in the
official code.** Scoring an ontology-free method on a remapped output would
measure the remapping.

We therefore stop one step earlier and keep:

- the free-form generated name, exactly as FlanT5 produced it;
- the box, in original image coordinates;
- `objectness × IoU-quality` — the native composite **before** ontology
  coupling — for ranking;
- the generation log-probability, as a second native signal.

No CLIP/BERT remapping and no projection into COCO or LVIS categories.

**Filtering.** `TEST.NUM_TEST_QUERIES = 300` with no score threshold, so the
model emits exactly 300 detections per image. The NMS branch in the official
`inference()` is gated on `max_num_inst > 300` and the config sets exactly 300,
so **no NMS runs** on this path. We added no filtering of our own.

**maxDets = 100** is applied once, inside the LAOD evaluator, exactly as for
every other result in this project. Predictions are deliberately *not* pre-capped
so the cap cannot be applied twice.

**Two label readings, neither privileged.** GenerateU's native label is the
three beam hypotheses joined by commas — `"bear, bears, bears fur"` — which is
not a single concept the way every other label in this project is.
`raw_beam_joined` keeps the string verbatim; `top_beam` takes the first
hypothesis. Both are reported.

## 4. What we got

Numbers to check a rebuild against. Same frozen splits, same evaluators, same
`maxDets = 100` as everything else in this repository.

**COCO-Val — 4,500 images, 32,814 objects**

| label variant | CAAP | SNAP | SemLoc-AP | det/img pre → post |
|---|---:|---:|---:|---|
| `raw_beam_joined` | 0.1976 | 0.0916 | 0.0513 | 300 → 100 |
| `top_beam` | 0.1976 | **0.1470** | **0.0821** | 300 → 100 |

**LVIS-minival — 4,327 images, 45,063 objects**

| label variant | CAAP | SNAP | SemLoc-AP | det/img pre → post |
|---|---:|---:|---:|---|
| `raw_beam_joined` | 0.0980 | 0.0653 | 0.0322 | 300 → 100 |
| `top_beam` | 0.0980 | **0.0984** | **0.0496** | 300 → 100 |

The paper quotes the `top_beam` row. That is the more favourable of the two
readings, and splitting the beam string is **our** post-processing, not
something the model emits — which is why both variants are here.

Two checks worth running on a rebuild:

- **CAAP must be identical across the two variants** (0.1976 COCO, 0.0980 LVIS).
  CAAP is class-agnostic and the variants share boxes and scores bit-for-bit, so
  any difference means the label split altered something it should not have.
- **Exactly 300 detections per image pre-cap, on every image.** A different
  number means NMS ran, or a score threshold was applied somewhere.

Beam formatting costs 34–38% of SNAP on its own (COCO 0.1470 → 0.0916), and the
interval breakdown localises it: COCO `snap_hi` collapses from 0.1012 to
**0.0009**. Embedding `"bear, bears, bears fur"` as one phrase moves it far
enough from `"bear"` that strict cosine thresholds stop matching. That is a
property of output formatting rather than of recognition.

Inference cost: ~80 minutes total for both datasets, ~0.55 s/image.
