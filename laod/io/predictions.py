"""Shared prediction / ground-truth schema and (de)serialisation.

The in-memory containers here are the single interchange format between
dataloaders, detectors and metrics. A versioned JSON form on disk lets
``eval_metrics_only.py`` score runs produced elsewhere, and a converter reads
the legacy ``.npy`` dumps from the original paper so those runs stay evaluable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "laod-predictions/1"
BoxFormat = Literal["xyxy", "xywh"]


def to_xyxy(boxes: np.ndarray, fmt: BoxFormat = "xyxy") -> np.ndarray:
    """Return ``boxes`` as float32 ``[x1, y1, x2, y2]``."""
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if fmt == "xywh":
        arr = arr.copy()
        arr[:, 2] += arr[:, 0]
        arr[:, 3] += arr[:, 1]
    elif fmt != "xyxy":
        raise ValueError(f"unknown box format {fmt!r}")
    return arr


@dataclass(slots=True)
class ImageGroundTruth:
    """Ground-truth boxes and category names for one image."""

    image_id: int
    boxes: np.ndarray                      # (M, 4) xyxy float32
    labels: list[str]                      # M category names

    def __post_init__(self) -> None:
        self.boxes = to_xyxy(self.boxes)
        if len(self.labels) != len(self.boxes):
            raise ValueError(
                f"image {self.image_id}: {len(self.boxes)} boxes vs {len(self.labels)} labels"
            )

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass(slots=True)
class ImagePredictions:
    """Detections for one image.

    ``labels`` holds the string the detector emitted, verbatim. ``raw_response``
    optionally carries the LLM's unparsed reply so naming analyses can tell a
    parsing artefact apart from a genuine naming choice.
    """

    image_id: int
    boxes: np.ndarray                      # (K, 4) xyxy float32
    scores: np.ndarray                     # (K,) float32
    labels: list[str]
    raw_response: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.boxes = to_xyxy(self.boxes)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        n = len(self.boxes)
        if not (len(self.scores) == len(self.labels) == n):
            raise ValueError(
                f"image {self.image_id}: ragged arrays "
                f"boxes={n} scores={len(self.scores)} labels={len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.boxes)


#: COCO's evaluation protocol keeps only this many detections per image.
#: ``pycocotools`` reports AP at maxDets=100, so a score computed without the
#: cap is not comparable to a published one.
DEFAULT_MAX_DETS = 100


def cap_detections(preds: Sequence[ImagePredictions],
                   max_dets: int | None = DEFAULT_MAX_DETS
                   ) -> list[ImagePredictions]:
    """Keep the top ``max_dets`` detections per image by score.

    The cap belongs to the submission, not to any one metric: it is applied
    once, and every metric is then computed over the same detection list.
    ``None`` disables it, which is what reproducing the original table needs.
    """
    if not max_dets:
        return list(preds)
    out = []
    for p in preds:
        if len(p.scores) <= max_dets:
            out.append(p)
            continue
        keep = np.argsort(-np.asarray(p.scores, dtype=float),
                          kind="stable")[:max_dets]
        keep.sort()                      # preserve the detector's own ordering
        out.append(ImagePredictions(p.image_id, p.boxes[keep], p.scores[keep],
                                    [p.labels[i] for i in keep]))
    return out


def align(
    gt: Sequence[ImageGroundTruth],
    preds: Sequence[ImagePredictions],
) -> tuple[list[ImageGroundTruth], list[ImagePredictions]]:
    """Pair ground truth and predictions by ``image_id``.

    Images with ground truth but no predictions get an empty prediction entry,
    so their objects still count toward recall. Predictions for images absent
    from the ground truth are dropped with a warning.
    """
    by_id = {p.image_id: p for p in preds}
    unknown = set(by_id) - {g.image_id for g in gt}
    if unknown:
        logger.warning("dropping predictions for %d image(s) absent from GT", len(unknown))
    out_gt, out_pred = [], []
    for g in gt:
        out_gt.append(g)
        out_pred.append(
            by_id.get(
                g.image_id,
                ImagePredictions(g.image_id, np.zeros((0, 4), np.float32),
                                 np.zeros((0,), np.float32), []),
            )
        )
    return out_gt, out_pred


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------

def save_predictions(preds: Iterable[ImagePredictions], path: str | Path,
                     meta: dict[str, Any] | None = None) -> Path:
    """Write predictions as versioned JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "meta": meta or {},
        "images": [
            {
                "image_id": int(p.image_id),
                "boxes": np.asarray(p.boxes, np.float32).round(2).tolist(),
                "scores": np.asarray(p.scores, np.float32).round(6).tolist(),
                "labels": list(p.labels),
                **({"raw_response": p.raw_response} if p.raw_response else {}),
            }
            for p in preds
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("wrote %s (%d images)", path, len(payload["images"]))
    return path


def load_predictions(path: str | Path) -> tuple[list[ImagePredictions], dict[str, Any]]:
    """Read predictions from versioned JSON, or from a COCO-style result list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return _from_coco_results(data), {"schema": "coco-results"}
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        logger.warning("schema %r != expected %r; attempting to read anyway",
                       schema, SCHEMA_VERSION)
    preds = [
        ImagePredictions(
            image_id=int(d["image_id"]),
            boxes=np.asarray(d["boxes"], np.float32).reshape(-1, 4),
            scores=np.asarray(d["scores"], np.float32),
            labels=list(d["labels"]),
            raw_response=d.get("raw_response"),
        )
        for d in data["images"]
    ]
    return preds, data.get("meta", {})


def _from_coco_results(rows: list[dict[str, Any]]) -> list[ImagePredictions]:
    """Group flat COCO-style ``[{image_id, bbox, score, category_name}]`` rows."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(int(r["image_id"]), []).append(r)
    out = []
    for image_id, items in buckets.items():
        out.append(ImagePredictions(
            image_id=image_id,
            boxes=to_xyxy(np.array([i["bbox"] for i in items], np.float32), "xywh"),
            scores=np.array([i.get("score", 1.0) for i in items], np.float32),
            labels=[str(i.get("category_name", i.get("category_id", ""))) for i in items],
        ))
    return out


# --------------------------------------------------------------------------
# Legacy .npy dumps from the original paper
# --------------------------------------------------------------------------

def load_legacy_npy(
    run_dir: str | Path,
    *,
    category_names: dict[int, str] | None = None,
) -> tuple[list[ImageGroundTruth], list[ImagePredictions]]:
    """Read an original ``{all_gt,all_dt}.npy`` pair.

    The dumps are index-aligned lists of per-image rows:
    ``gt = [x1, y1, x2, y2, category_id]`` and
    ``dt = [x1, y1, x2, y2, score, label]``.

    They carry no image ids, so positional index is used as a synthetic id;
    this is faithful because both files were written from one ordered pass.
    Prediction label strings are kept verbatim -- the original pipeline never
    stripped or lower-cased them, and reproducing its numbers requires the
    exact strings that were embedded.
    """
    run_dir = Path(run_dir)
    gt_raw = np.load(run_dir / "all_gt.npy", allow_pickle=True)
    dt_raw = np.load(run_dir / "all_dt.npy", allow_pickle=True)
    if len(gt_raw) != len(dt_raw):
        raise ValueError(f"{run_dir}: {len(gt_raw)} GT images vs {len(dt_raw)} DT images")

    gt, preds = [], []
    for i, (g_rows, d_rows) in enumerate(zip(gt_raw, dt_raw)):
        g_boxes = np.array([r[:4] for r in g_rows], np.float32).reshape(-1, 4)
        g_ids = [int(r[4]) for r in g_rows]
        g_labels = [category_names[c] if category_names else str(c) for c in g_ids]
        gt.append(ImageGroundTruth(i, g_boxes, g_labels))

        d_boxes = np.array([r[:4] for r in d_rows], np.float32).reshape(-1, 4)
        preds.append(ImagePredictions(
            image_id=i,
            boxes=d_boxes,
            scores=np.array([float(r[4]) for r in d_rows], np.float32),
            labels=[str(r[5]) for r in d_rows],
        ))
    logger.info("loaded legacy run %s: %d images, %d GT, %d detections",
                run_dir.name, len(gt), sum(map(len, gt)), sum(map(len, preds)))
    return gt, preds
