"""COCO-OOD loader."""

from __future__ import annotations

from pathlib import Path

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.base import DetectionImageDataset, build_dataloader

DATASET = "coco_ood"


def load(annotation_file: str | Path | None = None,
         image_dir: str | Path | None = None,
         *, split_size: int | None = None):
    """Return (ground_truth, dataset) for COCO-OOD.

    All three benchmarks index into the same COCO val2017 image pool, so
    ``image_dir`` defaults to the single in-repo copy.
    """
    ann = Path(annotation_file or PATHS.coco_ood_ann)
    gt = load_ground_truth(ann, DATASET, limit=split_size)
    return gt, DetectionImageDataset(gt, image_dir or PATHS.images)


__all__ = ["load", "build_dataloader", "DATASET"]
