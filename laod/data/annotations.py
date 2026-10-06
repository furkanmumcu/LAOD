"""Ground-truth loading from COCO-format annotation files.

Reads boxes and category names only -- no images -- so metric evaluation never
depends on the image directory being present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from laod.io.predictions import ImageGroundTruth, to_xyxy

logger = logging.getLogger(__name__)

DatasetName = Literal["coco", "lvis", "coco_ood"]


def normalise_category(name: str, dataset: DatasetName) -> str:
    """Make a category name suitable for a text encoder.

    LVIS writes categories as ``pan_(for_cooking)``; the original pipeline
    replaced underscores with spaces before embedding, and reproducing its
    numbers requires doing the same. COCO names need no change.
    """
    return name.replace("_", " ") if dataset == "lvis" else name


def load_ground_truth(
    annotation_file: str | Path,
    dataset: DatasetName = "coco",
    *,
    limit: int | None = None,
    include_empty: bool = True,
) -> list[ImageGroundTruth]:
    """Load per-image ground truth from a COCO-format JSON file.

    Args:
        annotation_file: path to ``instances_val2017.json`` or an LVIS/COCO-OOD
            file in the same schema.
        dataset: selects category-name normalisation.
        limit: keep only the first N images, ordered by image id for
            determinism. Used for pilot subsets.
        include_empty: keep images that have no annotations. They contribute no
            ground truth but do let predictions on them count as false
            positives, which is the correct accounting.
    """
    path = Path(annotation_file)
    if not path.is_file():
        raise FileNotFoundError(f"annotation file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    names = {c["id"]: normalise_category(c["name"], dataset) for c in data["categories"]}
    by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        by_image.setdefault(int(ann["image_id"]), []).append(ann)

    image_ids = sorted(int(im["id"]) for im in data["images"])
    if not include_empty:
        image_ids = [i for i in image_ids if i in by_image]
    if limit is not None:
        image_ids = image_ids[:limit]

    out = []
    for image_id in image_ids:
        anns = by_image.get(image_id, [])
        boxes = (to_xyxy(np.array([a["bbox"] for a in anns], np.float32), "xywh")
                 if anns else np.zeros((0, 4), np.float32))
        out.append(ImageGroundTruth(image_id, boxes,
                                    [names[int(a["category_id"])] for a in anns]))
    logger.info("loaded %s ground truth: %d images, %d objects, %d categories",
                dataset, len(out), sum(map(len, out)), len(names))
    return out
