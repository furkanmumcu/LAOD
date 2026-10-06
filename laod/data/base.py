"""Batched image loading shared by every dataset.

Decoding 5,000 JPEGs is pure CPU work that would otherwise idle the GPU between
generations, so it runs in worker processes. Images stay as PIL objects rather
than tensors because that is what every model wrapper in the roster expects, so
the collate function is a passthrough: batching here buys parallel decode, not
tensor stacking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from laod.io.predictions import ImageGroundTruth

logger = logging.getLogger(__name__)


class DetectionImageDataset(Dataset):
    """Pairs each ground-truth entry with its decoded image."""

    def __init__(self, ground_truth: Sequence[ImageGroundTruth], image_dir: str | Path,
                 *, file_names: dict[int, str] | None = None) -> None:
        self.ground_truth = list(ground_truth)
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"image directory not found: {self.image_dir}. See datasets/MANIFEST.md.")
        self.file_names = file_names or {}

    def __len__(self) -> int:
        return len(self.ground_truth)

    def file_name(self, image_id: int) -> str:
        # COCO val2017 file names are the zero-padded image id.
        return self.file_names.get(int(image_id), f"{int(image_id):012d}.jpg")

    def __getitem__(self, i: int) -> dict[str, Any]:
        gt = self.ground_truth[i]
        name = self.file_name(gt.image_id)
        image = Image.open(self.image_dir / name).convert("RGB")
        return {"image_id": gt.image_id, "image": image, "file_name": name,
                "width": image.width, "height": image.height,
                "gt_labels": list(gt.labels)}


def collate_passthrough(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep PIL images intact; models accept them directly."""
    return batch


def build_dataloader(dataset: DetectionImageDataset, *, batch_size: int = 8,
                     num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_passthrough,
                      pin_memory=False, persistent_workers=num_workers > 0)
