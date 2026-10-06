"""The hyperparameter-selection holdout.

Detector confidence thresholds are the only parameter v2 fits to data, and the
threshold sweep showed the choice moves CAAP by 74-102% -- more than the choice
of detector does. Fitting it on images that are then reported on would be
selecting a high-leverage parameter on the test set.

So a fixed subset of COCO val2017 is reserved for tuning and excluded from every
reported number. The split is drawn once with a fixed seed and **frozen to an
explicit id list on disk**, so it cannot drift if a loader's ordering changes,
and anyone can verify which images were used.

The exclusion applies to all three benchmarks, not just COCO. LVIS-Minival
(4,809) and COCO-OOD (504) are strict subsets of COCO val2017, so a COCO-only
holdout would still leave ~10% of each tuned on.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, Literal, Sequence

from laod.config import PATHS

logger = logging.getLogger(__name__)

Role = Literal["tune", "test", "all"]

TUNING_SPLIT_PATH = PATHS.root / "results" / "tuning_split.json"
TUNING_SIZE = 500
TUNING_SEED = 20261001

#: A frozen subset *of the test split*, used where a full pass is unnecessary --
#: the prompt ablation, where the effect shows in the naming distribution and
#: does not need 4,500 images per condition. Drawn from test, never from the
#: tuning holdout, so its numbers are reportable.
ABLATION_SPLIT_PATH = PATHS.root / "results" / "ablation_subset.json"
ABLATION_SIZE = 500
ABLATION_SEED = 20261002

#: The prompt ablation is repeated on LVIS to test whether a prompt tuned for
#: one annotation scheme transfers. Its own frozen subset, drawn the same way
#: and with its own seed so the two are independent draws.
ABLATION_LVIS_PATH = PATHS.root / "results" / "ablation_subset_lvis.json"
ABLATION_LVIS_SEED = 20261003


def build_tuning_split(image_ids: Sequence[int], *, size: int = TUNING_SIZE,
                       seed: int = TUNING_SEED, path: Path | None = None) -> list[int]:
    """Draw and freeze the tuning split. Refuses to overwrite silently."""
    path = path or TUNING_SPLIT_PATH
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; the split is frozen on purpose. Delete it "
            "deliberately if you intend to re-draw, and expect every tuned "
            "threshold to need recomputing.")
    ids = sorted(random.Random(seed).sample(sorted(image_ids), size))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"seed": seed, "size": size, "source": "COCO val2017",
         "note": "hyperparameter-selection holdout; excluded from all reported "
                 "numbers on COCO-Val, LVIS-Minival and COCO-OOD",
         "image_ids": ids}, indent=1), encoding="utf-8")
    logger.info("froze tuning split: %d images -> %s", len(ids), path)
    return ids


def build_ablation_subset(test_image_ids: Sequence[int], *,
                          size: int = ABLATION_SIZE, seed: int = ABLATION_SEED,
                          path: Path | None = None) -> list[int]:
    """Draw and freeze the ablation subset from the *test* split."""
    path = path or ABLATION_SPLIT_PATH
    if path.exists():
        raise FileExistsError(f"{path} already exists; the subset is frozen.")
    tune = load_tuning_split()
    pool = sorted(set(int(i) for i in test_image_ids) - tune)
    if len(pool) < size:
        raise ValueError(f"only {len(pool)} test images available for a {size} subset")
    ids = sorted(random.Random(seed).sample(pool, size))
    path.write_text(json.dumps(
        {"seed": seed, "size": size, "drawn_from": "COCO val2017 test split",
         "note": "prompt-ablation subset; disjoint from the tuning holdout",
         "image_ids": ids}, indent=1), encoding="utf-8")
    logger.info("froze ablation subset: %d images -> %s", len(ids), path)
    return ids


def build_lvis_ablation_subset(test_image_ids: Sequence[int], *,
                               size: int = ABLATION_SIZE,
                               seed: int = ABLATION_LVIS_SEED,
                               path: Path | None = None) -> list[int]:
    """Draw and freeze the LVIS prompt-ablation subset from its test split."""
    path = path or ABLATION_LVIS_PATH
    if path.exists():
        raise FileExistsError(f"{path} already exists; the subset is frozen.")
    tune = load_tuning_split()
    pool = sorted(set(int(i) for i in test_image_ids) - tune)
    if len(pool) < size:
        raise ValueError(f"only {len(pool)} test images available for a {size} subset")
    ids = sorted(random.Random(seed).sample(pool, size))
    path.write_text(json.dumps(
        {"seed": seed, "size": size, "drawn_from": "LVIS-minival test split",
         "note": "prompt-ablation subset; disjoint from the tuning holdout",
         "image_ids": ids}, indent=1), encoding="utf-8")
    logger.info("froze LVIS ablation subset: %d images -> %s", len(ids), path)
    return ids


def load_ablation_subset(path: Path | None = None) -> set[int]:
    path = path or ABLATION_SPLIT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"no ablation subset at {path}")
    return {int(i) for i in json.loads(path.read_text(encoding="utf-8"))["image_ids"]}


def load_tuning_split(path: Path | None = None) -> set[int]:
    """The frozen tuning image ids."""
    path = path or TUNING_SPLIT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"no tuning split at {path}; create it with build_tuning_split()")
    return {int(i) for i in json.loads(path.read_text(encoding="utf-8"))["image_ids"]}


def apply_split(items: Iterable, role: Role = "test", *, key=lambda g: g.image_id):
    """Keep only the tuning images, only the test images, or everything."""
    if role == "all":
        return list(items)
    tune = load_tuning_split()
    want_tune = role == "tune"
    out = [x for x in items if (int(key(x)) in tune) == want_tune]
    logger.info("split %r: kept %d item(s)", role, len(out))
    return out
