"""Shared fixtures. Heavy fixtures are session-scoped and skip when absent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laod.config import PATHS  # noqa: E402

LEGACY_RUN = PATHS.root / "reference/original_runs/coco_ours_results"


@pytest.fixture(scope="session")
def coco_category_names() -> dict[int, str]:
    if not PATHS.coco_ann.is_file():
        pytest.skip(f"COCO annotations not provisioned at {PATHS.coco_ann}")
    data = json.loads(PATHS.coco_ann.read_text())
    return {c["id"]: c["name"] for c in data["categories"]}


@pytest.fixture(scope="session")
def legacy_run(coco_category_names):
    """The original paper's COCO-Val predictions, the reproduction anchor."""
    if not (LEGACY_RUN / "all_dt.npy").is_file():
        pytest.skip(f"legacy dumps not provisioned at {LEGACY_RUN}")
    from laod.io.predictions import load_legacy_npy
    return load_legacy_npy(LEGACY_RUN, category_names=coco_category_names)
