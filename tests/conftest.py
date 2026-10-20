"""Shared fixtures. Heavy fixtures are session-scoped and skip when absent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laod.config import PATHS  # noqa: E402



@pytest.fixture(scope="session")
def coco_category_names() -> dict[int, str]:
    if not PATHS.coco_ann.is_file():
        pytest.skip(f"COCO annotations not provisioned at {PATHS.coco_ann}")
    data = json.loads(PATHS.coco_ann.read_text())
    return {c["id"]: c["name"] for c in data["categories"]}


