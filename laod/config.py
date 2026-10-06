"""Path and credential resolution for LAOD v2.

Every path resolves inside the repository by default. No module may hard-code
an absolute path; override via environment variables or an explicit ``Paths``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from ``.env`` into :data:`os.environ`.

    Existing environment variables win, so an exported value overrides the
    file. Commented lines are ignored. Missing file is not an error.
    """
    path = path or REPO_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.is_file():
        logger.debug("no .env at %s", path)
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not value:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All fields default to locations inside the repo."""

    root: Path = REPO_ROOT
    datasets: Path = REPO_ROOT / "datasets"
    weights: Path = REPO_ROOT / "weights"
    outputs: Path = REPO_ROOT / "outputs"

    @property
    def images(self) -> Path:
        return self.datasets / "images" / "val2017"

    @property
    def annotations(self) -> Path:
        return self.datasets / "annotations"

    @property
    def coco_ann(self) -> Path:
        return self.annotations / "instances_val2017.json"

    @property
    def lvis_ann(self) -> Path:
        return self.annotations / "lvis_v1_minival.json"

    @property
    def coco_ood_ann(self) -> Path:
        return self.annotations / "instances_val2017_coco_ood.json"

    @property
    def clip_dir(self) -> Path:
        return self.weights / "clip"

    @property
    def label_cache(self) -> Path:
        return self.outputs / "label_cache"

    @classmethod
    def from_env(cls) -> "Paths":
        load_dotenv()
        root = Path(os.environ.get("LAOD_ROOT", REPO_ROOT)).resolve()
        return cls(
            root=root,
            datasets=Path(os.environ.get("LAOD_DATA_ROOT", root / "datasets")),
            weights=Path(os.environ.get("LAOD_WEIGHTS_ROOT", root / "weights")),
            outputs=Path(os.environ.get("LAOD_OUTPUT_ROOT", root / "outputs")),
        )

    def require(self, *attrs: str) -> None:
        """Raise :class:`FileNotFoundError` if any named path is absent."""
        missing = [(a, getattr(self, a)) for a in attrs if not Path(getattr(self, a)).exists()]
        if missing:
            lines = "\n".join(f"  {a}: {p}" for a, p in missing)
            raise FileNotFoundError(
                f"required path(s) not found:\n{lines}\n"
                "See datasets/MANIFEST.md for how to provision them."
            )


PATHS: Final[Paths] = Paths.from_env()


def hf_token() -> str | None:
    """Return the HuggingFace token, or ``None`` if unset."""
    load_dotenv()
    return os.environ.get("HF_TOKEN") or None
