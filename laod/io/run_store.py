"""Durable, self-sufficient storage for a single benchmark run.

An LLM pass over 5,000 images costs hours, so its output is treated as the real
artefact and the metric as something cheap applied afterwards. The guiding rule
is that **a run directory must answer questions we have not thought of yet**: if
someone later wants to re-score naming with a sentence encoder instead of CLIP,
bucket labels by vocabulary novelty, or audit whether a strange label was the
model's choice or the parser's doing, everything needed is already on disk.

Concretely each run keeps, per image:

* the LLM's reply *verbatim*, alongside the parsed labels and the parse mode
* every detection: box, confidence, and the label string that produced it
* the ground-truth labels for that image, so text-space metrics need no join
* image dimensions, so boxes can be renormalised or areas computed later

and once per run: the full configuration, the annotation file with its checksum,
and the library versions in force. Layout is a directory with ``config.json`` and
an append-only ``detections.jsonl`` -- streamable, resumable, and readable by
anything that reads text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from laod.io.predictions import ImagePredictions

logger = logging.getLogger(__name__)

RUN_SCHEMA = "laod-run/1"


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    """md5 of a file, so a run can be tied to the exact annotations it used."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def environment() -> dict[str, str]:
    """Library versions that could plausibly change a number."""
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "ultralytics", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


@dataclass(slots=True)
class RunConfig:
    """Everything needed to say what a run was."""

    dataset: str
    llm: str
    detector: str
    prompt: str
    llm_model_id: str = ""
    detector_model_id: str = ""
    parse_mode: str = "strict"
    split_size: int | None = None
    detector_params: dict[str, Any] = field(default_factory=dict)
    generation_params: dict[str, Any] = field(default_factory=dict)
    annotation_file: str = ""
    annotation_md5: str = ""
    schema: str = RUN_SCHEMA
    created_at: str = ""
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.env:
            self.env = environment()

    @property
    def slug(self) -> str:
        parts = [self.dataset, self.llm, self.detector, self.prompt]
        if self.split_size:
            parts.append(f"n{self.split_size}")
        return "__".join(parts)


@dataclass(slots=True)
class ImageRecord:
    """One image's complete trace through the pipeline."""

    image_id: int
    file_name: str = ""
    width: int = 0
    height: int = 0
    raw_response: str = ""
    labels: list[str] = field(default_factory=list)
    parse_flags: dict[str, int] = field(default_factory=dict)
    boxes: list[list[float]] = field(default_factory=list)      # xyxy
    scores: list[float] = field(default_factory=list)
    pred_labels: list[str] = field(default_factory=list)
    gt_labels: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["boxes"] = [[round(float(v), 2) for v in b] for b in self.boxes]
        d["scores"] = [round(float(s), 6) for s in self.scores]
        return {k: v for k, v in d.items() if v not in ([], {}, "", 0)}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=int(d["image_id"]), file_name=d.get("file_name", ""),
            width=int(d.get("width", 0)), height=int(d.get("height", 0)),
            raw_response=d.get("raw_response", ""), labels=list(d.get("labels", [])),
            parse_flags=dict(d.get("parse_flags", {})),
            boxes=[list(map(float, b)) for b in d.get("boxes", [])],
            scores=[float(s) for s in d.get("scores", [])],
            pred_labels=list(d.get("pred_labels", [])),
            gt_labels=list(d.get("gt_labels", [])),
            timings=dict(d.get("timings", {})),
        )

    def to_predictions(self) -> ImagePredictions:
        return ImagePredictions(
            image_id=self.image_id,
            boxes=np.array(self.boxes, np.float32).reshape(-1, 4),
            scores=np.array(self.scores, np.float32),
            labels=list(self.pred_labels),
            raw_response=self.raw_response or None,
        )


class RunStore:
    """Append-only writer/reader for one run directory."""

    def __init__(self, root: str | Path, config: RunConfig, *, resume: bool = True) -> None:
        self.config = config
        self.dir = Path(root) / config.slug
        self.dir.mkdir(parents=True, exist_ok=True)
        self.detections_path = self.dir / "detections.jsonl"
        self.config_path = self.dir / "config.json"
        self._lock = threading.Lock()
        self._done: set[int] = set()

        if resume and self.detections_path.is_file():
            self._done = {r.image_id for r in self.read_records()}
            if self._done:
                logger.info("resuming %s: %d image(s) already written",
                            config.slug, len(self._done))
        elif not resume and self.detections_path.exists():
            self.detections_path.unlink()
        self.config_path.write_text(json.dumps(asdict(config), indent=2),
                                    encoding="utf-8")

    def __contains__(self, image_id: int) -> bool:
        return int(image_id) in self._done

    def missing(self, image_ids: Sequence[int]) -> list[int]:
        return [int(i) for i in image_ids if int(i) not in self._done]

    def append(self, record: ImageRecord) -> None:
        with self._lock:
            with self.detections_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_json()) + "\n")
            self._done.add(record.image_id)

    def read_records(self) -> Iterator[ImageRecord]:
        if not self.detections_path.is_file():
            return iter(())
        def gen():
            with self.detections_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield ImageRecord.from_json(json.loads(line))
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("%s: skipping unreadable line",
                                       self.detections_path.name)
        return gen()

    def summary(self) -> dict[str, Any]:
        n_img = n_det = n_gt = 0
        vocab: dict[str, int] = {}
        for r in self.read_records():
            n_img += 1
            n_det += len(r.scores)
            n_gt += len(r.gt_labels)
            for l in r.pred_labels:
                vocab[l] = vocab.get(l, 0) + 1
        return {"images": n_img, "detections": n_det, "gt_objects": n_gt,
                "unique_predicted_labels": len(vocab)}


def load_run(run_dir: str | Path) -> tuple[RunConfig, list[ImageRecord]]:
    """Read a finished run back: its configuration and every image record."""
    run_dir = Path(run_dir)
    cfg = RunConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    store = RunStore.__new__(RunStore)
    store.detections_path = run_dir / "detections.jsonl"
    return cfg, list(RunStore.read_records(store))


def run_to_predictions(run_dir: str | Path) -> list[ImagePredictions]:
    """Adapt a stored run into the metric layer's prediction form."""
    _, records = load_run(run_dir)
    return [r.to_predictions() for r in records]
