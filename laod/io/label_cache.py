"""Persistent cache of LLM label generations.

A label list depends only on ``(image, llm, prompt)`` -- never on the detector.
Caching it turns the experiment grid from multiplicative into additive: seven
LLM passes feed twenty-eight detector passes, instead of twenty-eight full
pipeline runs. It also makes the detector swap *exactly* controlled, since every
detector sees the identical label list rather than a fresh stochastic sample.

Storage is append-only JSON Lines, one file per ``(dataset, llm, prompt)``, so a
crashed run resumes from where it stopped and a finished run stays readable by
anything that can read a text file.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterator

from laod.models.label_parser import ParsedLabels

logger = logging.getLogger(__name__)


class LabelCache:
    """Append-only per-image label store for one (dataset, llm, prompt) triple."""

    def __init__(self, root: str | Path, dataset: str, llm: str, prompt: str) -> None:
        self.dataset, self.llm, self.prompt = dataset, llm, prompt
        self.path = Path(root) / dataset / f"{llm}__{prompt}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[int, ParsedLabels] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        bad = 0
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._entries[int(rec["image_id"])] = ParsedLabels.from_json(rec)
                except (json.JSONDecodeError, KeyError):
                    bad += 1
        if bad:
            # A truncated final line is the normal signature of an interrupted run.
            logger.warning("%s: skipped %d unreadable line(s)", self.path.name, bad)
        logger.info("label cache %s: %d image(s) already generated",
                    self.path.name, len(self._entries))

    def __contains__(self, image_id: int) -> bool:
        return int(image_id) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[int, ParsedLabels]]:
        return iter(self._entries.items())

    def get(self, image_id: int) -> ParsedLabels | None:
        return self._entries.get(int(image_id))

    def put(self, image_id: int, parsed: ParsedLabels) -> None:
        """Record one generation and flush it immediately.

        Flushing per image rather than per batch is deliberate: an LLM pass is
        hours long, and losing it to a crash costs far more than the syscalls.
        """
        image_id = int(image_id)
        with self._lock:
            self._entries[image_id] = parsed
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"image_id": image_id, **parsed.to_json()}) + "\n")

    def missing(self, image_ids) -> list[int]:
        """Which of ``image_ids`` still need generating -- the resume list."""
        return [int(i) for i in image_ids if int(i) not in self._entries]

    def vocabulary(self) -> dict[str, int]:
        """Label -> occurrence count, for vocabulary analyses."""
        counts: dict[str, int] = {}
        for _, parsed in self._entries.items():
            for label in parsed.labels:
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
