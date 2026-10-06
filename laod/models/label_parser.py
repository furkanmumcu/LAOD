"""Turn an LLM's free-text reply into a label list, without losing evidence.

The original pipeline did ``response.split(',')`` with no stripping or casing, so
detections carried labels like ``' person'`` and ``'Pots'`` straight into CLIP.
Across seven models of differing verbosity that gets worse -- long descriptive
phrases fall into the out-of-vocabulary bucket and look like naming choices when
they are really formatting.

So parsing is explicit, its mode is recorded, and the raw reply is always kept.
Without the raw text it is impossible to tell "the model chose an unusual name"
from "our parser mangled the output", and that distinction is the mechanism the
whole study is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

ParseMode = Literal["legacy", "strict"]

#: Labels longer than this are kept but flagged: they are usually a model
#: narrating rather than naming, and they skew vocabulary analyses.
LONG_PHRASE_WORDS = 4

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_WS = re.compile(r"\s+")


def _strip_preamble(body: str) -> str:
    """Drop a leading sentence like "Here are the objects I see:".

    Only prose is removed. An earlier version dropped any first line without a
    comma, which silently ate the first item of every newline-separated list --
    the exact format a model produces when it ignores the comma instruction.
    A line is treated as preamble only when it ends in a colon, or reads as a
    sentence: several words, no comma, and no bullet marker.
    """
    if "\n" not in body:
        return body
    first, rest = body.split("\n", 1)
    if not rest.strip():
        return body
    head = first.strip()
    if head.endswith(":"):
        return rest
    if ("," not in head and not _BULLET.match(first)
            and len(head.split()) > LONG_PHRASE_WORDS):
        return rest
    return body


@dataclass(slots=True)
class ParsedLabels:
    """A parsed reply, with everything needed to audit the parse later."""

    raw: str
    labels: list[str]
    mode: ParseMode
    dropped: list[str] = field(default_factory=list)
    flags: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def to_json(self) -> dict:
        out = {"raw": self.raw, "labels": self.labels, "mode": self.mode}
        if self.dropped:
            out["dropped"] = self.dropped
        if self.flags:
            out["flags"] = self.flags
        return out

    @classmethod
    def from_json(cls, d: dict) -> "ParsedLabels":
        return cls(raw=d.get("raw", ""), labels=list(d["labels"]),
                   mode=d.get("mode", "strict"), dropped=list(d.get("dropped", [])),
                   flags=dict(d.get("flags", {})))


def parse_labels(raw: str, mode: ParseMode = "strict", *,
                 max_labels: int | None = None) -> ParsedLabels:
    """Split an LLM reply into labels.

    ``legacy`` reproduces the original exactly -- a bare comma split, keeping
    leading spaces and original casing. Use it only to regenerate old numbers.

    ``strict`` strips whitespace and bullet markers, collapses internal
    whitespace, lower-cases, drops empties, and de-duplicates while preserving
    order. Long phrases are kept but counted, since discarding them would hide
    exactly the behaviour worth measuring.
    """
    if mode == "legacy":
        return ParsedLabels(raw=raw, labels=raw.split(","), mode="legacy")
    if mode != "strict":
        raise ValueError(f"unknown parse mode {mode!r}")

    body = raw.strip()
    body = _strip_preamble(body)

    seen: dict[str, None] = {}
    dropped: list[str] = []
    flags = {"empty": 0, "duplicate": 0, "long_phrase": 0, "bulleted": 0}

    for piece in re.split(r"[,\n]", body):
        token = piece
        if _BULLET.match(token):
            flags["bulleted"] += 1
            token = _BULLET.sub("", token)
        token = _WS.sub(" ", token.strip().strip(".;:")).strip().lower()
        if not token:
            flags["empty"] += 1
            continue
        if token in seen:
            flags["duplicate"] += 1
            dropped.append(token)
            continue
        if len(token.split()) > LONG_PHRASE_WORDS:
            flags["long_phrase"] += 1
        seen[token] = None

    labels = list(seen)
    if max_labels is not None and len(labels) > max_labels:
        dropped.extend(labels[max_labels:])
        flags["over_limit"] = len(labels) - max_labels
        labels = labels[:max_labels]

    return ParsedLabels(raw=raw, labels=labels, mode="strict", dropped=dropped,
                        flags={k: v for k, v in flags.items() if v})
