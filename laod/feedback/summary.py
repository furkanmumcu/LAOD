"""Summarise a round-1 detection set as per-concept evidence for the MLLM.

Two deliberate choices.

**Counts come from everything above the detector's own confidence threshold,
not from the top 100.** ``maxDets`` is a submission rule that exists so metrics
compare like with like; it says nothing about whether the detector found
support for a concept. Capping first would report "0 detections" for a concept
the detector did localise, merely because 100 higher-scoring boxes existed
elsewhere in the image.

**Detection count and maximum confidence only.** Mean confidence is omitted: at
YOLO-World's operating point (conf=0.001) it is dominated by near-zero tail
scores and would read as evidence of absence. Confidence scales are also
detector-specific -- YOLO-World at 0.001 and OWLv2 at 0.01 are not the same
quantity -- so the number is a relative signal within one image, never a
calibrated probability, and the prompt is written accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ConceptSupport:
    """What the detector returned for one proposed concept."""

    concept: str
    n_detections: int
    max_confidence: float | None

    def line(self) -> str:
        if self.n_detections == 0:
            return f"{self.concept}: 0 detections"
        noun = "detection" if self.n_detections == 1 else "detections"
        return (f"{self.concept}: {self.n_detections} {noun}, "
                f"max confidence {self.max_confidence:.2f}")


def concept_support(vocabulary: Sequence[str],
                    pred_labels: Sequence[str],
                    scores: Sequence[float]) -> list[ConceptSupport]:
    """Per-concept support, in the order the MLLM proposed the concepts.

    Concepts the detector never emitted still appear, with zero -- their absence
    is the most informative part of the summary and dropping them would hide it.
    """
    best: dict[str, float] = {}
    count: dict[str, int] = {}
    for label, score in zip(pred_labels, scores):
        key = str(label).strip().lower()
        if not key:
            continue
        count[key] = count.get(key, 0) + 1
        s = float(score)
        if s > best.get(key, float("-inf")):
            best[key] = s
    out = []
    for concept in vocabulary:
        key = str(concept).strip().lower()
        n = count.get(key, 0)
        out.append(ConceptSupport(concept, n, best.get(key) if n else None))
    return out


def feedback_block(support: Sequence[ConceptSupport]) -> str:
    """The text handed to the MLLM, one concept per line."""
    return "\n".join(s.line() for s in support)


def support_stats(support: Sequence[ConceptSupport]) -> Mapping[str, int]:
    """Counts used by the diagnostics, not shown to the model."""
    return {"n_concepts": len(support),
            "n_zero_detection": sum(1 for s in support if s.n_detections == 0)}
