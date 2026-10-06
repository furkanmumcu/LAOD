"""Class-Agnostic Average Precision (CAAP).

Localisation quality with category labels discarded: a prediction is a true
positive when it overlaps an as-yet-unmatched ground-truth box by at least the
IoU threshold, regardless of what either is called.

Reported over three IoU intervals plus the standard 50:95 macro average. The
interval boundaries follow the original paper (LO 0.50-0.60, MI 0.65-0.80,
HI 0.85-0.95).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from laod.io.predictions import (DEFAULT_MAX_DETS, ImageGroundTruth,
                                 ImagePredictions, align, cap_detections,
                                 to_xyxy)
from laod.metrics.common import (
    LEGACY_EPS,
    show_progress,
    MatchOrder,
    average_precision,
    box_iou_matrix,
    greedy_match,
    interval_mean,
)

logger = logging.getLogger(__name__)

CAAP_LO: tuple[float, ...] = (0.50, 0.55, 0.60)
CAAP_MI: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80)
CAAP_HI: tuple[float, ...] = (0.85, 0.90, 0.95)
CAAP_ALL: tuple[float, ...] = CAAP_LO + CAAP_MI + CAAP_HI


def _key(t: float) -> float:
    """Round a threshold so float arithmetic cannot fracture dict keys."""
    return round(float(t), 4)


@dataclass(frozen=True, slots=True)
class CAAPResult:
    """CAAP at every evaluated IoU threshold, plus interval summaries."""

    per_threshold: Mapping[float, float]
    n_gt: int
    n_pred: int

    @property
    def lo(self) -> float:
        return interval_mean(self.per_threshold, [_key(t) for t in CAAP_LO])

    @property
    def mi(self) -> float:
        return interval_mean(self.per_threshold, [_key(t) for t in CAAP_MI])

    @property
    def hi(self) -> float:
        return interval_mean(self.per_threshold, [_key(t) for t in CAAP_HI])

    @property
    def macro(self) -> float:
        """CAAP@.50:.95 -- the mean over all ten standard thresholds."""
        return interval_mean(self.per_threshold, [_key(t) for t in CAAP_ALL])

    def as_dict(self) -> dict[str, float]:
        out = {f"CAAP@{t:.2f}": v for t, v in sorted(self.per_threshold.items())}
        out.update(CAAP_LO=self.lo, CAAP_MI=self.mi, CAAP_HI=self.hi,
                   CAAP_50_95=self.macro)
        return out

    def report(self) -> str:
        lines = [
            f"CAAP  ({self.n_pred} detections vs {self.n_gt} ground truths)",
            "  " + "  ".join(f"@{t:.2f}={v:.4f}" for t, v in sorted(self.per_threshold.items())),
            f"  LO {self.lo:.4f}   MI {self.mi:.4f}   HI {self.hi:.4f}"
            f"   50:95 {self.macro:.4f}",
        ]
        return "\n".join(lines)


class CAAPEvaluator:
    """Evaluate class-agnostic localisation against a fixed ground-truth set.

    IoU matrices are computed once per image and reused across every threshold,
    which is where this differs in cost -- not in result -- from the original
    implementation that recomputed them inside each threshold loop.

    Args:
        ground_truth: per-image ground truth.
        iou_thresholds: thresholds to evaluate. Defaults to the standard ten.
        match_order: order in which predictions claim ground truth. Matching
            is one-to-one and first-come-first-served, so this decides which
            prediction wins a contested object. ``"score"`` (the default)
            visits the most confident first, which is what the metric
            definition specifies. ``"given"`` visits them in the order the
            detector returned them and reproduces the original implementation;
            it is equivalent for detectors whose output is already ranked
            (YOLO-World) and materially wrong for those whose output is not
            (Grounding DINO, OWLv2 -- ~1% of their images come out sorted).
        max_dets: detections kept per image, highest score first, before
            anything is scored. Defaults to COCO's 100, so results are
            comparable to published numbers; ``None`` disables the cap and is
            what reproducing the original table requires.
        eps: denominator epsilon, kept for numerical parity with the original.
    """

    def __init__(
        self,
        ground_truth: Sequence[ImageGroundTruth],
        iou_thresholds: Sequence[float] = CAAP_ALL,
        *,
        match_order: MatchOrder = "score",
        max_dets: int | None = DEFAULT_MAX_DETS,
        eps: float = LEGACY_EPS,
    ) -> None:
        if not iou_thresholds:
            raise ValueError("at least one IoU threshold is required")
        self.ground_truth = list(ground_truth)
        self.iou_thresholds = [_key(t) for t in iou_thresholds]
        self.match_order = match_order
        self.max_dets = max_dets
        self.eps = eps

    def evaluate(
        self,
        predictions: Sequence[ImagePredictions] | Sequence[Mapping],
        *,
        progress: bool = True,
    ) -> CAAPResult:
        preds = cap_detections([_coerce(p) for p in predictions], self.max_dets)
        gt, preds = align(self.ground_truth, preds)

        n_gt = sum(len(g) for g in gt)
        if n_gt == 0:
            raise ValueError("ground truth is empty; CAAP is undefined")

        ious, scores = [], []
        for g, p in zip(tqdm(gt, desc="CAAP: IoU", disable=not show_progress(progress), leave=False), preds):
            ious.append(box_iou_matrix(p.boxes, g.boxes))
            scores.append(p.scores)
        flat_scores = np.concatenate(scores) if scores else np.zeros(0)

        per_threshold: dict[float, float] = {}
        for thr in tqdm(self.iou_thresholds, desc="CAAP: thresholds",
                        disable=not show_progress(progress), leave=False):
            flags = np.concatenate([
                greedy_match(m, thr, scores=p.scores, order=self.match_order)
                for m, p in zip(ious, preds)
            ]) if ious else np.zeros(0, bool)
            per_threshold[thr] = average_precision(flat_scores, flags, n_gt, eps=self.eps)

        return CAAPResult(per_threshold, n_gt, int(flat_scores.size))


def _coerce(p: ImagePredictions | Mapping) -> ImagePredictions:
    """Accept either the dataclass or the plain dict form from the spec."""
    if isinstance(p, ImagePredictions):
        return p
    boxes = to_xyxy(np.asarray(p["boxes"], np.float32), p.get("box_format", "xyxy"))
    return ImagePredictions(
        image_id=int(p["image_id"]),
        boxes=boxes,
        scores=np.asarray(p["scores"], np.float32),
        labels=list(p.get("labels", [""] * len(boxes))),
    )
