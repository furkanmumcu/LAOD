"""Unknown-object detection metrics for COCO-OOD.

COCO-OOD annotates a single category, ``unknow object`` [sic], so naming is
meaningless here and only localisation is scored. These are the metrics the
original paper's Table 2 uses to compare against OWOD and UOD baselines:
U-AP, U-F1, U-PRE and U-REC.

U-AP is CAAP at a single IoU threshold under a different name. The F1 /
precision / recall trio differs in kind: they are computed at one operating
point rather than integrated over the ranking, so they need a confidence
threshold, which is why ``score_threshold`` is explicit rather than implied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from laod.io.predictions import (DEFAULT_MAX_DETS, ImageGroundTruth,
                                 ImagePredictions, align, cap_detections)
from laod.metrics.common import (
    LEGACY_EPS,
    MatchOrder,
    average_precision,
    box_iou_matrix,
    greedy_match,
)

logger = logging.getLogger(__name__)

DEFAULT_IOU = 0.5
DEFAULT_SCORE_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class UAPResult:
    """Unknown-object detection at one IoU and one confidence threshold."""

    u_ap: float
    u_precision: float
    u_recall: float
    u_f1: float
    iou_threshold: float
    score_threshold: float
    n_gt: int
    n_pred: int
    n_considered: int

    def as_dict(self) -> Mapping[str, float]:
        return {"U-AP": self.u_ap, "U-F1": self.u_f1,
                "U-PRE": self.u_precision, "U-REC": self.u_recall}

    def report(self) -> str:
        return (f"Unknown-object detection "
                f"(IoU {self.iou_threshold:.2f}, score >= {self.score_threshold:.2f}; "
                f"{self.n_considered}/{self.n_pred} detections kept, "
                f"{self.n_gt} unknown objects)\n"
                f"  U-AP {self.u_ap:.4f}   U-F1 {self.u_f1:.4f}   "
                f"U-PRE {self.u_precision:.4f}   U-REC {self.u_recall:.4f}")


class UAPEvaluator:
    """Score class-agnostic detection of objects outside the known vocabulary.

    Args:
        ground_truth: per-image unknown-object boxes.
        iou_threshold: overlap required for a match.
        score_threshold: confidence cut for the F1/precision/recall trio.
            U-AP ignores it, since average precision integrates over the whole
            ranking; reporting both at once is why it is a separate knob.
    """

    def __init__(
        self,
        ground_truth: Sequence[ImageGroundTruth],
        *,
        iou_threshold: float = DEFAULT_IOU,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        match_order: MatchOrder = "score",
        max_dets: int | None = DEFAULT_MAX_DETS,
        eps: float = LEGACY_EPS,
    ) -> None:
        self.ground_truth = list(ground_truth)
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        self.match_order = match_order
        self.max_dets = max_dets
        self.eps = eps

    def evaluate(self, predictions: Sequence[ImagePredictions]) -> UAPResult:
        gt, preds = align(self.ground_truth,
                          cap_detections(predictions, self.max_dets))
        n_gt = sum(len(g) for g in gt)
        if n_gt == 0:
            raise ValueError("ground truth is empty; U-AP is undefined")

        flags, scores = [], []
        for g, p in zip(gt, preds):
            iou = box_iou_matrix(p.boxes, g.boxes)
            flags.append(greedy_match(iou, self.iou_threshold,
                                      scores=p.scores, order=self.match_order))
            scores.append(p.scores)
        flat_flags = np.concatenate(flags) if flags else np.zeros(0, bool)
        flat_scores = np.concatenate(scores) if scores else np.zeros(0)

        u_ap = average_precision(flat_scores, flat_flags, n_gt, eps=self.eps)

        keep = flat_scores >= self.score_threshold
        tp = int(flat_flags[keep].sum())
        fp = int((~flat_flags[keep]).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_gt
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)

        return UAPResult(u_ap=u_ap, u_precision=precision, u_recall=recall, u_f1=f1,
                         iou_threshold=self.iou_threshold,
                         score_threshold=self.score_threshold,
                         n_gt=n_gt, n_pred=int(flat_scores.size),
                         n_considered=int(keep.sum()))
