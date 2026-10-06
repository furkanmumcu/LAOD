"""Shared matching and precision-recall utilities for CAAP and SNAP.

Both metrics have the same shape: build a per-image affinity matrix between
predictions and ground truths (IoU for CAAP, text-embedding cosine for SNAP),
greedily assign each prediction to at most one unused ground truth above a
threshold, then integrate a dataset-level precision-recall curve ranked by
detector confidence. Only the affinity differs, so it lives here once.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

MatchOrder = Literal["given", "score"]

# The original implementation added this to precision and recall denominators.
# Reproducing its published numbers requires keeping it; it is exposed rather
# than buried so the choice is visible at the call site.
LEGACY_EPS = 1e-8


def show_progress(requested: bool = True) -> bool:
    """Whether to draw a progress bar: only when asked *and* on a terminal."""
    return bool(requested) and sys.stderr.isatty()


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Returns an ``(len(a), len(b))`` array; empty inputs give an empty matrix.
    Degenerate boxes (zero union) score 0 rather than raising.
    """
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), np.float64)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def greedy_match(
    affinity: np.ndarray,
    threshold: float,
    *,
    scores: np.ndarray | None = None,
    order: MatchOrder = "given",
) -> np.ndarray:
    """Assign predictions to ground truths, one-to-one, above ``threshold``.

    Rows are visited in ``order`` -- ``"given"`` keeps the caller's ordering
    (what the original code did, since detector output is already ranked),
    ``"score"`` sorts by descending confidence as the metric definition
    specifies. Each row takes the highest-affinity ground truth not yet taken;
    if that best value is below ``threshold`` the row is a false positive.

    Returns a boolean array, ``True`` where the prediction is a true positive.
    """
    n_pred, n_gt = affinity.shape
    flags = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or n_gt == 0:
        return flags

    if order == "score":
        if scores is None:
            raise ValueError("order='score' requires scores")
        visit = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    elif order == "given":
        visit = np.arange(n_pred)
    else:
        raise ValueError(f"unknown match order {order!r}")

    used = np.zeros(n_gt, dtype=bool)
    for i in visit:
        row = affinity[i].copy()
        row[used] = -np.inf
        j = int(np.argmax(row))
        if row[j] >= threshold and np.isfinite(row[j]):
            used[j] = True
            flags[i] = True
    return flags


@dataclass(frozen=True, slots=True)
class PRCurve:
    """A dataset-level precision-recall curve, ranked by confidence."""

    precision: np.ndarray
    recall: np.ndarray
    n_gt: int
    n_pred: int

    @property
    def n_tp(self) -> int:
        return int(self.precision.size and round(self.recall[-1] * self.n_gt))


def pr_curve(
    scores: Sequence[float] | np.ndarray,
    flags: Sequence[bool] | np.ndarray,
    n_gt: int,
    *,
    eps: float = LEGACY_EPS,
    recall_eps: float = 0.0,
) -> PRCurve:
    """Build the precision-recall curve from dataset-wide ranked detections.

    ``eps`` guards the precision denominator, which is always safe to add since
    it is only ever reached when there is at least one detection.

    ``recall_eps`` guards the recall denominator and defaults to zero, because
    adding it has a visible cost: recall then asymptotes just below 1.0, the
    top interpolation level finds nothing, and a perfect detector scores
    ~0.990 instead of 1.0. The original CAAP used no epsilon here and the
    original SNAP used 1e-8, so each evaluator passes its own value to stay
    faithful rather than inheriting one default.
    """
    scores = np.asarray(scores, np.float64)
    flags = np.asarray(flags, bool)
    if scores.shape != flags.shape:
        raise ValueError(f"scores {scores.shape} and flags {flags.shape} disagree")

    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(flags[order])
    fp = np.cumsum(~flags[order])
    precision = tp / (tp + fp + eps)
    recall = (tp / (n_gt + recall_eps) if n_gt
              else np.zeros_like(tp, dtype=np.float64))
    return PRCurve(precision, recall, n_gt, len(scores))


def average_precision_101(curve: PRCurve) -> float:
    """101-point interpolated average precision (COCO convention).

    At each of 101 evenly spaced recall levels take the maximum precision
    attained at that recall or beyond, then average. Taking the forward maximum
    is what makes the curve monotonically non-increasing, so no separate
    envelope pass is needed.
    """
    if curve.precision.size == 0 or curve.n_gt == 0:
        return 0.0
    levels = np.linspace(0.0, 1.0, 101)
    # reachable[k] -> max precision over the suffix where recall >= levels[k]
    envelope = np.maximum.accumulate(curve.precision[::-1])[::-1]
    idx = np.searchsorted(curve.recall, levels, side="left")
    out = np.where(idx < envelope.size, envelope[np.minimum(idx, envelope.size - 1)], 0.0)
    return float(out.mean())


def average_precision(
    scores: Sequence[float] | np.ndarray,
    flags: Sequence[bool] | np.ndarray,
    n_gt: int,
    *,
    eps: float = LEGACY_EPS,
    recall_eps: float = 0.0,
) -> float:
    """Convenience wrapper: ranked detections in, 101-point AP out."""
    return average_precision_101(
        pr_curve(scores, flags, n_gt, eps=eps, recall_eps=recall_eps))


def interval_mean(values: dict[float, float], thresholds: Sequence[float]) -> float:
    """Mean of ``values`` over ``thresholds``; missing keys are an error."""
    missing = [t for t in thresholds if t not in values]
    if missing:
        raise KeyError(f"no value computed for threshold(s) {missing}")
    return float(np.mean([values[t] for t in thresholds]))
