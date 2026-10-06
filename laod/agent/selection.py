"""Deciding which detections deserve a second look.

The oracle ceilings in Phase 2a selected boxes using ground-truth IoU: a box was
a near-miss if ``0.25 <= IoU(box, gt) < 0.5``. An agent cannot do that -- the
ground truth is the thing being predicted. So selection has to run on observable
signals alone, and whatever recall it achieves caps the whole loop:

    achievable gain = oracle ceiling x selection recall x decision accuracy

This module fits that predictor on the hyperparameter holdout, where ground
truth may legitimately be consulted, and applies it unchanged at evaluation
time. The model is deliberately a logistic regression over six interpretable
features rather than anything stronger: the point is to characterise how much
signal is available, and a simple model that can be read off is more useful for
that than an accurate one that cannot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from laod.metrics.common import box_iou_matrix

logger = logging.getLogger(__name__)

FEATURES = ("score", "score_rank", "log_area", "aspect", "max_peer_iou", "label_count")
NEAR_MISS = (0.25, 0.50)


def features(boxes: np.ndarray, scores: np.ndarray, labels: list[str],
             width: int, height: int) -> np.ndarray:
    """Observable per-box features. Never touches ground truth."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, len(FEATURES)), np.float64)
    wh = np.clip(boxes[:, 2:] - boxes[:, :2], 1e-6, None)
    area = (wh[:, 0] * wh[:, 1]) / max(width * height, 1)
    aspect = wh[:, 0] / wh[:, 1]
    order = np.argsort(-scores)
    rank = np.empty(n); rank[order] = np.arange(n) / max(n - 1, 1)
    peer = box_iou_matrix(boxes, boxes)
    np.fill_diagonal(peer, 0.0)
    max_peer = peer.max(axis=1) if n > 1 else np.zeros(n)
    counts = {l: labels.count(l) for l in set(labels)}
    lab_n = np.array([counts[l] for l in labels], np.float64)
    return np.column_stack([scores, rank, np.log(area + 1e-9), np.log(aspect + 1e-9),
                            max_peer, np.log1p(lab_n)])


def near_miss_target(boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Ground-truth label for training: is this box a near-miss?"""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    if len(gt_boxes) == 0:
        return np.zeros(len(boxes), bool)
    best = box_iou_matrix(boxes, gt_boxes).max(axis=1)
    return (best >= NEAR_MISS[0]) & (best < NEAR_MISS[1])


@dataclass
class Selector:
    """Logistic regression over :data:`FEATURES`, fitted by gradient descent.

    Implemented directly rather than pulled from a dependency so the fitted
    weights stay inspectable and the module has no requirement beyond numpy.
    """

    w: np.ndarray | None = None
    b: float = 0.0
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 300,
            lr: float = 0.5) -> "Selector":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w = np.zeros(Z.shape[1]); self.b = 0.0
        # class weighting: near-misses are a small minority, and an unweighted
        # fit collapses to predicting "never"
        pos = max(y.sum(), 1); neg = max((~y).sum(), 1)
        wt = np.where(y, neg / pos, 1.0)
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = (p - y) * wt
            self.w -= lr * (Z.T @ g) / len(Z)
            self.b -= lr * g.mean()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("selector is not fitted")
        Z = (X - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))

    def describe(self) -> str:
        return "  ".join(f"{n}={v:+.2f}" for n, v in zip(FEATURES, self.w))
