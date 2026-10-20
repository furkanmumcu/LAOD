"""Semantic Localization Average Precision (SemLoc-AP).

CAAP asks whether an annotated object was localised. SNAP asks whether the
names produced were semantically appropriate. Neither asks whether those two
successes happened on the **same object**, and they can both be satisfied by a
prediction that got it wrong: a box tightly on the dog, labelled "cat", while a
cat exists elsewhere in the image. CAAP matches it to the dog on IoU alone;
SNAP matches it to the cat on label alone, because SNAP's affinity matrix is
built from labels and boxes never enter it.

SemLoc-AP closes that gap. A prediction is a true positive only when some single
unmatched ground-truth instance satisfies **both** criteria:

    IoU(box_i, box_j) >= theta   and   sim(label_i, label_j) >= tau

Among the instances that qualify, the prediction is assigned the one with the
highest IoU. The semantic criterion decides which objects are *eligible*; IoU
decides which eligible object the box actually corresponds to. Choosing by
similarity instead would let the label determine which object was localised,
inverting the question the metric asks.

Note what this does **not** claim. At a permissive tau the metric has itself
declared two names interchangeable -- MiniLM puts cat/dog at 0.661, so at
tau <= 0.65 a "cat" on a dog is semantically acceptable *by definition*, not by
accident. That is the reason the score is averaged over a range of tau rather
than asserted at one cutoff.

**Pooled AP, not per-class mAP.** Detections are pooled across the whole dataset
and ranked by confidence once; ``n_gt`` is the total number of ground-truth
objects in the dataset. There is no per-class AP and no averaging over classes,
so these numbers are not comparable with published COCO mAP, which the high-tau
corner of the matrix otherwise resembles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from laod.io.predictions import (DEFAULT_MAX_DETS, ImageGroundTruth,
                                 ImagePredictions, align, cap_detections)
from laod.metrics.common import (LEGACY_EPS, MatchOrder, average_precision,
                                 box_iou_matrix, greedy_match, show_progress)
from laod.metrics.grids import key
from laod.metrics.snap import TextEmbedder

logger = logging.getLogger(__name__)

#: The ten IoU thresholds CAAP uses, and the ten cosine thresholds SNAP uses.
#: Evaluated as a full Cartesian product: an IoU of 0.70 has no principled
#: correspondence to a cosine of 0.70, so pairing them on the diagonal would
#: assert a relationship that does not exist.
IOU_THRESHOLDS: tuple[float, ...] = tuple(key(0.50 + 0.05 * i) for i in range(10))
SEM_THRESHOLDS: tuple[float, ...] = tuple(key(0.50 + 0.05 * i) for i in range(10))


def is_valid_label(label) -> bool:
    """Whether a predicted label can participate in a semantic match at all.

    Empty and whitespace-only labels are excluded explicitly rather than left to
    the encoder. MiniLM happily embeds the empty string -- it lands within 0.410
    of some COCO name -- so relying on the cosine to reject it would make the
    rule an accident of the checkpoint rather than a property of the metric.
    """
    return bool(label is not None and str(label).strip())


@dataclass(frozen=True, slots=True)
class SemLocResult:
    """SemLoc-AP over the full (IoU, cosine) threshold grid."""

    per_pair: Mapping[tuple[float, float], float]
    iou_thresholds: tuple[float, ...]
    sem_thresholds: tuple[float, ...]
    n_gt: int = 0
    n_pred: int = 0
    n_invalid_labels: int = 0
    encoder: str = ""

    @property
    def macro(self) -> float:
        """The headline score: unweighted mean over all 100 threshold pairs."""
        return float(np.mean([self.per_pair[(t, u)]
                              for t in self.iou_thresholds
                              for u in self.sem_thresholds]))

    def matrix(self) -> np.ndarray:
        """The 10x10 grid, IoU down the rows and cosine across the columns."""
        return np.array([[self.per_pair[(t, u)] for u in self.sem_thresholds]
                         for t in self.iou_thresholds])

    def by_iou(self) -> dict[float, float]:
        """Marginal over cosine: how the score decays with localisation strictness."""
        return {t: float(np.mean([self.per_pair[(t, u)] for u in self.sem_thresholds]))
                for t in self.iou_thresholds}

    def by_sem(self) -> dict[float, float]:
        """Marginal over IoU: how the score decays with naming strictness."""
        return {u: float(np.mean([self.per_pair[(t, u)] for t in self.iou_thresholds]))
                for u in self.sem_thresholds}

    def monotonicity(self, tol: float = 1e-12) -> dict[str, object]:
        """Where the grid rises as a threshold is tightened.

        Raising either threshold shrinks each prediction's eligible set, which
        would normally only remove true positives. It can also *redirect* a
        match: with the highest-IoU rule, dropping a semantically marginal
        instance can free the object a later prediction needed, so the score is
        not guaranteed monotone in tau. This reports whether that happens on
        real data rather than assuming either way -- it is a diagnostic, and a
        violation is information about the data, not a failure.
        """
        m = self.matrix()
        d_iou = np.diff(m, axis=0)      # tightening IoU, down the rows
        d_sem = np.diff(m, axis=1)      # tightening cosine, across the columns
        up_iou = int((d_iou > tol).sum())
        up_sem = int((d_sem > tol).sum())
        return {
            "monotone_in_iou": up_iou == 0,
            "monotone_in_sem": up_sem == 0,
            "increases_in_iou": up_iou, "steps_in_iou": int(d_iou.size),
            "increases_in_sem": up_sem, "steps_in_sem": int(d_sem.size),
            "largest_increase": float(max(d_iou.max(initial=0.0),
                                          d_sem.max(initial=0.0))),
        }

    def report(self) -> str:
        lines = [f"SemLoc-AP  {self.macro:.4f}   ({self.n_pred} detections vs "
                 f"{self.n_gt} ground truths; encoder {self.encoder or 'n/a'})"]
        if self.n_invalid_labels:
            lines.append(f"  {self.n_invalid_labels} detection(s) carry an "
                         f"unusable label and can never match semantically")
        head = "   tau ->  " + " ".join(f"{u:6.2f}" for u in self.sem_thresholds)
        lines += ["", head]
        for t, row in zip(self.iou_thresholds, self.matrix()):
            lines.append(f"  IoU {t:.2f} " + " ".join(f"{v:6.4f}" for v in row))
        mono = self.monotonicity()
        lines += ["", f"  monotone in IoU: {mono['monotone_in_iou']} "
                      f"({mono['increases_in_iou']}/{mono['steps_in_iou']} steps rise)",
                  f"  monotone in tau: {mono['monotone_in_sem']} "
                  f"({mono['increases_in_sem']}/{mono['steps_in_sem']} steps rise)"]
        return "\n".join(lines)


class SemLocEvaluator:
    """Evaluate joint localisation-and-naming against a fixed ground-truth set.

    Args:
        ground_truth: per-image ground truth, with real category names.
        embedder: text encoder. Defaults to the same Sentence-BERT model SNAP
            uses, so the semantic criterion here and SNAP's are the same
            function of the same strings.
        iou_thresholds / sem_thresholds: the grid. Defaults to the ten CAAP IoU
            values and the ten SNAP cosine values.
        match_order: ``"score"`` visits the most confident prediction first,
            matching CAAP and SNAP.
        max_dets: detections kept per image, highest score first, applied once
            before anything is scored. Defaults to COCO's 100 -- the same cap
            CAAP and SNAP apply, so all three see an identical detection set.
    """

    def __init__(
        self,
        ground_truth: Sequence[ImageGroundTruth],
        embedder: TextEmbedder | None = None,
        *,
        iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
        sem_thresholds: Sequence[float] = SEM_THRESHOLDS,
        match_order: MatchOrder = "score",
        max_dets: int | None = DEFAULT_MAX_DETS,
        eps: float = LEGACY_EPS,
    ) -> None:
        if not iou_thresholds or not sem_thresholds:
            raise ValueError("both threshold grids must be non-empty")
        self.ground_truth = list(ground_truth)
        self.embedder = embedder or TextEmbedder()
        self.iou_thresholds = tuple(key(t) for t in iou_thresholds)
        self.sem_thresholds = tuple(key(t) for t in sem_thresholds)
        self.match_order = match_order
        self.max_dets = max_dets
        self.eps = eps

    def evaluate(
        self,
        predictions: Sequence[ImagePredictions],
        *,
        progress: bool = True,
    ) -> SemLocResult:
        gt, preds = align(self.ground_truth,
                          cap_detections(predictions, self.max_dets))
        n_gt = sum(len(g) for g in gt)
        if n_gt == 0:
            raise ValueError("ground truth is empty; SemLoc-AP is undefined")

        gt_labels = [g.labels for g in gt]
        pred_labels = [p.labels for p in preds]
        valid = [np.array([is_valid_label(l) for l in pl], dtype=bool)
                 for pl in pred_labels]
        n_invalid = int(sum((~v).sum() for v in valid))

        vocab = sorted({l for im in gt_labels for l in im} |
                       {l for im in pred_labels for l in im if is_valid_label(l)})
        emb = self.embedder.encode(vocab) if vocab else np.zeros((0, 1))
        index = {l: i for i, l in enumerate(vocab)}
        logger.info("SemLoc-AP vocabulary: %d unique label(s); %d detection(s) "
                    "with an unusable label", len(vocab), n_invalid)

        # IoU and cosine are computed once per image and reused across all 100
        # threshold pairs; only the masking and the matching are repeated.
        ious, sims, scores = [], [], []
        for g, p, v in zip(tqdm(gt, desc="SemLoc: affinity",
                                disable=not show_progress(progress), leave=False),
                           preds, valid):
            ious.append(box_iou_matrix(p.boxes, g.boxes))
            if len(p.labels) and len(g.labels):
                rows = np.full((len(p.labels), len(g.labels)), -np.inf)
                keep = np.flatnonzero(v)
                if keep.size:
                    rows[keep] = (emb[[index[p.labels[i]] for i in keep]]
                                  @ emb[[index[l] for l in g.labels]].T)
                sims.append(rows)
            else:
                sims.append(np.zeros((len(p.labels), len(g.labels))))
            scores.append(p.scores)
        flat_scores = np.concatenate(scores) if scores else np.zeros(0)

        per_pair: dict[tuple[float, float], float] = {}
        pairs = [(t, u) for t in self.iou_thresholds for u in self.sem_thresholds]
        for theta, tau in tqdm(pairs, desc="SemLoc: grid",
                               disable=not show_progress(progress), leave=False):
            flags = []
            for iou, sim, p in zip(ious, sims, preds):
                # The semantic criterion is applied as a *mask on the IoU
                # matrix*, not as a second test after the fact. Matching then
                # picks the highest IoU among semantically eligible instances,
                # which is the rule. Testing the overall best-IoU instance and
                # then checking its similarity would be a different metric: it
                # would reject a prediction whose best-IoU object fails the
                # semantic test even when a slightly worse-overlapping object
                # satisfies both. greedy_match already treats -inf as
                # unmatchable, so ineligible pairs simply never win.
                masked = np.where(sim >= tau, iou, -np.inf)
                flags.append(greedy_match(masked, theta, scores=p.scores,
                                          order=self.match_order))
            flat = np.concatenate(flags) if flags else np.zeros(0, bool)
            per_pair[(theta, tau)] = average_precision(flat_scores, flat, n_gt,
                                                       eps=self.eps)

        return SemLocResult(per_pair, self.iou_thresholds, self.sem_thresholds,
                            n_gt=n_gt, n_pred=int(flat_scores.size),
                            n_invalid_labels=n_invalid,
                            encoder=self.embedder.model_name)
