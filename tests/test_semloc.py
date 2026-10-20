"""SemLoc-AP: the joint criterion, the masking rule, and label validity.

These use a stub encoder with hand-set similarities rather than MiniLM, so the
behaviour under test is the *matching rule* and not the checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from laod.io.predictions import ImageGroundTruth, ImagePredictions
from laod.metrics.semloc import SemLocEvaluator, is_valid_label


class StubEmbedder:
    """Orthonormal basis per label, so cosine is 1 on a match and 0 otherwise,
    with an explicit override table for the pairs a test cares about."""

    model_name = "stub"

    def __init__(self, pairs: dict[tuple[str, str], float] | None = None):
        self.pairs = {}
        for (a, b), v in (pairs or {}).items():
            self.pairs[(a, b)] = v
            self.pairs[(b, a)] = v
        self._vocab: list[str] = []

    def encode(self, labels):
        # Build a Gram matrix with the requested off-diagonal values and return
        # any factorisation of it; the evaluator only ever forms inner products.
        self._vocab = list(labels)
        n = len(labels)
        g = np.eye(n)
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i != j and (a, b) in self.pairs:
                    g[i, j] = self.pairs[(a, b)]
        w, v = np.linalg.eigh(g)
        return v * np.sqrt(np.clip(w, 0, None))


def gt(image_id, boxes, labels):
    return ImageGroundTruth(image_id, np.array(boxes, float).reshape(-1, 4), labels)


def pred(image_id, boxes, scores, labels):
    return ImagePredictions(image_id, np.array(boxes, float).reshape(-1, 4),
                            np.array(scores, float), labels)


GRID = dict(iou_thresholds=(0.50,), sem_thresholds=(0.50,))


def ap_at(g, p, emb, theta=0.50, tau=0.50):
    ev = SemLocEvaluator([g], emb, iou_thresholds=(theta,), sem_thresholds=(tau,))
    return ev.evaluate([p], progress=False).per_pair[(theta, tau)]


def test_right_box_right_name_is_a_true_positive():
    g = gt(1, [[0, 0, 10, 10]], ["dog"])
    p = pred(1, [[0, 0, 10, 10]], [0.9], ["dog"])
    assert ap_at(g, p, StubEmbedder()) == pytest.approx(1.0)


def test_right_box_wrong_name_is_rejected():
    """The motivating case: box on the dog, labelled 'cat', a cat elsewhere.

    CAAP would accept it (it localised the dog); SNAP would accept it (a cat is
    present and SNAP never looks at boxes). SemLoc-AP must not: no single
    instance satisfies both criteria.
    """
    g = gt(1, [[0, 0, 10, 10], [100, 100, 110, 110]], ["dog", "cat"])
    p = pred(1, [[0, 0, 10, 10]], [0.9], ["cat"])
    assert ap_at(g, p, StubEmbedder()) == pytest.approx(0.0)


def test_permissive_tau_accepts_the_same_prediction():
    """Below the similarity of cat/dog the metric has declared them
    interchangeable, so the same prediction is a true positive by definition."""
    emb = StubEmbedder({("cat", "dog"): 0.661})
    g = gt(1, [[0, 0, 10, 10], [100, 100, 110, 110]], ["dog", "cat"])
    p = pred(1, [[0, 0, 10, 10]], [0.9], ["cat"])
    # Two ground-truth objects and one prediction, so a match caps AP at ~0.5;
    # what distinguishes the two cases is matched vs not matched.
    assert ap_at(g, p, emb, tau=0.65) > 0.0
    assert ap_at(g, p, emb, tau=0.70) == pytest.approx(0.0)


def test_mask_before_match_not_after():
    """The trap: the best-IoU instance fails the semantic test while a slightly
    worse-overlapping one satisfies both.

    Checking the overall best-IoU instance and *then* its similarity would call
    this a false positive. Masking first finds the cat and matches it.
    """
    g = gt(1, [[0, 0, 10, 10], [0, 0, 10, 8]], ["dog", "cat"])
    p = pred(1, [[0, 0, 10, 10]], [0.9], ["cat"])
    iou_dog, iou_cat = 1.0, 0.8          # both clear theta=0.50
    assert iou_dog > iou_cat >= 0.50
    # Mask-first matches the cat (eligible, best IoU among eligible) -> AP > 0.
    # Best-IoU-then-check would land on the dog, fail the semantic test and
    # score exactly 0. The two rules are separated by this one case.
    assert ap_at(g, p, StubEmbedder()) > 0.0


def test_one_ground_truth_instance_is_claimed_once():
    g = gt(1, [[0, 0, 10, 10]], ["dog"])
    p = pred(1, [[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], ["dog", "dog"])
    r = SemLocEvaluator([g], StubEmbedder(), **GRID).evaluate([p], progress=False)
    assert r.per_pair[(0.50, 0.50)] < 1.0           # the duplicate is a false positive


def test_empty_and_whitespace_labels_never_match():
    assert not is_valid_label("")
    assert not is_valid_label("   ")
    assert not is_valid_label(None)
    assert is_valid_label("dog")
    g = gt(1, [[0, 0, 10, 10]], ["dog"])
    p = pred(1, [[0, 0, 10, 10]], [0.9], [""])
    r = SemLocEvaluator([g], StubEmbedder(), **GRID).evaluate([p], progress=False)
    assert r.per_pair[(0.50, 0.50)] == pytest.approx(0.0)
    assert r.n_invalid_labels == 1


def test_max_dets_is_applied_and_matches_the_other_metrics():
    from laod.io.predictions import DEFAULT_MAX_DETS
    assert SemLocEvaluator([], StubEmbedder()).max_dets == DEFAULT_MAX_DETS
    g = gt(1, [[0, 0, 10, 10]], ["dog"])
    boxes = [[0, 0, 10, 10]] + [[500, 500, 510, 510]] * 150
    scores = [0.01] + [0.9] * 150            # the true positive ranks last
    labels = ["dog"] * 151
    p = pred(1, boxes, scores, labels)
    r = SemLocEvaluator([g], StubEmbedder(), **GRID).evaluate([p], progress=False)
    assert r.n_pred == 100                   # capped, not 151


def test_macro_is_the_mean_over_the_whole_grid():
    g = gt(1, [[0, 0, 10, 10]], ["dog"])
    p = pred(1, [[0, 0, 10, 10]], [0.9], ["dog"])
    r = SemLocEvaluator([g], StubEmbedder()).evaluate([p], progress=False)
    m = r.matrix()
    assert m.shape == (10, 10)
    assert r.macro == pytest.approx(float(np.mean(m)))
    assert len(r.per_pair) == 100


def test_monotonicity_diagnostic_reports_shape():
    g = gt(1, [[0, 0, 10, 10]], ["dog"])
    p = pred(1, [[0, 0, 10, 10]], [0.9], ["dog"])
    r = SemLocEvaluator([g], StubEmbedder()).evaluate([p], progress=False)
    mono = r.monotonicity()
    assert mono["steps_in_iou"] == 90 and mono["steps_in_sem"] == 90
    assert set(mono) >= {"monotone_in_iou", "monotone_in_sem", "largest_increase"}
