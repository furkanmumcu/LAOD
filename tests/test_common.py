"""Unit tests for the shared matching and PR machinery."""

from __future__ import annotations

import numpy as np
import pytest

from laod.metrics.common import (
    average_precision,
    box_iou_matrix,
    greedy_match,
    pr_curve,
)


class TestIoU:
    def test_identical_boxes_score_one(self):
        b = np.array([[0, 0, 10, 10]], float)
        assert box_iou_matrix(b, b)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self):
        a = np.array([[0, 0, 10, 10]], float)
        b = np.array([[20, 20, 30, 30]], float)
        assert box_iou_matrix(a, b)[0, 0] == 0.0

    def test_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], float)      # area 100
        b = np.array([[5, 0, 15, 10]], float)      # area 100, inter 50
        assert box_iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)

    def test_empty_inputs_give_empty_matrix(self):
        assert box_iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)
        assert box_iou_matrix(np.ones((2, 4)), np.zeros((0, 4))).shape == (2, 0)

    def test_degenerate_box_does_not_divide_by_zero(self):
        a = np.array([[5, 5, 5, 5]], float)        # zero area
        b = np.array([[0, 0, 10, 10]], float)
        assert box_iou_matrix(a, b)[0, 0] == 0.0


class TestGreedyMatch:
    def test_each_ground_truth_claimed_once(self):
        # both predictions overlap the single GT; only one may be a TP
        aff = np.array([[0.9], [0.8]])
        flags = greedy_match(aff, 0.5)
        assert flags.tolist() == [True, False]

    def test_below_threshold_is_false_positive(self):
        assert greedy_match(np.array([[0.4]]), 0.5).tolist() == [False]

    def test_threshold_is_inclusive(self):
        assert greedy_match(np.array([[0.5]]), 0.5).tolist() == [True]

    def test_score_order_changes_which_prediction_wins(self):
        aff = np.array([[0.9], [0.95]])
        scores = np.array([0.1, 0.99])
        assert greedy_match(aff, 0.5, order="given").tolist() == [True, False]
        assert greedy_match(aff, 0.5, scores=scores, order="score").tolist() == [False, True]

    def test_score_order_requires_scores(self):
        with pytest.raises(ValueError, match="requires scores"):
            greedy_match(np.array([[0.9]]), 0.5, order="score")

    def test_no_ground_truth_means_all_false_positives(self):
        assert greedy_match(np.zeros((3, 0)), 0.5).tolist() == [False] * 3

    def test_no_predictions_returns_empty(self):
        assert greedy_match(np.zeros((0, 4)), 0.5).size == 0


class TestAveragePrecision:
    def test_perfect_ranking_scores_one(self):
        ap = average_precision([0.9, 0.8], [True, True], n_gt=2)
        assert ap == pytest.approx(1.0, abs=1e-3)

    def test_all_false_positives_score_zero(self):
        assert average_precision([0.9, 0.8], [False, False], n_gt=2) == 0.0

    def test_half_recall_caps_ap_near_half(self):
        # one of two GT found, perfectly ranked -> precision 1 up to recall 0.5
        ap = average_precision([0.9], [True], n_gt=2)
        assert 0.49 < ap < 0.52

    def test_ranking_matters(self):
        good = average_precision([0.9, 0.1], [True, False], n_gt=1)
        bad = average_precision([0.9, 0.1], [False, True], n_gt=1)
        assert good > bad

    def test_empty_predictions_score_zero(self):
        assert average_precision([], [], n_gt=5) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="disagree"):
            average_precision([0.9, 0.5], [True], n_gt=1)

    def test_recall_never_exceeds_one(self):
        curve = pr_curve([0.9, 0.8, 0.7], [True] * 3, n_gt=3)
        assert curve.recall.max() <= 1.0 + 1e-9
