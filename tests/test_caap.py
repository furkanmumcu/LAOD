"""CAAP: synthetic behaviour plus a regression pin on the original run."""

from __future__ import annotations

import numpy as np
import pytest

from laod.io.predictions import ImageGroundTruth, ImagePredictions
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP_LEGACY, CAAP_V2


def gt(image_id, *boxes, labels=None):
    boxes = np.array(boxes, float).reshape(-1, 4)
    return ImageGroundTruth(image_id, boxes, labels or ["obj"] * len(boxes))


def pred(image_id, *rows):
    """rows: (x1, y1, x2, y2, score)"""
    rows = np.array(rows, float).reshape(-1, 5)
    return ImagePredictions(image_id, rows[:, :4], rows[:, 4], ["obj"] * len(rows))


class TestCAAPBehaviour:
    def test_exact_box_match_scores_one(self):
        g = [gt(0, [0, 0, 10, 10])]
        p = [pred(0, [0, 0, 10, 10, 0.9])]
        res = CAAPEvaluator(g, [0.5]).evaluate(p, progress=False)
        assert res.per_threshold[0.5] == pytest.approx(1.0, abs=1e-3)

    def test_zero_overlap_scores_zero(self):
        g = [gt(0, [0, 0, 10, 10])]
        p = [pred(0, [100, 100, 110, 110, 0.9])]
        res = CAAPEvaluator(g, [0.5]).evaluate(p, progress=False)
        assert res.per_threshold[0.5] == 0.0

    def test_redundant_predictions_on_one_object_are_false_positives(self):
        g = [gt(0, [0, 0, 10, 10])]
        p = [pred(0, [0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8], [0, 0, 10, 10, 0.7])]
        res = CAAPEvaluator(g, [0.5]).evaluate(p, progress=False)
        # one TP then two FP: precision decays, so AP falls below a clean 1.0
        assert 0.3 < res.per_threshold[0.5] < 1.0

    def test_empty_predictions_score_zero_but_do_not_raise(self):
        g = [gt(0, [0, 0, 10, 10])]
        p = [ImagePredictions(0, np.zeros((0, 4)), np.zeros(0), [])]
        res = CAAPEvaluator(g, [0.5]).evaluate(p, progress=False)
        assert res.per_threshold[0.5] == 0.0
        assert res.n_gt == 1 and res.n_pred == 0

    def test_empty_ground_truth_raises(self):
        with pytest.raises(ValueError, match="ground truth is empty"):
            CAAPEvaluator([gt(0)], [0.5]).evaluate([pred(0, [0, 0, 1, 1, 0.5])],
                                                   progress=False)

    def test_labels_are_ignored(self):
        g = [ImageGroundTruth(0, np.array([[0, 0, 10, 10]], float), ["cat"])]
        same = ImagePredictions(0, np.array([[0, 0, 10, 10]], float),
                                np.array([0.9]), ["cat"])
        other = ImagePredictions(0, np.array([[0, 0, 10, 10]], float),
                                 np.array([0.9]), ["aubergine"])
        ev = CAAPEvaluator(g, [0.5])
        a = ev.evaluate([same], progress=False).per_threshold[0.5]
        b = ev.evaluate([other], progress=False).per_threshold[0.5]
        assert a == b, "CAAP must be class-agnostic"

    def test_predictions_do_not_match_across_images(self):
        # a box that would match perfectly, but in the wrong image
        g = [gt(0, [0, 0, 10, 10]), gt(1, [50, 50, 60, 60])]
        p = [pred(0, [50, 50, 60, 60, 0.9]),
             ImagePredictions(1, np.zeros((0, 4)), np.zeros(0), [])]
        res = CAAPEvaluator(g, [0.5]).evaluate(p, progress=False)
        assert res.per_threshold[0.5] == 0.0

    def test_higher_iou_threshold_is_never_easier(self):
        g = [gt(0, [0, 0, 10, 10])]
        p = [pred(0, [1, 1, 11, 11, 0.9])]
        res = CAAPEvaluator(g, [0.5, 0.95]).evaluate(p, progress=False)
        assert res.per_threshold[0.5] >= res.per_threshold[0.95]

    def test_dict_input_form_is_accepted(self):
        g = [gt(0, [0, 0, 10, 10])]
        p = [{"image_id": 0, "boxes": np.array([[0, 0, 10, 10]]),
              "scores": np.array([0.9]), "labels": ["obj"]}]
        res = CAAPEvaluator(g, [0.5]).evaluate(p, progress=False)
        assert res.per_threshold[0.5] == pytest.approx(1.0, abs=1e-3)

    def test_xywh_input_is_converted(self):
        g = [gt(0, [0, 0, 10, 10])]
        p = [{"image_id": 0, "boxes": np.array([[0, 0, 10, 10]]), "box_format": "xywh",
              "scores": np.array([0.9]), "labels": ["obj"]}]
        res = CAAPEvaluator(g, [0.5]).evaluate(p, progress=False)
        assert res.per_threshold[0.5] == pytest.approx(1.0, abs=1e-3)


class TestLegacyGrid:
    def test_arange_quirk_is_preserved(self):
        # the published intervals really do contain four thresholds each
        assert CAAP_LEGACY.lo == (0.50, 0.55, 0.60, 0.65)
        assert CAAP_LEGACY.hi == (0.85, 0.90, 0.95, 1.00)

    def test_v2_grid_is_disjoint_and_attainable(self):
        assert set(CAAP_V2.lo) & set(CAAP_V2.mi) == set()
        assert max(CAAP_V2.hi) < 1.00


@pytest.mark.regression
class TestReproduction:
    """Pins the published COCO-Val CAAP row to the archived predictions."""

    EXPECTED = {0.50: 0.2624, 0.55: 0.2538, 0.60: 0.2453, 0.65: 0.2412,
                0.70: 0.2263, 0.75: 0.2145, 0.80: 0.1953, 0.85: 0.1672,
                0.90: 0.1196, 0.95: 0.0448}

    def test_per_threshold_matches_original_implementation(self, legacy_run):
        gt_, preds = legacy_run
        res = CAAPEvaluator(gt_, sorted(self.EXPECTED)).evaluate(preds, progress=False)
        for thr, want in self.EXPECTED.items():
            assert res.per_threshold[thr] == pytest.approx(want, abs=1e-3), f"IoU {thr}"

    def test_published_table1_values(self, legacy_run):
        gt_, preds = legacy_run
        res = CAAPEvaluator(gt_, CAAP_LEGACY.all_thresholds).evaluate(preds, progress=False)
        s = CAAP_LEGACY.summarise(res.per_threshold)
        assert round(s["LO"], 2) == 0.25
        assert round(s["MI"], 2) == 0.22
        assert round(s["HI"], 2) == 0.08

    def test_unreachable_threshold_depresses_legacy_hi(self, legacy_run):
        """HI is ~25% lower than it should be purely from averaging in IoU=1.00."""
        gt_, preds = legacy_run
        res = CAAPEvaluator(gt_, CAAP_LEGACY.all_thresholds).evaluate(preds, progress=False)
        assert res.per_threshold[1.00] < 0.001
        legacy_hi = CAAP_LEGACY.summarise(res.per_threshold)["HI"]
        fixed_hi = CAAP_V2.summarise(res.per_threshold)["HI"]
        assert fixed_hi > legacy_hi * 1.2
