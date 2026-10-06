"""SNAP: bipartite matching, thresholds, calibration and the chance control.

Matching behaviour is tested against a synthetic embedding space with known
angles, so the assertions do not depend on a CLIP checkpoint. Reproduction of
published numbers is a separate, marked class.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from laod.io.predictions import ImageGroundTruth, ImagePredictions
from laod.metrics.grids import SNAP_LEGACY, SNAP_YEN
from laod.metrics.snap import SNAPEvaluator, calibrate_thresholds


def _on_unit_circle(x: float) -> list[float]:
    """Unit vector whose cosine against [1, 0] is exactly ``x``."""
    return [x, math.sqrt(1.0 - x * x)]


class FakeEmbedder:
    """Places each label on the unit circle so cosines are exactly known.

    Components are built from the target cosine rather than from an angle, so
    the similarity against ``dog`` is exact in floating point and threshold
    inclusivity can be asserted without a tolerance.
    """

    VECTORS = {
        "dog": _on_unit_circle(1.0),       # cos(dog, dog)    = 1.00
        "puppy": _on_unit_circle(0.9),     # cos(dog, puppy)  = 0.90
        "canine": _on_unit_circle(0.7),    # cos(dog, canine) = 0.70
        "hound": _on_unit_circle(0.5),     # cos(dog, hound)  = 0.50
        "car": _on_unit_circle(0.0),       # cos(dog, car)    = 0.00
        "cat": _on_unit_circle(0.0),
    }

    def encode_raw(self, labels):
        try:
            return np.array([self.VECTORS[l] for l in labels], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(f"FakeEmbedder has no vector for {exc.args[0]!r}") from None

    def encode(self, labels, batch_size=256):
        raw = self.encode_raw(labels)
        return raw / np.linalg.norm(raw, axis=-1, keepdims=True)


def gt(image_id, labels):
    boxes = np.tile(np.array([[0, 0, 10, 10]], float), (len(labels), 1))
    return ImageGroundTruth(image_id, boxes, list(labels))


def pred(image_id, labels, scores):
    boxes = np.tile(np.array([[0, 0, 10, 10]], float), (len(labels), 1))
    return ImagePredictions(image_id, boxes, np.array(scores, float), list(labels))


def snap_at(g, p, tau, **kw):
    ev = SNAPEvaluator(g, FakeEmbedder(), [tau], **kw)
    return ev.evaluate(p, control=False, progress=False).per_threshold[round(tau, 4)]


class TestThreshold:
    def test_similar_label_above_threshold_matches(self):
        assert snap_at([gt(0, ["dog"])], [pred(0, ["puppy"], [0.9])], 0.8) == \
            pytest.approx(1.0, abs=1e-2)

    def test_similar_label_below_threshold_does_not_match(self):
        assert snap_at([gt(0, ["dog"])], [pred(0, ["canine"], [0.9])], 0.8) == 0.0

    def test_threshold_is_inclusive(self):
        assert snap_at([gt(0, ["dog"])], [pred(0, ["hound"], [0.9])], 0.5) > 0.0

    def test_unrelated_label_never_matches(self):
        assert snap_at([gt(0, ["dog"])], [pred(0, ["car"], [0.9])], 0.1) == 0.0

    def test_raising_the_threshold_is_monotone(self):
        g, p = [gt(0, ["dog"])], [pred(0, ["canine"], [0.9])]
        scores = [snap_at(g, p, t) for t in (0.5, 0.7, 0.9)]
        assert scores[0] >= scores[1] >= scores[2]

    def test_localisation_is_ignored(self):
        """A box nowhere near the object still counts if the name is right."""
        g = [ImageGroundTruth(0, np.array([[0, 0, 10, 10]], float), ["dog"])]
        p = [ImagePredictions(0, np.array([[900, 900, 910, 910]], float),
                              np.array([0.9]), ["puppy"])]
        assert snap_at(g, p, 0.8) == pytest.approx(1.0, abs=1e-2)


class TestBipartiteAssignment:
    def test_one_ground_truth_absorbs_only_one_prediction(self):
        g = [gt(0, ["dog"])]
        p = [pred(0, ["puppy", "puppy", "puppy"], [0.9, 0.8, 0.7])]
        # 1 TP + 2 FP -> strictly below a clean 1.0
        assert 0.3 < snap_at(g, p, 0.8) < 1.0

    def test_two_ground_truths_absorb_two_predictions(self):
        g = [gt(0, ["dog", "dog"])]
        p = [pred(0, ["puppy", "puppy"], [0.9, 0.8])]
        assert snap_at(g, p, 0.8) == pytest.approx(1.0, abs=1e-2)

    def test_prediction_matches_best_available_ground_truth(self):
        g = [gt(0, ["car", "dog"])]
        p = [pred(0, ["puppy"], [0.9])]
        # must pick 'dog' (0.90) over 'car' (0.0), so it is a TP at tau=0.8
        assert snap_at(g, p, 0.8) > 0.0

    def test_no_cross_image_matching(self):
        g = [gt(0, ["car"]), gt(1, ["dog"])]
        p = [pred(0, ["puppy"], [0.9]),
             ImagePredictions(1, np.zeros((0, 4)), np.zeros(0), [])]
        assert snap_at(g, p, 0.8) == 0.0


class TestConfidenceOrdering:
    def test_score_order_gives_the_match_to_the_confident_prediction(self):
        """With one GT and two candidates, order decides which one wins."""
        g = [gt(0, ["dog"])]
        # listed worst-first; 'canine' (0.70) appears before 'puppy' (0.90)
        p = [pred(0, ["canine", "puppy"], [0.1, 0.99])]
        given = snap_at(g, p, 0.6, match_order="given")
        by_score = snap_at(g, p, 0.6, match_order="score")
        # ranking by confidence puts the TP at the top of the PR curve
        assert by_score > given

    def test_ranking_rewards_confident_true_positives(self):
        g = [gt(0, ["dog", "car"])]
        good = [pred(0, ["puppy", "cat"], [0.9, 0.1])]
        bad = [pred(0, ["puppy", "cat"], [0.1, 0.9])]
        assert snap_at(g, good, 0.8) >= snap_at(g, bad, 0.8)


class TestEdgeCases:
    def test_empty_predictions_score_zero(self):
        g = [gt(0, ["dog"])]
        p = [ImagePredictions(0, np.zeros((0, 4)), np.zeros(0), [])]
        assert snap_at(g, p, 0.8) == 0.0

    def test_image_with_no_ground_truth_yields_only_false_positives(self):
        g = [gt(0, ["dog"]), ImageGroundTruth(1, np.zeros((0, 4)), [])]
        p = [pred(0, ["puppy"], [0.9]), pred(1, ["puppy"], [0.95])]
        # the high-confidence FP outranks the TP and depresses AP below 1.0
        assert 0.0 < snap_at(g, p, 0.8) < 1.0

    def test_empty_ground_truth_raises(self):
        g = [ImageGroundTruth(0, np.zeros((0, 4)), [])]
        with pytest.raises(ValueError, match="SNAP is undefined"):
            snap_at(g, [pred(0, ["dog"], [0.5])], 0.8)

    def test_ragged_arrays_are_rejected(self):
        with pytest.raises(ValueError, match="ragged"):
            ImagePredictions(0, np.zeros((2, 4)), np.array([0.5]), ["a", "b"])


class TestChanceControl:
    def test_control_reports_a_baseline(self):
        g = [gt(i, ["dog", "car"]) for i in range(20)]
        p = [pred(i, ["puppy", "cat"], [0.9, 0.8]) for i in range(20)]
        res = SNAPEvaluator(g, FakeEmbedder(), [0.8]).evaluate(p, control=True,
                                                               progress=False)
        assert 0.8 in res.chance
        assert res.gain(0.8) is not None

    def test_gain_is_none_without_control(self):
        g = [gt(0, ["dog"])]
        p = [pred(0, ["puppy"], [0.9])]
        res = SNAPEvaluator(g, FakeEmbedder(), [0.8]).evaluate(p, control=False,
                                                               progress=False)
        assert res.gain(0.8) is None

    def test_a_permissive_threshold_has_no_gain_over_chance(self):
        """The defect this control exists to catch: labels stop mattering."""
        g = [gt(i, ["dog", "car"]) for i in range(10)]
        p = [pred(i, ["puppy", "cat"], [0.9, 0.8]) for i in range(10)]
        # every pairwise cosine here is >= 0.0, so tau=-1 admits everything
        res = SNAPEvaluator(g, FakeEmbedder(), [-1.0]).evaluate(p, control=True,
                                                                progress=False)
        assert res.gain(-1.0) == pytest.approx(0.0, abs=1e-9)


class TestCalibration:
    def test_thresholds_track_the_requested_false_match_rate(self):
        vocab = ["dog", "car", "hound", "canine"]
        out = calibrate_thresholds(FakeEmbedder(), vocab,
                                   {"loose": 0.5, "strict": 0.01}, center=False)
        assert out["strict"] > out["loose"], "a stricter FMR needs a higher cosine"

    def test_calibration_needs_at_least_two_labels(self):
        with pytest.raises(ValueError, match="at least two"):
            calibrate_thresholds(FakeEmbedder(), ["dog"], {"x": 0.1})


class TestGrids:
    def test_legacy_snap_grid_inherits_the_arange_quirk(self):
        assert SNAP_LEGACY.lo == (0.50, 0.55, 0.60, 0.65)
        assert 1.00 in SNAP_LEGACY.hi

    def test_yen_macro_is_the_four_fixed_thresholds(self):
        assert SNAP_YEN.macro == (0.60, 0.70, 0.80, 0.90)

    def test_unknown_grid_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown SNAP grid"):
            SNAPEvaluator([gt(0, ["dog"])], FakeEmbedder(), grid="nonsense")
