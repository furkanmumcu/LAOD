"""Label parsing, caching and run persistence.

These cover the layer that must not lose information: if a run's stored output
is incomplete, an LLM pass measured in hours has to be repeated.
"""

from __future__ import annotations

import json

import pytest

from laod.io.label_cache import LabelCache
from laod.io.run_store import ImageRecord, RunConfig, RunStore, load_run
from laod.models.label_parser import parse_labels
from laod.models.registry import DETECTORS, LLMS, PROMPTS


class TestLabelParser:
    def test_strict_mode_normalises(self):
        p = parse_labels("Person,  Chair , TV", "strict")
        assert p.labels == ["person", "chair", "tv"]

    def test_legacy_mode_preserves_the_original_defect(self):
        """The original never stripped or cased; reproduction needs that back."""
        p = parse_labels("Person, Chair", "legacy")
        assert p.labels == ["Person", " Chair"]

    def test_duplicates_are_dropped_and_counted(self):
        p = parse_labels("book, book, pen", "strict")
        assert p.labels == ["book", "pen"]
        assert p.flags["duplicate"] == 1

    def test_empty_segments_are_counted_not_emitted(self):
        p = parse_labels("a, , b", "strict")
        assert p.labels == ["a", "b"]
        assert p.flags["empty"] == 1

    def test_long_phrases_are_kept_but_flagged(self):
        """Dropping them would hide the behaviour worth measuring."""
        p = parse_labels("cat, a large wooden dining table by the window", "strict")
        assert len(p.labels) == 2
        assert p.flags["long_phrase"] == 1

    def test_bullet_markers_are_stripped(self):
        p = parse_labels("- cat\n- dog\n3. bird", "strict")
        assert p.labels == ["cat", "dog", "bird"]

    def test_preamble_line_is_discarded(self):
        p = parse_labels("Here are the objects:\ncat, dog", "strict")
        assert p.labels == ["cat", "dog"]

    def test_raw_is_always_retained(self):
        raw = "Person, Chair"
        assert parse_labels(raw, "strict").raw == raw

    def test_empty_response_yields_no_labels(self):
        assert parse_labels("", "strict").labels == []

    def test_max_labels_truncates_and_records(self):
        p = parse_labels("a,b,c,d", "strict", max_labels=2)
        assert p.labels == ["a", "b"] and p.flags["over_limit"] == 2

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown parse mode"):
            parse_labels("a", "nonsense")

    def test_json_round_trip(self):
        p = parse_labels("Cat, dog, dog", "strict")
        from laod.models.label_parser import ParsedLabels
        assert ParsedLabels.from_json(p.to_json()).labels == p.labels


class TestLabelCache:
    def test_put_then_get(self, tmp_path):
        c = LabelCache(tmp_path, "coco", "llm", "original")
        c.put(7, parse_labels("cat, dog", "strict"))
        assert c.get(7).labels == ["cat", "dog"]
        assert 7 in c

    def test_survives_reopen(self, tmp_path):
        LabelCache(tmp_path, "coco", "llm", "p").put(7, parse_labels("cat", "strict"))
        assert LabelCache(tmp_path, "coco", "llm", "p").get(7).labels == ["cat"]

    def test_missing_drives_resume(self, tmp_path):
        c = LabelCache(tmp_path, "coco", "llm", "p")
        c.put(1, parse_labels("cat", "strict"))
        assert c.missing([1, 2, 3]) == [2, 3]

    def test_truncated_final_line_is_tolerated(self, tmp_path):
        """An interrupted run leaves a half-written line; it must not be fatal."""
        c = LabelCache(tmp_path, "coco", "llm", "p")
        c.put(1, parse_labels("cat", "strict"))
        with c.path.open("a") as fh:
            fh.write('{"image_id": 2, "lab')
        assert LabelCache(tmp_path, "coco", "llm", "p").get(1).labels == ["cat"]

    def test_caches_are_keyed_separately(self, tmp_path):
        LabelCache(tmp_path, "coco", "a", "p").put(1, parse_labels("cat", "strict"))
        assert len(LabelCache(tmp_path, "coco", "b", "p")) == 0

    def test_vocabulary_counts(self, tmp_path):
        c = LabelCache(tmp_path, "coco", "llm", "p")
        c.put(1, parse_labels("cat, dog", "strict"))
        c.put(2, parse_labels("cat", "strict"))
        assert c.vocabulary()["cat"] == 2


def _record(image_id=1):
    return ImageRecord(
        image_id=image_id, file_name=f"{image_id:012d}.jpg", width=640, height=480,
        raw_response="Person, Chair", labels=["person", "chair"],
        boxes=[[1, 2, 3, 4]], scores=[0.9], pred_labels=["person"],
        gt_labels=["person", "chair"])


class TestRunStore:
    def test_config_slug_identifies_the_cell(self):
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="original")
        assert cfg.slug == "coco__g__y__original"

    def test_split_size_appears_in_slug(self):
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="p", split_size=200)
        assert cfg.slug.endswith("__n200")

    def test_environment_is_captured_automatically(self):
        assert "python" in RunConfig(dataset="c", llm="l", detector="d", prompt="p").env

    def test_round_trip_preserves_everything_needed_later(self, tmp_path):
        """The point of the store: re-scoring must never need re-inference."""
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="original")
        store = RunStore(tmp_path, cfg)
        store.append(_record())
        _, records = load_run(store.dir)
        r = records[0]
        assert r.raw_response == "Person, Chair"     # audit the parse
        assert r.gt_labels == ["person", "chair"]    # re-score text with any encoder
        assert r.pred_labels == ["person"]
        assert (r.width, r.height) == (640, 480)     # recompute areas / renormalise

    def test_resume_skips_written_images(self, tmp_path):
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="p")
        RunStore(tmp_path, cfg).append(_record(1))
        reopened = RunStore(tmp_path, cfg, resume=True)
        assert 1 in reopened and reopened.missing([1, 2]) == [2]

    def test_no_resume_discards_previous(self, tmp_path):
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="p")
        RunStore(tmp_path, cfg).append(_record(1))
        assert 1 not in RunStore(tmp_path, cfg, resume=False)

    def test_summary_counts(self, tmp_path):
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="p")
        s = RunStore(tmp_path, cfg)
        s.append(_record(1)); s.append(_record(2))
        assert s.summary() == {"images": 2, "detections": 2, "gt_objects": 4,
                               "unique_predicted_labels": 1}

    def test_adapts_to_metric_layer(self, tmp_path):
        from laod.io.run_store import run_to_predictions
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="p")
        store = RunStore(tmp_path, cfg)
        store.append(_record())
        p = run_to_predictions(store.dir)[0]
        assert p.image_id == 1 and p.labels == ["person"] and p.boxes.shape == (1, 4)

    def test_config_is_written_as_readable_json(self, tmp_path):
        cfg = RunConfig(dataset="coco", llm="g", detector="y", prompt="p")
        store = RunStore(tmp_path, cfg)
        assert json.loads(store.config_path.read_text())["llm"] == "g"


class TestRegistry:
    def test_matrix_size(self):
        from laod.models.registry import ACTIVE_LLMS
        assert len(ACTIVE_LLMS) == 7 and len(DETECTORS) == 4

    def test_unsupported_models_are_declared_but_excluded(self):
        """An exclusion must stay discoverable, not vanish from the registry."""
        from laod.models.registry import ACTIVE_LLMS
        excluded = {k for k, v in LLMS.items() if not v.supported}
        assert excluded == {"phi35-vision"}
        assert excluded.isdisjoint(ACTIVE_LLMS)
        assert all(LLMS[k].notes for k in excluded), "an exclusion needs a reason"

    def test_building_an_unsupported_model_fails_loudly(self):
        from laod.models.llm_agent import build_llm_agent
        with pytest.raises(RuntimeError, match="unsupported"):
            build_llm_agent("phi35-vision")

    def test_reasoning_model_has_thinking_disabled(self):
        """Left in thinking mode it emits chain-of-thought instead of labels."""
        assert LLMS["qwen35-9b"].pipeline_kwargs.get("enable_thinking") is False

    def test_only_gemma3_is_gated(self):
        assert [k for k, s in LLMS.items() if s.gated] == ["gemma3-4b"]

    def test_phi_record_is_retained_with_its_adapter(self):
        assert LLMS["phi35-vision"].adapter == "phi3_vision"
        assert LLMS["phi35-vision"].trust_remote_code

    def test_default_prompt_is_what_was_actually_run(self):
        from laod.models.registry import DEFAULT_PROMPT
        assert "Do not list sky, street." in PROMPTS[DEFAULT_PROMPT].user

    def test_minimal_prompt_differs_from_the_default(self):
        """The paper's text describes the minimal prompt; the code ran the
        default. They differ, which is why both are kept."""
        assert PROMPTS["minimal"].user != PROMPTS["default"].user

    def test_pre_rename_prompt_names_still_resolve(self):
        """Runs and caches written before the rename must keep working."""
        from laod.models.registry import resolve_prompt
        assert resolve_prompt("original") == "default"
        assert resolve_prompt("paper") == "minimal"
        assert resolve_prompt("granular") == "coco-specific"
        for k in PROMPTS:
            assert resolve_prompt(k) == k, "canonical keys pass through"
        with pytest.raises(ValueError, match="unknown prompt"):
            resolve_prompt("no-such-prompt")

    def test_every_key_is_self_consistent(self):
        assert all(k == s.key for k, s in LLMS.items())
        assert all(k == s.key for k, s in DETECTORS.items())


class TestUAP:
    """Unknown-object metrics for COCO-OOD (Table 2 of the original paper)."""

    @staticmethod
    def _case():
        import numpy as np
        from laod.io.predictions import ImageGroundTruth, ImagePredictions
        gt = [ImageGroundTruth(0, np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float),
                               ["unknow object"] * 2)]
        preds = [ImagePredictions(0, np.array([[0, 0, 10, 10], [90, 90, 99, 99]], float),
                                  np.array([0.9, 0.8]), ["thing", "thing"])]
        return gt, preds

    def test_precision_recall_f1(self):
        from laod.metrics.uap import UAPEvaluator
        gt, preds = self._case()
        r = UAPEvaluator(gt, score_threshold=0.5).evaluate(preds)
        assert r.u_precision == pytest.approx(0.5)   # 1 TP of 2 kept detections
        assert r.u_recall == pytest.approx(0.5)      # 1 of 2 unknown objects
        assert r.u_f1 == pytest.approx(0.5)

    def test_score_threshold_filters_the_operating_point(self):
        from laod.metrics.uap import UAPEvaluator
        gt, preds = self._case()
        r = UAPEvaluator(gt, score_threshold=0.85).evaluate(preds)
        assert r.n_considered == 1 and r.u_precision == pytest.approx(1.0)

    def test_u_ap_ignores_the_score_threshold(self):
        """AP integrates the whole ranking, so the operating point must not move it."""
        from laod.metrics.uap import UAPEvaluator
        gt, preds = self._case()
        a = UAPEvaluator(gt, score_threshold=0.1).evaluate(preds).u_ap
        b = UAPEvaluator(gt, score_threshold=0.9).evaluate(preds).u_ap
        assert a == pytest.approx(b)

    def test_empty_ground_truth_raises(self):
        import numpy as np
        from laod.io.predictions import ImageGroundTruth
        from laod.metrics.uap import UAPEvaluator
        gt, preds = self._case()
        empty = [ImageGroundTruth(0, np.zeros((0, 4)), [])]
        with pytest.raises(ValueError, match="U-AP is undefined"):
            UAPEvaluator(empty).evaluate(preds)


class TestGenerationRouting:
    """Generation parameters must reach generate(), not the processor.

    Passed as bare **kwargs the image-text-to-text pipeline routes them to the
    processor, which ignores them -- so do_sample=False never takes effect and
    each model silently runs with its own config. Gemma-3 defaults to
    do_sample=True/top_k=64/top_p=0.95, which made 'greedy' runs sample and
    produced a 61% label-overlap rate between identical reruns.
    """

    class _StubPipe:
        def __init__(self):
            self.seen = None

        def __call__(self, text=None, **kwargs):
            self.seen = kwargs
            return [{"generated_text": [{"content": "cat, dog"}]}]

    def _agent(self):
        from laod.models.llm_agent import PipelineAgent
        from laod.models.registry import LLMS
        agent = PipelineAgent.__new__(PipelineAgent)
        agent.spec = LLMS["gemma3-4b"]
        agent.generation = {"max_new_tokens": 256, "do_sample": False}
        agent.pipe = self._StubPipe()
        return agent

    def test_params_go_through_generate_kwargs(self):
        from laod.models.registry import PROMPTS
        agent = self._agent()
        agent.generate(object(), PROMPTS["default"])
        assert "generate_kwargs" in agent.pipe.seen, \
            "generation params must be nested under generate_kwargs"
        assert agent.pipe.seen["generate_kwargs"]["do_sample"] is False

    def test_params_are_not_passed_bare(self):
        from laod.models.registry import PROMPTS
        agent = self._agent()
        agent.generate(object(), PROMPTS["default"])
        assert "do_sample" not in agent.pipe.seen
        assert "max_new_tokens" not in agent.pipe.seen

    def test_greedy_is_the_default(self):
        from laod.models.llm_agent import BaseLLMAgent
        import inspect
        assert inspect.signature(BaseLLMAgent.__init__).parameters["do_sample"].default is False


class TestDetectorPresets:
    """Confidence thresholds are measured, not inherited.

    Thresholding before computing AP truncates the ranked PR curve, so every
    backend's shipped default understates it. The v2 values come from sweeping
    each detector on 100 COCO images against a fixed label set.
    """

    def test_legacy_preserves_the_original_thresholds(self):
        from laod.models.registry import detector_params
        assert detector_params("yolo-world", "legacy") == {"conf": 0.25}
        assert detector_params("gdino-tiny", "legacy")["score_threshold"] == 0.4

    def test_v2_overrides_only_what_was_measured(self):
        from laod.models.registry import detector_params
        v2 = detector_params("gdino-tiny", "v2")
        assert v2["score_threshold"] == 0.03
        assert v2["text_threshold"] == 0.3, "unmeasured params carry over"

    def test_every_detector_sits_at_its_grid_floor(self):
        """Under maxDets the cap thresholds, not the confidence cut.

        Every (dataset, LLM) fit selects the lowest value its detector was
        swept at, because for a crowded image the top 100 by score are the
        same whether the cut is 0.01 or 0.05. OWLv2 previously sat at 0.15,
        an apparent interior optimum that turned out to be an artifact of
        visiting predictions in detector order rather than by score.
        """
        from laod.models.registry import detector_params
        floors = {"yolo-world": ("conf", 0.001),
                  "gdino-tiny": ("score_threshold", 0.03),
                  "gdino-base": ("score_threshold", 0.03),
                  "owlv2-base": ("score_threshold", 0.01)}
        for key, (param, want) in floors.items():
            assert detector_params(key, "v2")[param] == want, key

    def test_unknown_preset_is_rejected(self):
        from laod.models.registry import detector_params
        with pytest.raises(ValueError, match="unknown detector preset"):
            detector_params("yolo-world", "nonsense")

    def test_every_detector_declares_both_presets(self):
        from laod.models.registry import DETECTORS, detector_params
        for key in DETECTORS:
            assert detector_params(key, "legacy")
            assert detector_params(key, "v2")


class TestHoldoutSplit:
    """The hyperparameter-selection holdout must actually be held out.

    Detector thresholds are fitted on 500 COCO images. The sweep showed
    threshold choice moves CAAP by ~78% (median) -- more than the detector
    choice does -- so fitting it on images that are then reported on would be
    selecting a high-leverage parameter on the test set.
    """

    def test_split_is_frozen_to_an_explicit_id_list(self):
        import json
        from laod.data.splits import TUNING_SPLIT_PATH
        data = json.loads(TUNING_SPLIT_PATH.read_text())
        assert len(data["image_ids"]) == 500
        assert data["seed"] == 20261001
        assert len(set(data["image_ids"])) == 500, "no duplicates"

    def test_split_cannot_be_silently_redrawn(self):
        from laod.data.splits import build_tuning_split
        with pytest.raises(FileExistsError, match="frozen on purpose"):
            build_tuning_split(range(5000))

    def test_tune_and_test_are_disjoint_and_exhaustive(self):
        from laod.data.annotations import load_ground_truth
        from laod.data.splits import apply_split
        from laod.config import PATHS
        if not PATHS.coco_ann.is_file():
            pytest.skip("annotations not provisioned")
        gt = load_ground_truth(PATHS.coco_ann, "coco")
        tune = {g.image_id for g in apply_split(gt, "tune")}
        test = {g.image_id for g in apply_split(gt, "test")}
        assert tune & test == set(), "an image cannot be in both"
        assert len(tune | test) == len(gt) == 5000
        assert len(tune) == 500 and len(test) == 4500

    def test_holdout_is_excluded_from_lvis_and_coco_ood_too(self):
        """Both are strict subsets of val2017, so a COCO-only holdout leaks."""
        from laod.data.annotations import load_ground_truth
        from laod.data.splits import apply_split, load_tuning_split
        from laod.config import PATHS
        if not PATHS.lvis_ann.is_file():
            pytest.skip("annotations not provisioned")
        tune = load_tuning_split()
        for ann, name, expected in [(PATHS.lvis_ann, "lvis", 4327),
                                    (PATHS.coco_ood_ann, "coco_ood", 438)]:
            gt = load_ground_truth(ann, name)
            test = apply_split(gt, "test")
            assert len(test) == expected
            assert not ({g.image_id for g in test} & tune)
