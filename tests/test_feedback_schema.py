"""Round-2 revision parsing: recovery, no-op discipline, condition enforcement."""

from __future__ import annotations

import json

import pytest

from laod.feedback.schema import (Revision, normalise, normalise_all,
                                  parse_revision, recover_vocabulary)
from laod.feedback.summary import concept_support, feedback_block

C0 = ["person", "bicycle", "zebra"]


def reply(**kw) -> str:
    base = {"keep": [], "remove": [], "rename": {}, "add": [],
            "revised_vocabulary": []}
    base.update(kw)
    return json.dumps(base)


# --------------------------------------------------------------- parsing ---
def test_clean_json_is_parsed():
    r = parse_revision(reply(keep=["person", "bicycle"], remove=["zebra"],
                             revised_vocabulary=["person", "bicycle"]),
                       C0, "feedback_full")
    assert r.parse_ok and r.fallback_level == 0
    assert r.c1 == ["person", "bicycle"]
    assert r.remove == ["zebra"]


def test_code_fence_and_surrounding_prose_are_tolerated():
    raw = ('Sure, here is my revision:\n```json\n'
           + reply(revised_vocabulary=["person"]) + '\n```\nHope that helps.')
    r = parse_revision(raw, C0, "feedback_full")
    assert r.parse_ok and r.c1 == ["person"]


def test_malformed_object_recovers_vocabulary_only():
    raw = '{"keep": [person, bicycle], "revised_vocabulary": ["person", "car"]}'
    r = parse_revision(raw, C0, "feedback_full")
    assert r.fallback_level == 1
    assert r.c1 == ["person", "car"]
    assert any("recovered" in v for v in r.violations)


def test_unparseable_reply_is_a_logged_noop_not_a_prose_parse():
    """The reply must never be scraped for words.

    A generic label parser over this string would emit 'i think there is a
    person and maybe a dog' or its fragments as concepts and hand them to the
    detector. The correct outcome is C1 = C0 with an explicit failure.
    """
    raw = "I think there is a person and maybe a dog, but I am not certain."
    r = parse_revision(raw, C0, "feedback_full")
    assert not r.parse_ok
    assert r.fallback_level == 2
    assert r.c1 == normalise_all(C0)
    assert r.is_noop
    assert "dog" not in r.c1


def test_recover_vocabulary_returns_none_without_the_key():
    assert recover_vocabulary('{"keep": ["person"]}') is None
    assert recover_vocabulary("no json at all") is None
    assert recover_vocabulary('"revised_vocabulary": ["a", "b"]') == ["a", "b"]


def test_empty_revised_vocabulary_is_a_noop():
    r = parse_revision(reply(revised_vocabulary=[]), C0, "feedback_full")
    assert not r.parse_ok and r.is_noop


# ------------------------------------------- refinement-only enforcement ---
def test_refine_rejects_additions_hidden_in_the_vocabulary():
    """The add field is empty but a new concept appears in the vocabulary.

    Checking `add` alone would let the no-additions condition leak exactly the
    mechanism it exists to exclude.
    """
    r = parse_revision(reply(keep=["person"], add=[],
                             revised_vocabulary=["person", "traffic light"]),
                       C0, "feedback_refine")
    assert r.c1 == ["person"]
    assert r.rejected == ["traffic light"]
    assert r.violations


def test_refine_allows_a_rename_target_from_c0():
    r = parse_revision(reply(rename={"zebra": "horse"}, keep=["person"],
                             revised_vocabulary=["person", "horse"]),
                       C0, "feedback_refine")
    assert r.c1 == ["person", "horse"]
    assert not r.rejected


def test_refine_rejects_a_rename_target_whose_source_is_not_in_c0():
    r = parse_revision(reply(rename={"giraffe": "horse"},
                             revised_vocabulary=["person", "horse"]),
                       C0, "feedback_refine")
    assert r.c1 == ["person"]
    assert r.rejected == ["horse"]


def test_refine_clears_a_non_empty_add_field():
    r = parse_revision(reply(add=["dog"], revised_vocabulary=["person"]),
                       C0, "feedback_refine")
    assert r.add == []
    assert any("add field" in v for v in r.violations)


def test_full_and_selfreflect_allow_additions():
    for cond in ("feedback_full", "selfreflect"):
        r = parse_revision(reply(add=["traffic light"],
                                 revised_vocabulary=["person", "traffic light"]),
                           C0, cond)
        assert r.c1 == ["person", "traffic light"]
        assert not r.rejected


# ------------------------------------------------------------ safeguards ---
def test_max_vocab_truncates_and_records():
    r = parse_revision(reply(revised_vocabulary=[f"c{i}" for i in range(50)]),
                       C0, "feedback_full", max_vocab=10)
    assert len(r.c1) == 10
    assert len(r.rejected) == 40
    assert any("truncated" in v for v in r.violations)


def test_normalisation_matches_round_one():
    assert normalise("  Traffic   Light. ") == "traffic light"
    assert normalise_all(["Dog", "dog", " DOG "]) == ["dog"]


# -------------------------------------------------------------- feedback ---
def test_feedback_reports_zero_detection_concepts():
    sup = concept_support(["person", "zebra"],
                          ["person", "person", "person"], [0.84, 0.3, 0.1])
    block = feedback_block(sup)
    assert "person: 3 detections, max confidence 0.84" in block
    assert "zebra: 0 detections" in block
    assert "mean" not in block           # mean confidence is deliberately absent


def test_feedback_singular_plural_and_case_insensitive_match():
    sup = concept_support(["Traffic Light"], ["traffic light"], [0.5])
    assert sup[0].n_detections == 1
    assert "1 detection," in sup[0].line()
