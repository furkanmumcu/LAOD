"""Round-2 prompts for the three iterative conditions.

All three hand back the original image: it is the primary evidence, and the
detector summary is secondary. None of them mentions coordinates, box geometry
or overlap -- the division of labour under test is that the MLLM supplies
semantics and the detector supplies localisation, and a prompt that asked about
boxes would quietly dissolve it.

The grounding conditions tell the model explicitly that a confidence figure is
a relative signal within one image rather than a probability, because the two
detectors here run at operating points two orders of magnitude apart
(YOLO-World 0.001, OWLv2 0.01) and a number read as a probability would be
read wrongly.
"""

from __future__ import annotations

from laod.models.registry import PromptSpec

def _fmt(add_hint: str) -> str:
    """The output-format block.

    Braces are doubled because the result is passed through ``str.format``
    once more, in ``build_prompt``, to substitute the vocabulary and feedback.
    Built by concatenation rather than an f-string so the doubling survives.
    """
    return ('Reply with one JSON object and nothing else:\n'
            '{{"keep": [...], "remove": [...], "rename": {{"old": "new"}}, '
            '"add": [' + add_hint + '], "revised_vocabulary": [...]}}\n'
            '"revised_vocabulary" must list the complete final vocabulary.')

_GROUNDED_SYSTEM = (
    "You revise an object vocabulary for an image using both the image and "
    "feedback from an object grounding model. The grounding feedback is "
    "evidence, not ground truth. A concept with no detections may be "
    "incorrect, but the grounding model may also have missed a real object. "
    "Confidence values are a relative signal within this one image, not a "
    "probability that the object exists. Use the image as the primary "
    "evidence. Reply with a single JSON object and nothing else.")

_PLAIN_SYSTEM = (
    "You review an object vocabulary you previously proposed for an image, "
    "using the image itself as the evidence. Reply with a single JSON object "
    "and nothing else.")

_SELFREFLECT_USER = (
    "You previously proposed the following object concepts for this image:\n"
    "{vocabulary}\n\n"
    "Re-examine the image and revise the vocabulary. You may retain correct "
    "concepts, remove concepts that are not visibly present, rename inaccurate "
    "concepts, and add important visible objects that were missing from the "
    "original list.\n\n" + _fmt("..."))

_REFINE_USER = (
    "You previously proposed the following object concepts for this image:\n"
    "{vocabulary}\n\n"
    "The grounding detector reported:\n{feedback}\n\n"
    "Re-examine the image and revise the vocabulary. You may retain correct "
    "concepts, remove concepts that are not visibly present, and rename "
    "inaccurate concepts. Do not add any new concepts: every entry in the "
    "revised vocabulary must be one of the original concepts or a rename of "
    "one. Do not assume that zero detections means a concept is absent -- the "
    "detector may have failed to find a real object.\n\n"
    + _fmt(""))

_FULL_USER = (
    "You previously proposed the following object concepts for this image:\n"
    "{vocabulary}\n\n"
    "The grounding detector reported:\n{feedback}\n\n"
    "Re-examine the image and revise the vocabulary. You may retain correct "
    "concepts, remove concepts that are not visibly present, rename inaccurate "
    "concepts, and add important visible objects that were missing from the "
    "original list. Do not assume that zero detections means a concept is "
    "absent -- the detector may have failed to find a real object.\n\n"
    + _fmt("..."))

#: The three Round-2 conditions. Condition A (baseline) makes no second call
#: and therefore has no prompt: it is the stored Round-1 run, unmodified.
ROUND2_PROMPTS: dict[str, PromptSpec] = {
    "selfreflect": PromptSpec(
        "selfreflect", _PLAIN_SYSTEM, _SELFREFLECT_USER,
        provenance="Condition B -- controls for the effect of simply looking "
                   "at the image a second time. No detector feedback."),
    "feedback_refine": PromptSpec(
        "feedback_refine", _GROUNDED_SYSTEM, _REFINE_USER,
        provenance="Condition C -- detector feedback, removal and renaming "
                   "only. Isolates precision repair from recall recovery."),
    "feedback_full": PromptSpec(
        "feedback_full", _GROUNDED_SYSTEM, _FULL_USER,
        provenance="Condition D -- detector feedback with additions allowed. "
                   "The full iterative hypothesis."),
}

NEEDS_FEEDBACK = frozenset({"feedback_refine", "feedback_full"})


def build_prompt(condition: str, vocabulary: list[str], feedback: str) -> PromptSpec:
    """The filled prompt for one image under one condition."""
    spec = ROUND2_PROMPTS[condition]
    vocab_text = ", ".join(vocabulary)
    user = (spec.user.format(vocabulary=vocab_text, feedback=feedback)
            if condition in NEEDS_FEEDBACK
            else spec.user.format(vocabulary=vocab_text))
    return PromptSpec(spec.key, spec.system, user, spec.provenance)
