"""The experiment matrix as data: LLMs, detectors and prompts.

Every model and prompt used anywhere in v2 is declared here. Nothing constructs
a model id or prompt string inline, so a run's configuration is always a set of
keys that can be written into an output file and read back later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Adapter = Literal["pipeline", "phi3_vision"]
DetectorKind = Literal["yolo_world", "grounding_dino", "owlv2"]


@dataclass(frozen=True, slots=True)
class LLMSpec:
    """A vision-language model used as the label generator."""

    key: str
    model_id: str
    params_b: float
    family: str
    role: str
    adapter: Adapter = "pipeline"
    gated: bool = False
    trust_remote_code: bool = False
    cached: bool = False
    supported: bool = True
    pipeline_kwargs: dict = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """An open-vocabulary detector used to localise generated labels.

    ``defaults`` / ``v2_defaults`` hold *confidence* thresholds -- the score a
    detection must clear to be kept. They are not IoU thresholds (those are
    CAAP's evaluation grid) and are not comparable between backends.
    """

    key: str
    model_id: str
    kind: DetectorKind
    params_b: float
    text_encoder: str
    defaults: dict[str, float] = field(default_factory=dict)
    v2_defaults: dict[str, float] = field(default_factory=dict)
    local_weight: str | None = None
    cached: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """A system/user instruction pair put to the label generator."""

    key: str
    system: str
    user: str
    provenance: str


# --------------------------------------------------------------------------
# LLMs -- see REPRODUCIBILITY.md section 6 for why each is here
# --------------------------------------------------------------------------

LLMS: dict[str, LLMSpec] = {
    s.key: s for s in (
        LLMSpec("gemma3-4b", "google/gemma-3-4b-it", 4.3, "gemma-3",
                role="replication anchor -- produced the original Table 1",
                gated=True,
                notes="The only gated asset in the matrix; needs HF_TOKEN and "
                      "accepted licence terms."),
        LLMSpec("gemma4-e2b", "google/gemma-4-E2B-it", 5.1, "gemma-4",
                role="capacity ladder, rung 1"),
        LLMSpec("gemma4-e4b", "google/gemma-4-E4B-it", 8.0, "gemma-4",
                role="capacity ladder, rung 2"),
        LLMSpec("gemma4-12b", "google/gemma-4-12B-it", 12.0, "gemma-4",
                role="capacity ladder, rung 3",
                notes="Gemma4UnifiedForConditionalGeneration -- a different "
                      "architecture class from its siblings."),
        LLMSpec("qwen25-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct", 7.6, "qwen-2.5",
                role="continuity with the downstream analysis", cached=True),
        LLMSpec("qwen35-9b", "Qwen/Qwen3.5-9B", 9.0, "qwen-3.5",
                role="newer generation, vision-capable", cached=True,
                pipeline_kwargs={"enable_thinking": False},
                notes="A reasoning model: left in thinking mode it returns its "
                      "chain of thought instead of a label list, and the parser "
                      "faithfully turns that into nonsense labels. The original "
                      "codebase set enable_thinking=False for the same reason."),
        LLMSpec("internvl3-8b", "OpenGVLab/InternVL3-8B-hf", 7.9, "internvl-3",
                role="third family, size-matched to Qwen2.5-VL-7B",
                notes="Replaces Phi-3.5-vision, which cannot run on transformers "
                      "5.x (see phi35-vision). Native transformers support, no "
                      "remote code. At 7.9B it pairs with Qwen2.5-VL (7.6B) to "
                      "isolate model family at near-constant capacity -- the same "
                      "control Phi was chosen for, at a different capacity point."),
        LLMSpec("phi35-vision", "microsoft/Phi-3.5-vision-instruct", 4.1, "phi-3.5",
                role="INCOMPATIBLE -- retained for the record, excluded from runs",
                adapter="phi3_vision", trust_remote_code=True, supported=False,
                notes="Ships modeling code written against transformers ~4.4x. "
                      "On 5.12 generation fails: seen_tokens, get_max_length and "
                      "get_usable_length are all gone, and past those the cache "
                      "layout itself changed (DynamicCache is now per-layer), so "
                      "its attention path cannot be shimmed without "
                      "reimplementing it. Superseded by internvl3-8b."),
    )
}

# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

DETECTORS: dict[str, DetectorSpec] = {
    s.key: s for s in (
        DetectorSpec("yolo-world", "yolov8x-worldv2.pt", "yolo_world", 0.0, "CLIP",
                     defaults={"conf": 0.25}, v2_defaults={"conf": 0.001},
                     local_weight="yolov8x-worldv2.pt", cached=True,
                     notes="The detector behind every published number."),
        DetectorSpec("gdino-tiny", "IDEA-Research/grounding-dino-tiny",
                     "grounding_dino", 0.17, "BERT phrase-grounding",
                     defaults={"score_threshold": 0.4, "text_threshold": 0.3},
                     v2_defaults={"score_threshold": 0.03, "text_threshold": 0.3},
                     cached=True,
                     notes="Thresholds match the original demo script."),
        DetectorSpec("gdino-base", "IDEA-Research/grounding-dino-base",
                     "grounding_dino", 0.23, "BERT phrase-grounding",
                     defaults={"score_threshold": 0.4, "text_threshold": 0.3},
                     v2_defaults={"score_threshold": 0.03, "text_threshold": 0.3},
                     notes="The more common benchmark default; run alongside "
                           "tiny so the cross-paper comparison is unambiguous."),
        DetectorSpec("owlv2-base", "google/owlv2-base-patch16-ensemble",
                     "owlv2", 0.15, "CLIP",
                     defaults={"score_threshold": 0.1},
                     v2_defaults={"score_threshold": 0.01},
                     notes="CLIP text encoder -- the sharpest test of whether "
                           "grounding degrades text-side under novel phrasing. "
                           "The only detector measured with an interior optimum: "
                           "CAAP peaks at 0.15 and FALLS to 0.0945 at 0.02, "
                           "because CAAP ranks detections dataset-wide and "
                           "OWLv2's low-confidence scores are not comparable "
                           "across images."),
    )
}

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_TERSE_SYSTEM = ("Just Give the list of objects in given picture seperated by comma. "
                 "Do not write anything else.")

PROMPTS: dict[str, PromptSpec] = {
    s.key: s for s in (
        PromptSpec(
            "default", _TERSE_SYSTEM,
            "Give me a list of objects that you see in this image. Just give the "
            "list of objects comma seperated, don't explain anything. Do not list "
            "sky, street.",
            provenance="The generic instruction used throughout all main "
                       "experiments, byte-identical across COCO and LVIS. This "
                       "is what the original runs executed.",
        ),
        PromptSpec(
            "minimal", _TERSE_SYSTEM,
            "List the objects that you see in this image.",
            provenance="Most formatting and exclusion constraints removed. "
                       "This is also the prompt the original paper's text "
                       "describes, which differs from what its code ran.",
        ),
        PromptSpec(
            "coco-specific", _TERSE_SYSTEM,
            "Give me a list of objects that you see in this image. Just give the "
            "list of objects comma seperated, don't explain anything. Write only "
            "the general name of the objects. Do not list infrastructural "
            "objects. Don't use plural, use singular object names.",
            provenance="Encourages coarse, singular names and suppresses "
                       "infrastructural objects -- tuned to COCO's annotation "
                       "scheme. Present as a commented-out variant in both "
                       "original runner scripts, so a real historical "
                       "condition rather than one invented for the ablation.",
        ),
    )
}

DEFAULT_PROMPT = "default"

#: Earlier internal names, accepted wherever a prompt key is given so that runs
#: and caches written before the rename still resolve. The current names match
#: the paper.
PROMPT_ALIASES = {"original": "default", "paper": "minimal",
                  "granular": "coco-specific"}


def resolve_prompt(key: str) -> str:
    """Canonical prompt key, accepting the pre-rename names."""
    k = PROMPT_ALIASES.get(key, key)
    if k not in PROMPTS:
        raise ValueError(f"unknown prompt {key!r}; have {sorted(PROMPTS)} "
                         f"(aliases: {sorted(PROMPT_ALIASES)})")
    return k

#: Confidence thresholds. "legacy" is what each backend ships or what the
#: original run inherited -- required to regenerate the published table.
#: "v2" was fitted on the frozen 500-image hyperparameter-selection holdout
#: (laod/data/splits.py), which is excluded from every reported number. Each
#: detector was swept against all seven LLMs' label sets and given the threshold
#: maximising mean CAAP@.5:.95. Per-(LLM, detector) tuning was also measured and
#: gained +0.00% to +0.18% over these shared values -- within noise -- so one
#: threshold per detector is used. The per-pair fits are archived in
#: results/tuned_thresholds.json.
#: Thresholding before computing AP truncates the ranked PR curve, so the
#: legacy values systematically understate every detector.
DETECTOR_PRESETS = ("legacy", "v2")


def detector_params(key: str, preset: str = "legacy") -> dict[str, float]:
    """Confidence parameters for a detector under the named preset."""
    if preset not in DETECTOR_PRESETS:
        raise ValueError(f"unknown detector preset {preset!r}; have {DETECTOR_PRESETS}")
    spec = DETECTORS[key]
    return dict(spec.defaults) if preset == "legacy" else {**spec.defaults,
                                                           **spec.v2_defaults}


#: Models that actually run. ``LLMS`` keeps unsupported entries so the reason
#: for an exclusion stays discoverable instead of vanishing from history.
ACTIVE_LLMS: dict[str, LLMSpec] = {k: v for k, v in LLMS.items() if v.supported}


def describe_matrix() -> str:
    """Human-readable summary of the full configuration space."""
    lines = [f"LLMs ({len(ACTIVE_LLMS)} active, {len(LLMS)} declared):"]
    for s in LLMS.values():
        tags = ",".join(t for t, on in
                        (("gated", s.gated), ("cached", s.cached),
                         ("UNSUPPORTED", not s.supported)) if on) or "-"
        lines.append(f"  {s.key:<14} {s.params_b:>5.1f}B  {s.family:<9} [{tags}]  {s.role}")
    lines.append(f"\nDetectors ({len(DETECTORS)}):")
    for d in DETECTORS.values():
        lines.append(f"  {d.key:<12} {d.text_encoder:<22} {'cached' if d.cached else 'download'}")
    lines.append(f"\nPrompts ({len(PROMPTS)}): {', '.join(PROMPTS)}")
    lines.append(f"\nFull grid: {len(ACTIVE_LLMS)} x {len(DETECTORS)} = "
                 f"{len(ACTIVE_LLMS)*len(DETECTORS)} cells")
    return "\n".join(lines)
