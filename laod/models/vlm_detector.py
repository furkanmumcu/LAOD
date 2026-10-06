"""Direct detection by a vision-language model, with no OVOD in the loop.

LAOD splits perception in two: an LLM names objects, an open-vocabulary detector
localises the names. A VLM with grounding ability can do both at once, which
makes it the baseline the two-stage design should be measured against.

Two things make the comparison awkward, and both are handled explicitly here
rather than papered over.

**Coordinates.** Qwen-VL emits boxes in the coordinate system of the *resized*
image it was shown, not the original. The resize is content-dependent (a
"smart resize" to a multiple of the patch size within a pixel budget), so the
scale factor differs per image. Getting this wrong yields boxes that look
plausible but sit slightly off everywhere, which would read as a localisation
finding rather than a bug -- so the mapping is derived from the processor's own
grid rather than assumed.

**Confidence.** The model emits JSON, not scores, while CAAP and SNAP both rank
detections by confidence. Three sources are produced here:

``logprob_label``   mean token log-probability of the label text
``logprob_bbox``    mean token log-probability of the coordinate text
``order``           position in the emitted list, as a fallback heuristic

Scoring the label and the box separately lets SNAP rank by naming confidence and
CAAP by localisation confidence -- a distinction the two-stage pipeline cannot
make, since there a single detector score stands in for both.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ScoreSource = Literal["logprob_label", "logprob_bbox", "self_conf",
                      "order", "uniform"]

GROUNDING_PROMPT = (
    "Detect all objects in this image and provide their bounding box "
    "coordinates. Output the results strictly in a JSON list format using "
    "'bbox_2d' and 'label'."
)

#: Variant that additionally asks the model to rate its own certainty. This is
#: a *different prompt*, so it needs its own pass -- the self-report cannot be
#: recovered from a generation that never produced it. Running both also
#: measures whether being asked for confidence changes what the model detects,
#: which is a side-effect worth knowing rather than absorbing.
GROUNDING_PROMPT_CONF = (
    "Detect all objects in this image and provide their bounding box "
    "coordinates. Output the results strictly in a JSON list format using "
    "'bbox_2d', 'label', and 'confidence' (a number between 0 and 1 giving "
    "how certain you are that the object is present and correctly located)."
)

PATCH = 14  # Qwen-VL vision patch size; grid units are patches


@dataclass
class VLMDetection:
    """One parsed detection, with every candidate confidence kept."""

    box: list[float]
    label: str
    order_score: float
    logprob_label: float | None = None
    logprob_bbox: float | None = None
    self_conf: float | None = None


@dataclass
class VLMResult:
    raw: str
    detections: list[VLMDetection] = field(default_factory=list)
    parse_ok: bool = True
    parse_note: str = ""

    def as_arrays(self, score: ScoreSource = "logprob_label"):
        """(boxes, scores, labels) using the requested confidence source."""
        if not self.detections:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        boxes = np.array([d.box for d in self.detections], np.float32)
        labels = [d.label for d in self.detections]
        if score == "uniform":
            s = np.ones(len(labels), np.float32)
        elif score == "order":
            s = np.array([d.order_score for d in self.detections], np.float32)
        elif score == "self_conf":
            raw = [d.self_conf for d in self.detections]
            # fall back rather than invent a value: a missing self-report is
            # information, and substituting a constant would silently flatten
            # the ranking for part of the list
            s = (np.array([d.order_score for d in self.detections], np.float32)
                 if any(v is None for v in raw)
                 else np.array(raw, np.float32))
        else:
            raw = [getattr(d, score) for d in self.detections]
            if any(v is None for v in raw):
                s = np.array([d.order_score for d in self.detections], np.float32)
            else:
                # log-probs are negative; exp maps them to (0, 1] so they can be
                # ranked alongside detector scores without changing the order
                s = np.exp(np.array(raw, np.float64)).astype(np.float32)
        return boxes, s, labels


_JSON_BLOCK = re.compile(r"\[.*\]", re.S)


def parse_detections(text: str, scale_x: float, scale_y: float,
                     width: int, height: int) -> VLMResult:
    """Parse the model's JSON reply into boxes in original-image coordinates."""
    res = VLMResult(raw=text)
    m = _JSON_BLOCK.search(text)
    if not m:
        res.parse_ok = False
        res.parse_note = "no JSON list found"
        return res
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        res.parse_ok = False
        res.parse_note = f"invalid JSON: {exc.msg}"
        return res
    if not isinstance(items, list):
        res.parse_ok = False
        res.parse_note = "JSON root is not a list"
        return res

    n = len(items)
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        box = it.get("bbox_2d") or it.get("bbox") or it.get("box_2d")
        label = it.get("label") or it.get("name") or it.get("category")
        conf = it.get("confidence", it.get("score", it.get("conf")))
        if box is None or label is None or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1 * scale_x, x2 * scale_x))
        y1, y2 = sorted((y1 * scale_y, y2 * scale_y))
        x1 = max(0.0, min(x1, width)); x2 = max(0.0, min(x2, width))
        y1 = max(0.0, min(y1, height)); y2 = max(0.0, min(y2, height))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        try:
            sc = float(conf) if conf is not None else None
            if sc is not None and not (0.0 <= sc <= 1.0):
                sc = None       # out-of-range self-reports are unusable
        except (TypeError, ValueError):
            sc = None
        res.detections.append(VLMDetection(
            box=[x1, y1, x2, y2], label=str(label).strip().lower(),
            order_score=1.0 - i / max(n, 1), self_conf=sc))
    if not res.detections:
        res.parse_note = res.parse_note or "no usable entries"
    return res


def token_spans(tokenizer, token_ids) -> list[tuple[int, int]]:
    """Character span of each generated token in the decoded string."""
    spans, prev = [], ""
    for i in range(len(token_ids)):
        cur = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        spans.append((len(prev), len(cur)))
        prev = cur
    return spans


def span_logprob(spans, logprobs, lo: int, hi: int) -> float | None:
    """Mean log-probability of the tokens covering ``[lo, hi)``."""
    vals = [lp for (a, b), lp in zip(spans, logprobs) if a < hi and b > lo]
    return float(np.mean(vals)) if vals else None


def attach_logprobs(res: VLMResult, text: str, spans, logprobs) -> None:
    """Score each detection by the tokens that produced its label and box."""
    m = _JSON_BLOCK.search(text)
    if not m:
        return
    body = m.group(0)
    base = m.start()
    # locate each entry by its label string, then the bbox array near it
    cursor = 0
    for det in res.detections:
        li = body.find(f'"{det.label}"', cursor)
        if li < 0:
            li = body.lower().find(det.label, cursor)
        if li < 0:
            continue
        det.logprob_label = span_logprob(spans, logprobs,
                                         base + li, base + li + len(det.label) + 2)
        bi = body.find("[", body.rfind("{", 0, li) if body.rfind("{", 0, li) > 0 else li)
        bj = body.find("]", bi + 1)
        if 0 <= bi < bj:
            det.logprob_bbox = span_logprob(spans, logprobs, base + bi, base + bj + 1)
        cursor = li + len(det.label)


class QwenVLDetector:
    """Qwen-VL used directly as a detector: image in, boxes and labels out.

    No OVOD is involved. The model is asked for a JSON list and its own token
    probabilities supply the confidence that CAAP and SNAP need for ranking.

    The coordinate scale is read from the processor's ``image_grid_thw`` rather
    than assumed, because the resize is content-dependent: Qwen pads to a
    multiple of the patch size inside a pixel budget, so the factor differs per
    image. Assuming a constant would produce boxes that are plausible but
    uniformly displaced -- a failure that reads as poor localisation rather than
    as a bug, which is why it is derived and then checked against ground truth
    before any number is reported.
    """

    #: How a model expresses coordinates. This cannot be inferred -- Qwen-VL
    #: emits pixels in its own resized frame and exposes the grid to recover
    #: the scale; InternVL emits 0-1000 normalised values and exposes no grid,
    #: so asking for one silently yields a scale of 1.0 and boxes roughly half
    #: the size they should be. Each model's mode is declared and then checked
    #: against ground truth before any number is reported.
    COORD_MODES = ("grid", "norm1000")

    def __init__(self, model_id: str, *, device: str = "cuda:0",
                 dtype: str = "bfloat16", max_new_tokens: int = 1024,
                 coord_mode: str = "grid") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        if coord_mode not in self.COORD_MODES:
            raise ValueError(f"unknown coord_mode {coord_mode!r}; "
                             f"have {self.COORD_MODES}")
        self.coord_mode = coord_mode
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.thinking_disabled = None
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype)).to(device).eval()
        logger.info("loaded %s on %s", model_id, device)

    def _inputs(self, image, prompt: str):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        # Reasoning models default to emitting their chain of thought, which
        # here means pages of deliberation and no JSON at all -- Qwen3.5
        # produced 1,511 characters of reasoning per image and zero parseable
        # detections. The chat template accepts enable_thinking; templates that
        # do not understand it raise, so fall back rather than assume.
        try:
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            self.thinking_disabled = True
        except TypeError:
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            self.thinking_disabled = False
        return self.processor(text=[text], images=[image], return_tensors="pt")

    def _scale(self, inputs, image) -> tuple[float, float]:
        """Original pixels per unit of whatever the model emitted."""
        if self.coord_mode == "norm1000":
            return image.width / 1000.0, image.height / 1000.0
        grid = inputs.get("image_grid_thw")
        if grid is None:
            raise RuntimeError(
                "coord_mode='grid' but the processor returned no image_grid_thw; "
                "the scale cannot be recovered and boxes would be wrong")
        g = grid[0].tolist()
        rh, rw = g[1] * PATCH, g[2] * PATCH
        if rh <= 0 or rw <= 0:
            raise RuntimeError(f"degenerate vision grid {g}")
        return image.width / rw, image.height / rh

    def detect(self, image, prompt: str = GROUNDING_PROMPT) -> VLMResult:
        inputs = self._inputs(image, prompt)
        sx, sy = self._scale(inputs, image)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True)
        n_in = inputs["input_ids"].shape[1]
        new_ids = out.sequences[0, n_in:]
        # keep only the chosen token's log-probability at each step; holding the
        # full vocabulary distribution would cost ~120 MB per image
        logprobs = []
        for step, tok in zip(out.scores, new_ids):
            lp = self.torch.log_softmax(step[0].float(), dim=-1)
            logprobs.append(float(lp[int(tok)]))
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        res = parse_detections(text, sx, sy, image.width, image.height)
        if res.detections:
            spans = token_spans(self.processor.tokenizer, new_ids.tolist())
            attach_logprobs(res, text, spans, logprobs)
        return res
