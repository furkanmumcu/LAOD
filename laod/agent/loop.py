"""The LAOD agentic loop: propose, ground, inspect, revise.

One extra round on top of the feedforward pipeline. The agent picks the least
confident detections in an image, crops each one, looks at it, and decides
whether the detection is real -- discarding it if not, and re-grounding it at
crop resolution if it is.

Selection is deliberately parameter-free rather than fitted. For each image let
``o = max(score) - min(score)``; candidates are detections scoring below ``o``,
and at most ``max_crops`` of them are inspected, lowest score first. On the
holdout this picks boxes that are 98-99.6% non-true-positives, so ``discard`` is
low risk, while 13-17% are near-misses, which is what ``refine`` acts on.

Detector confidence thresholds are **not** touched -- they were fitted on the
hyperparameter holdout in Phase 1 and varying them here would confound the loop
with the operating point, which the threshold sweep showed dominates CAAP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image

from laod.io.predictions import ImagePredictions
from laod.models.registry import PromptSpec

logger = logging.getLogger(__name__)

Action = Literal["confirm", "refine", "discard"]

VERIFY_SYSTEM = ("You judge whether a named object is clearly visible in a small "
                 "image crop. Answer with one word only: YES or NO.")
VERIFY_USER = ('Is there a clearly visible "{label}" in this crop? Answer YES or NO.')

#: Padding around a box before cropping, as a fraction of its size. Some context
#: is needed -- a crop tight to a loose box can exclude the very evidence that
#: would reveal it is loose.
PAD = 0.25
MIN_CROP = 32


@dataclass
class LoopStats:
    """What the agent did, counted. The primary output of the experiment."""

    inspected: int = 0
    confirmed: int = 0
    refined: int = 0
    discarded: int = 0
    refine_improved: int = 0
    refine_worsened: int = 0
    refine_unchanged: int = 0
    iou_before: list[float] = field(default_factory=list)
    iou_after: list[float] = field(default_factory=list)
    llm_calls: int = 0
    parse_failures: int = 0

    def merge(self, other: "LoopStats") -> None:
        for k, v in vars(other).items():
            cur = getattr(self, k)
            setattr(self, k, cur + v)


def select(scores: np.ndarray, max_crops: int = 8,
           end: Literal["low", "high"] = "low") -> np.ndarray:
    """Indices to inspect: detections scoring below the image's score range.

    ``end="low"`` takes the least confident of them, which is the specified
    rule and the one most likely to be wrong. ``end="high"`` takes the most
    confident instead: those sit near the top of the ranked precision-recall
    curve, where average precision is actually sensitive, so the two together
    separate "the agent did nothing" from "the agent improved boxes the metric
    cannot see".
    """
    if len(scores) < 2:
        return np.zeros(0, int)
    o = float(scores.max() - scores.min())
    cand = np.where(scores < o)[0]
    if not len(cand):
        return cand
    order = np.argsort(scores[cand])
    if end == "high":
        order = order[::-1]
    return cand[order][:max_crops]


def crop_box(image: Image.Image, box: np.ndarray, pad: float = PAD):
    """Padded crop around a box, plus the offset needed to map coordinates back."""
    w, h = image.size
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx1 = max(0, int(x1 - pad * bw)); cy1 = max(0, int(y1 - pad * bh))
    cx2 = min(w, int(x2 + pad * bw)); cy2 = min(h, int(y2 + pad * bh))
    if cx2 - cx1 < MIN_CROP or cy2 - cy1 < MIN_CROP:
        return None, None
    return image.crop((cx1, cy1, cx2, cy2)), (cx1, cy1)


def parse_yes_no(reply: str) -> bool | None:
    """YES / NO from a free-text reply; ``None`` when the model did neither."""
    r = reply.strip().lower()
    if r.startswith("yes") or " yes" in r[:40]:
        return True
    if r.startswith("no") or " no" in r[:40]:
        return False
    return None


class AgentLoop:
    """One propose-ground-inspect-revise round over a feedforward result."""

    def __init__(self, agent, detector, *, max_crops: int = 8,
                 refine: bool = True, discard: bool = True,
                 select_end: Literal["low", "high"] = "low") -> None:
        self.agent = agent
        self.detector = detector
        self.max_crops = max_crops
        self.select_end = select_end
        self.do_refine = refine
        self.do_discard = discard

    def run(self, image: Image.Image, preds: ImagePredictions,
            gt_boxes: np.ndarray | None = None) -> tuple[ImagePredictions, LoopStats]:
        """Revise ``preds`` for one image.

        ``gt_boxes`` is used only to record whether each revision helped; it
        never influences a decision.
        """
        from laod.metrics.common import box_iou_matrix

        st = LoopStats()
        idx = select(preds.scores, self.max_crops, self.select_end)
        if not len(idx):
            return preds, st

        boxes = preds.boxes.copy()
        scores = preds.scores.copy()
        labels = list(preds.labels)
        drop = np.zeros(len(boxes), bool)

        for i in idx:
            patch, origin = crop_box(image, boxes[i])
            if patch is None:
                continue
            st.inspected += 1
            prompt = PromptSpec("verify", VERIFY_SYSTEM,
                                VERIFY_USER.format(label=labels[i]), "agent loop")
            try:
                reply = self.agent.generate(patch, prompt)
                st.llm_calls += 1
            except Exception:
                logger.exception("verifier failed on image %s", preds.image_id)
                continue
            ans = parse_yes_no(reply)
            if ans is None:
                st.parse_failures += 1
                ans = True            # ambiguous reply: keep the detection

            before = after = None
            if gt_boxes is not None and len(gt_boxes):
                before = float(box_iou_matrix(boxes[i:i+1], gt_boxes).max())

            if not ans and self.do_discard:
                drop[i] = True
                st.discarded += 1
                continue

            if not self.do_refine:
                st.confirmed += 1
                continue

            # re-ground the label inside the crop, where it occupies far more
            # pixels than it did in the full image
            self.detector.set_labels([labels[i]])
            nb, ns, _ = self.detector.detect(patch)
            if not len(nb):
                st.confirmed += 1
                continue
            j = int(np.argmax(ns))
            cand = nb[j].copy()
            cand[[0, 2]] += origin[0]
            cand[[1, 3]] += origin[1]
            if ns[j] <= scores[i]:
                st.confirmed += 1
                continue
            boxes[i] = cand
            st.refined += 1
            if gt_boxes is not None and len(gt_boxes):
                after = float(box_iou_matrix(boxes[i:i+1], gt_boxes).max())
                st.iou_before.append(before); st.iou_after.append(after)
                if after > before + 1e-6:
                    st.refine_improved += 1
                elif after < before - 1e-6:
                    st.refine_worsened += 1
                else:
                    st.refine_unchanged += 1

        keep = ~drop
        return ImagePredictions(preds.image_id, boxes[keep], scores[keep],
                                [l for l, k in zip(labels, keep) if k],
                                raw_response=preds.raw_response), st
