"""Open-vocabulary detector wrappers.

All three backends present the same two-step contract -- set the label
vocabulary, then detect -- even though they consume text very differently:
YOLO-World embeds class names with CLIP, Grounding DINO takes one dot-separated
phrase string through BERT, OWLv2 takes a list of queries through CLIP. Keeping
the difference inside these classes is what lets the runner treat the detector
as a swappable component.

Returned labels are always mapped back to the exact strings the caller supplied,
never to whatever the model's post-processing echoes, so downstream vocabulary
analysis compares like with like across backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from laod.config import PATHS
from laod.models.registry import DETECTORS, DetectorSpec

logger = logging.getLogger(__name__)

Detection = tuple[np.ndarray, np.ndarray, list[str]]   # boxes xyxy, scores, labels


def _empty() -> Detection:
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []


class BaseDetector(ABC):
    """Set a vocabulary, then localise it in an image.

    A note on ``score_threshold``. Every backend here gates detections on a
    *confidence score* -- keep the box if its score clears the bar. Grounding
    DINO's own API calls this ``box_threshold``, which reads as a threshold on
    box geometry and is not: HuggingFace documents the same argument as
    "Threshold to keep object detection predictions based on confidence score".
    The original scripts used Grounding DINO's name, so it is still accepted as
    an alias, but ``score_threshold`` is what it means.

    These values are **not** IoU thresholds -- those belong to CAAP's evaluation
    grid (0.50-0.95) and never reach a detector. Nor are they comparable across
    backends: each model's score comes off a differently calibrated head, which
    is why each one's threshold was swept separately.
    """

    def __init__(self, spec: DetectorSpec, device: str = "cuda", **overrides) -> None:
        self.spec = spec
        self.device = device
        if "box_threshold" in overrides:
            overrides.setdefault("score_threshold", overrides.pop("box_threshold"))
        self.params = {**spec.defaults, **overrides}
        if "box_threshold" in self.params:
            self.params.setdefault("score_threshold", self.params.pop("box_threshold"))
        self._labels: list[str] = []

    @property
    def labels(self) -> list[str]:
        return self._labels

    def set_labels(self, labels: Sequence[str]) -> None:
        self._labels = [l for l in labels if l and l.strip()]

    @abstractmethod
    def detect(self, image) -> Detection:
        """Return (boxes xyxy, scores, labels) for the current vocabulary."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.spec.key}, {self.params})"


class YoloWorldDetector(BaseDetector):
    """Ultralytics YOLO-World. The detector behind every published number."""

    def __init__(self, spec, device="cuda", weights: Path | None = None, **kw) -> None:
        super().__init__(spec, device, **kw)
        from ultralytics import YOLO
        path = Path(weights or (PATHS.weights / (spec.local_weight or spec.model_id)))
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO-World weights not found at {path}. See datasets/MANIFEST.md.")
        self.model = YOLO(str(path)).to(device)
        logger.info("loaded %s from %s", spec.key, path)

    def set_labels(self, labels):
        super().set_labels(labels)
        if self._labels:
            # set_classes re-embeds the vocabulary; it is the expensive part,
            # so it happens once per image rather than once per detection.
            self.model.set_classes(self._labels)

    def detect(self, image) -> Detection:
        if not self._labels:
            return _empty()
        res = self.model.predict(image, device=self.device, verbose=False,
                                 conf=self.params.get("conf", 0.25))[0]
        if res.boxes is None or len(res.boxes) == 0:
            return _empty()
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        idx = res.boxes.cls.cpu().numpy().astype(int)
        labels = [self._labels[i] if 0 <= i < len(self._labels) else "" for i in idx]
        return boxes, scores, labels


#: Grounding DINO's text encoder is capped at this many tokens. A verbose LLM
#: can exceed it in a single image -- InternVL3 produced 65 labels / 265 tokens
#: on a COCO image during pre-flight, which raises inside the model.
GDINO_MAX_TEXT_TOKENS = 256


class GroundingDinoDetector(BaseDetector):
    """Grounding DINO. Consumes one dot-separated phrase string via BERT.

    Vocabularies longer than the text encoder's limit are split across several
    forward passes and the detections concatenated. Truncating instead would be
    simpler but silently discards labels, which biases exactly the
    vocabulary-coverage analysis this benchmark exists to make -- a verbose
    model would be measured on a quietly shortened label list.

    The trade-off is that labels only share text self-attention within their own
    chunk rather than across the whole list. Grounding DINO scores phrases
    largely independently, so the effect is small, but it is a real difference
    from a single pass and is why the chunk count is recorded per detection.
    """

    def __init__(self, spec, device="cuda", **kw) -> None:
        super().__init__(spec, device, **kw)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(spec.model_id)
        self.model = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(spec.model_id).to(device).eval())
        logger.info("loaded %s", spec.key)

    @staticmethod
    def _as_text(labels: Sequence[str]) -> str:
        """Grounding DINO expects lower-cased phrases terminated by periods."""
        return ". ".join(l.lower() for l in labels) + "." if labels else ""

    @staticmethod
    def _char_ranges(labels: Sequence[str]) -> list[tuple[int, int]]:
        """Character span of every entry inside ``_as_text(labels)``.

        Computed while the prompt is laid out rather than recovered from it, so
        it is exact by construction.
        """
        out, pos = [], 0
        for i, l in enumerate(labels):
            low = l.lower()
            out.append((pos, pos + len(low)))
            pos += len(low) + (2 if i < len(labels) - 1 else 1)   # ". " / "."
        return out

    def _token_owner(self, labels: Sequence[str]) -> np.ndarray:
        """Map each text-token position to the vocabulary entry that owns it.

        Returns an array of length ``GDINO_MAX_TEXT_TOKENS`` holding an entry
        index, or -1 for separators, special tokens and padding.

        Ownership is decided by **character offset**, not by splitting the
        prompt on periods. Splitting is the obvious implementation and it is
        wrong for any entry that contains a period itself -- an LLM naming
        'j.p. morgan sign' or 'usopen.org sign' would have that entry shattered
        into three phantom entries, silently misaligning every later one.
        """
        enc = self.processor.tokenizer(
            self._as_text(labels), truncation=True,
            max_length=GDINO_MAX_TEXT_TOKENS,
            return_offsets_mapping=True, return_special_tokens_mask=True)
        owner = np.full(GDINO_MAX_TEXT_TOKENS, -1, dtype=np.int16)
        ranges = self._char_ranges(labels)
        for ti, ((a, b), special) in enumerate(zip(enc["offset_mapping"],
                                                   enc["special_tokens_mask"])):
            if ti >= GDINO_MAX_TEXT_TOKENS or special or b <= a:
                continue
            for ei, (lo, hi) in enumerate(ranges):
                if a >= lo and b <= hi:
                    owner[ti] = ei
                    break
        return owner

    def _n_tokens(self, labels: Sequence[str]) -> int:
        return len(self.processor.tokenizer(self._as_text(labels))["input_ids"])

    def _chunk(self, labels: Sequence[str]) -> list[list[str]]:
        """Greedily pack labels into groups that fit the text encoder."""
        chunks: list[list[str]] = []
        current: list[str] = []
        for label in labels:
            if current and self._n_tokens(current + [label]) > GDINO_MAX_TEXT_TOKENS:
                chunks.append(current)
                current = [label]
            else:
                current.append(label)
        if current:
            chunks.append(current)
        # A single label over the limit cannot be packed; keep it alone and let
        # the tokenizer truncate rather than failing the whole image.
        return chunks

    def set_labels(self, labels):
        super().set_labels(labels)
        self._chunks = self._chunk(self._labels) if self._labels else []
        self._owners = [self._token_owner(c) for c in self._chunks]
        if len(self._chunks) > 1:
            logger.debug("%d labels exceed %d tokens; split into %d passes",
                         len(self._labels), GDINO_MAX_TEXT_TOKENS, len(self._chunks))

    def _detect_chunk(self, image, labels: Sequence[str],
                      owner: np.ndarray | None = None) -> Detection:
        inputs = self.processor(images=image, text=self._as_text(labels),
                                truncation=True, max_length=GDINO_MAX_TEXT_TOKENS,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=self.params.get("score_threshold", 0.4),
            text_threshold=self.params.get("text_threshold", 0.3),
            target_sizes=[image.size[::-1]],
        )[0]
        if len(res["boxes"]) == 0:
            return _empty()
        boxes = res["boxes"].cpu().numpy().astype(np.float32)
        scores = res["scores"].cpu().numpy().astype(np.float32)
        names = self._labels_from_spans(outputs, labels, owner, scores)
        if names is None:
            raw = res.get("text_labels", res.get("labels", []))
            names = [self._snap_to_vocabulary(str(l), labels) for l in raw]
        return boxes, scores, names

    def _labels_from_spans(self, outputs, labels: Sequence[str],
                           owner: np.ndarray | None,
                           hf_scores: np.ndarray) -> list[str] | None:
        """The vocabulary entry each retained box actually grounds to.

        Grounding DINO scores every (query, text-token) pair, and a box's
        confidence is the maximum over that row. The entry owning the winning
        token is therefore the entry the box is *about* -- and the score the
        detector already reports is that entry's score, so nothing about the
        ranking changes.

        The post-processor does not expose this. It rebuilds a phrase from the
        tokens above ``text_threshold``, which at our operating point
        (score 0.03, text 0.30) is usually *no tokens at all*: ~97% of retained
        boxes come back with an empty name, and multi-token entries that do
        clear it come back as fragments ('##et' for 'tennis racket'). Reading
        the owning entry straight off the logits recovers both cases, and
        returns the caller's exact string rather than a decoded approximation.

        Returns ``None`` if the recomputed score vector does not match the one
        the post-processor returned, which is the only way the query ordering
        could have drifted; the caller then falls back to phrase decoding.
        """
        if owner is None:
            return None
        probs = outputs.logits[0].sigmoid()
        scores, argmax = probs.max(dim=1)
        keep = (scores > self.params.get("score_threshold", 0.4)).nonzero().ravel()
        mine = scores[keep].detach().cpu().numpy().astype(np.float32)
        if mine.shape != hf_scores.shape or not np.array_equal(mine, hf_scores):
            logger.warning("span-label alignment failed (%d vs %d boxes); "
                           "falling back to phrase decoding",
                           mine.size, hf_scores.size)
            return None
        idx = argmax[keep].detach().cpu().numpy()
        out = []
        for t in idx:
            e = int(owner[t]) if t < len(owner) else -1
            # Measured at 0/55,928 boxes, but a winning token outside every
            # entry span would have no vocabulary term to name, so say so
            # rather than inventing one.
            out.append(labels[e] if 0 <= e < len(labels) else "")
        return out

    def detect(self, image) -> Detection:
        if not self._chunks:
            return _empty()
        boxes, scores, labels = [], [], []
        for chunk, owner in zip(self._chunks, self._owners):
            b, s, l = self._detect_chunk(image, chunk, owner)
            if len(b):
                boxes.append(b)
                scores.append(s)
                labels.extend(l)
        if not boxes:
            return _empty()
        return np.concatenate(boxes), np.concatenate(scores), labels

    def _snap_to_vocabulary(self, emitted: str,
                            vocabulary: Sequence[str] | None = None) -> str:
        """Map a decoded phrase back to the caller's exact label string.

        Phrase grounding can return a fragment or a merged span, so an exact
        match is tried first, then containment either way, and only then the
        decoded text itself -- flagged by being absent from the vocabulary.
        """
        vocabulary = self._labels if vocabulary is None else vocabulary
        e = emitted.strip().lower()
        for label in vocabulary:
            if label.lower() == e:
                return label
        for label in vocabulary:
            if e and (e in label.lower() or label.lower() in e):
                return label
        return emitted


class Owlv2Detector(BaseDetector):
    """OWLv2. CLIP text encoder -- the same family SNAP scores with."""

    def __init__(self, spec, device="cuda", **kw) -> None:
        super().__init__(spec, device, **kw)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.processor = Owlv2Processor.from_pretrained(spec.model_id)
        self.model = (Owlv2ForObjectDetection
                      .from_pretrained(spec.model_id).to(device).eval())
        logger.info("loaded %s", spec.key)

    def detect(self, image) -> Detection:
        if not self._labels:
            return _empty()
        queries = [f"a photo of a {l}" for l in self._labels]
        # An LLM that ignores the format instruction can emit a sentence as a
        # "label". Without truncation the processor cannot build a batched
        # tensor and raises, taking down the whole cell; CLIP would ignore the
        # overflow anyway, so truncating is both safe and necessary.
        inputs = self.processor(text=[queries], images=image, padding=True,
                                truncation=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target = self.torch.tensor([image.size[::-1]]).to(self.device)
        res = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target,
            threshold=self.params.get("score_threshold", 0.1),
        )[0]
        if len(res["boxes"]) == 0:
            return _empty()
        idx = res["labels"].cpu().numpy().astype(int)
        return (res["boxes"].cpu().numpy().astype(np.float32),
                res["scores"].cpu().numpy().astype(np.float32),
                [self._labels[i] if 0 <= i < len(self._labels) else "" for i in idx])


_BACKENDS = {
    "yolo_world": YoloWorldDetector,
    "grounding_dino": GroundingDinoDetector,
    "owlv2": Owlv2Detector,
}


def build_detector(key: str, *, device: str = "cuda", **overrides) -> BaseDetector:
    """Instantiate a detector by registry key."""
    if key not in DETECTORS:
        raise ValueError(f"unknown detector {key!r}; have {sorted(DETECTORS)}")
    spec = DETECTORS[key]
    return _BACKENDS[spec.kind](spec, device=device, **overrides)
