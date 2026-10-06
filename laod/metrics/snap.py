"""Semantic Naming Average Precision (SNAP).

Naming quality with localisation discarded: a prediction is a true positive
when its label is close enough, in CLIP text-embedding space, to the label of
an as-yet-unmatched ground-truth object in the same image. No IoU term.

Calibration
-----------
Raw CLIP text embeddings occupy a narrow cone, so cosine similarity between
*unrelated* labels has a high floor -- on COCO-80 with ViT-B/32 the minimum over
all 6,320 unrelated pairs is 0.567 and the median is 0.786 (``person`` vs
``bird`` scores 0.889). Any threshold at or below ~0.65 therefore admits
essentially every pair, and SNAP stops depending on the labels at all: shuffling
every predicted label leaves the score bit-identical.

Two defences are built in.

``center=True`` subtracts the mean text embedding before comparing, which
removes the cone and puts unrelated pairs near zero. The mean is computed from
the *ground-truth* vocabulary alone and never from predictions, so the embedding
space stays a fixed property of the benchmark instead of shifting with whichever
system is under evaluation.

``control=True`` additionally scores a label-shuffled copy of the same
detections. The gap between the two is the only evidence that a given threshold
measures naming rather than nothing, and it is reported on every line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from laod.config import PATHS
from laod.io.predictions import (DEFAULT_MAX_DETS, ImageGroundTruth,
                                 ImagePredictions, align, cap_detections)
from laod.metrics.common import (
    LEGACY_EPS,
    MatchOrder,
    average_precision,
    greedy_match,
    show_progress,
)
from laod.metrics.grids import Grid, SNAP_GRIDS, key

logger = logging.getLogger(__name__)

Backend = Literal["clip", "hf", "sbert"]

#: Sentence-BERT is the default text encoder. CLIP's text tower was trained to
#: align text with *images*, so text-text cosine is incidental to its objective
#: and its embeddings sit in a narrow cone: unrelated COCO labels have a median
#: cosine of +0.786 and a threshold of 0.50 admits 100% of them. SNAP_LO and
#: SNAP_MI are then reproducible by shuffling the predicted labels -- they
#: measure nothing. MiniLM is trained contrastively for sentence similarity,
#: puts unrelated pairs at +0.274, and admits 2% at the same threshold, which
#: makes all three intervals informative against the shuffle control.
#: ``backend="clip"`` reproduces the original behaviour.
DEFAULT_BACKEND: Backend = "sbert"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Per-backend prompt template. CLIP wants "a photo of {}", which matches its
#: training distribution; for a sentence encoder a shared prefix is pure
#: contamination -- it injects the common component the cone problem is made
#: of, and raises MiniLM's unrelated-pair median from +0.274 to +0.386.
TEMPLATES = {"clip": "a photo of {}", "hf": "a photo of {}", "sbert": "{}"}
DEFAULT_TEMPLATE = "a photo of {}"      # the original wrapper: no article
SHUFFLE_SEED = 0


class TextEmbedder:
    """CLIP text encoder with a per-label cache.

    Every unique label is embedded once; repeated labels across 5,000 images
    cost nothing. ``backend="clip"`` is the original OpenAI package loaded from
    a local checkpoint (fp16 on CUDA, matching the original numerics);
    ``backend="hf"`` is ``transformers`` in fp32.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        backend: Backend = DEFAULT_BACKEND,
        device: str | None = None,
        template: str | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.model_name = model
        self.backend = backend
        self.template = template or TEMPLATES.get(backend, "{}")
        self.checkpoint_dir = Path(checkpoint_dir or PATHS.clip_dir)
        self._cache: dict[str, np.ndarray] = {}
        self._model = None
        self._proc = None

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

    def _local_checkpoint(self) -> str:
        """Prefer the in-repo checkpoint so nothing is fetched at eval time."""
        local = self.checkpoint_dir / f"{self.model_name.replace('/', '-')}.pt"
        return str(local) if local.is_file() else self.model_name

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        if self.backend == "clip":
            import clip
            self._model, _ = clip.load(self._local_checkpoint(), device=self.device)
            self._tokenize = clip.tokenize
        elif self.backend == "hf":
            from transformers import CLIPModel, CLIPTokenizerFast
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._proc = CLIPTokenizerFast.from_pretrained(self.model_name)
        elif self.backend == "sbert":
            # Sentence-BERT through plain transformers: AutoModel plus
            # attention-masked mean pooling is exactly what the
            # sentence-transformers package does for these checkpoints, so no
            # extra dependency is needed.
            from transformers import AutoModel, AutoTokenizer
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._proc = AutoTokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError(f"unknown text-encoder backend {self.backend!r}")
        self._model.eval()
        self._torch = torch
        logger.info("text encoder ready: backend=%s model=%s device=%s template=%r",
                    self.backend, self.model_name, self.device, self.template)

    def encode(self, labels: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """Return L2-normalised embeddings, one row per label, order preserved."""
        todo = [l for l in dict.fromkeys(labels) if l not in self._cache]
        if todo:
            self._load()
            torch = self._torch
            for i in range(0, len(todo), batch_size):
                chunk = todo[i:i + batch_size]
                prompts = [self.template.format(l) for l in chunk]
                with torch.no_grad():
                    if self.backend == "clip":
                        tok = self._tokenize(prompts, truncate=True).to(self.device)
                        feats = self._model.encode_text(tok).float()
                    elif self.backend == "sbert":
                        tok = self._proc(prompts, padding=True, truncation=True,
                                         return_tensors="pt").to(self.device)
                        out = self._model(**tok).last_hidden_state
                        m = tok["attention_mask"].unsqueeze(-1).float()
                        feats = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
                        feats = feats.float()
                    else:
                        tok = self._proc(prompts, padding=True, truncation=True,
                                         return_tensors="pt").to(self.device)
                        feats = self._model.get_text_features(**tok).float()
                arr = feats.cpu().numpy()
                for label, vec in zip(chunk, arr):
                    self._cache[label] = vec
        raw = np.stack([self._cache[l] for l in labels])
        return raw / np.linalg.norm(raw, axis=-1, keepdims=True)

    def encode_raw(self, labels: Sequence[str]) -> np.ndarray:
        """Embeddings before normalisation -- needed to compute a centring mean."""
        self.encode(labels)
        return np.stack([self._cache[l] for l in labels])


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def calibrate_thresholds(
    embedder: TextEmbedder,
    vocabulary: Sequence[str],
    target_fmr: Mapping[str, float],
    *,
    center: bool = True,
) -> dict[str, float]:
    """Pick cosine thresholds hitting a target false-match rate.

    The null distribution is every off-diagonal similarity among ``vocabulary``
    -- distinct ground-truth categories, which are unrelated by construction.
    For a target rate ``f`` the threshold is the ``1 - f`` quantile of that
    distribution, so at most an ``f`` fraction of unrelated pairs can match.

    Doing this per dataset and per checkpoint is the point: COCO's 80 categories
    and LVIS's 1,203 have different null distributions, and so do different CLIP
    checkpoints, which is why one hard-coded cosine value is not portable.
    """
    vocab = list(dict.fromkeys(vocabulary))
    if len(vocab) < 2:
        raise ValueError("calibration needs at least two distinct labels")
    raw = embedder.encode_raw(vocab)
    emb = _unit(raw - raw.mean(0, keepdims=True)) if center else _unit(raw)
    sim = emb @ emb.T
    null = sim[~np.eye(len(vocab), dtype=bool)]
    return {name: float(np.quantile(null, 1.0 - f)) for name, f in target_fmr.items()}


@dataclass(frozen=True, slots=True)
class SNAPResult:
    """SNAP at every threshold, with its label-shuffled chance baseline."""

    per_threshold: Mapping[float, float]
    chance: Mapping[float, float] = field(default_factory=dict)
    n_gt: int = 0
    n_pred: int = 0
    centered: bool = False
    grid: Grid | None = None

    def gain(self, t: float) -> float | None:
        """SNAP above chance; ``None`` when no control was run."""
        t = key(t)
        if t not in self.chance:
            return None
        return self.per_threshold[t] - self.chance[t]

    def summary(self) -> dict[str, float]:
        if self.grid is None:
            raise ValueError("no grid attached to this result")
        return self.grid.summarise(self.per_threshold)

    def report(self) -> str:
        head = (f"SNAP  ({self.n_pred} detections vs {self.n_gt} ground truths; "
                f"embeddings {'mean-centred' if self.centered else 'raw'})")
        lines = [head, f"  {'tau':>6} {'SNAP':>9} {'chance':>9} {'gain':>9}"]
        for t, v in sorted(self.per_threshold.items()):
            g = self.gain(t)
            c = f"{self.chance[t]:9.4f}" if key(t) in self.chance else " " * 9
            gs = f"{g:+9.4f}" if g is not None else " " * 9
            flag = "  <- uninformative" if g is not None and g < 0.02 else ""
            lines.append(f"  {t:6.2f} {v:9.4f} {c} {gs}{flag}")
        if self.grid is not None:
            s = self.summary()
            lines.append(f"  [{self.grid.name}] LO {s['LO']:.4f}  MI {s['MI']:.4f}  "
                         f"HI {s['HI']:.4f}  MACRO {s['MACRO']:.4f}")
        return "\n".join(lines)


class SNAPEvaluator:
    """Evaluate semantic naming against a fixed ground-truth set."""

    def __init__(
        self,
        ground_truth: Sequence[ImageGroundTruth],
        embedder: TextEmbedder | None = None,
        thresholds: Sequence[float] | None = None,
        *,
        grid: Grid | str | None = None,
        center: bool = False,
        match_order: MatchOrder = "score",
        max_dets: int | None = DEFAULT_MAX_DETS,
        eps: float = LEGACY_EPS,
    ) -> None:
        self.ground_truth = list(ground_truth)
        self.embedder = embedder or TextEmbedder()
        self.center = center
        self.match_order = match_order
        self.max_dets = max_dets
        self.eps = eps

        if isinstance(grid, str):
            if grid not in SNAP_GRIDS:
                raise ValueError(f"unknown SNAP grid {grid!r}; have {sorted(SNAP_GRIDS)}")
            grid = SNAP_GRIDS[grid]
        self.grid = grid
        if thresholds is None:
            if grid is None:
                raise ValueError("pass either thresholds or a grid")
            thresholds = grid.all_thresholds
        self.thresholds = [key(t) for t in thresholds]

        # Centring mean comes from ground-truth labels only, so the embedding
        # space does not move when the system under evaluation changes.
        self._gt_vocab = sorted({l for g in self.ground_truth for l in g.labels})
        self._mu: np.ndarray | None = None
        if center:
            self._mu = self.embedder.encode_raw(self._gt_vocab).mean(0, keepdims=True)

    def _embed(self, labels: Sequence[str]) -> np.ndarray:
        raw = self.embedder.encode_raw(labels)
        return _unit(raw - self._mu) if self._mu is not None else _unit(raw)

    def evaluate(
        self,
        predictions: Sequence[ImagePredictions],
        *,
        control: bool = True,
        progress: bool = True,
    ) -> SNAPResult:
        gt, preds = align(self.ground_truth,
                          cap_detections(predictions, self.max_dets))
        n_gt = sum(len(g) for g in gt)
        if n_gt == 0:
            raise ValueError("ground truth is empty; SNAP is undefined")

        gt_labels = [g.labels for g in gt]
        pred_labels = [p.labels for p in preds]
        scores = [p.scores for p in preds]

        vocab = sorted({l for im in gt_labels for l in im} |
                       {l for im in pred_labels for l in im})
        emb = self._embed(vocab)
        index = {l: i for i, l in enumerate(vocab)}
        logger.info("SNAP vocabulary: %d unique labels", len(vocab))

        sims = [
            emb[[index[l] for l in pl]] @ emb[[index[l] for l in gl]].T
            if pl and gl else np.zeros((len(pl), len(gl)))
            for pl, gl in zip(pred_labels, gt_labels)
        ]
        flat_scores = np.concatenate(scores) if scores else np.zeros(0)

        shuffled_sims = None
        if control:
            rng = np.random.default_rng(SHUFFLE_SEED)
            flat = [l for im in pred_labels for l in im]
            rng.shuffle(flat)
            it = iter(flat)
            shuffled = [[next(it) for _ in im] for im in pred_labels]
            shuffled_sims = [
                emb[[index[l] for l in pl]] @ emb[[index[l] for l in gl]].T
                if pl and gl else np.zeros((len(pl), len(gl)))
                for pl, gl in zip(shuffled, gt_labels)
            ]

        def score_at(matrices: list[np.ndarray], tau: float) -> float:
            flags = np.concatenate([
                greedy_match(m, tau, scores=p.scores, order=self.match_order)
                for m, p in zip(matrices, preds)
            ]) if matrices else np.zeros(0, bool)
            return average_precision(flat_scores, flags, n_gt, eps=self.eps,
                                     recall_eps=self.eps)

        per_threshold, chance = {}, {}
        for tau in tqdm(self.thresholds, desc="SNAP: thresholds",
                        disable=not show_progress(progress), leave=False):
            per_threshold[tau] = score_at(sims, tau)
            if shuffled_sims is not None:
                chance[tau] = score_at(shuffled_sims, tau)

        return SNAPResult(per_threshold, chance, n_gt, int(flat_scores.size),
                          centered=self._mu is not None, grid=self.grid)
