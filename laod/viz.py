"""Render detections onto an image.

The visual language follows the original LAOD demo renderer: a fixed vibrant
palette cycled **per detection**, a 2 px box outline, and a filled label chip in
the box's own colour sitting directly above it, flipping inside the box when
that would clip the top edge. Text is white, ``label (0.87)``.

Drawing is deliberately separate from the metric path. Boxes come from a run
store, so what is drawn is what was scored, but the *selection* of which boxes
to draw is a presentation choice: at a benchmark operating point a cell emits
~90 detections per image, which is correct for average precision and unreadable
on a page. ``score_threshold`` and ``top_k`` control that, and the caller says
which it used.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

#: Cycled per detection, as in the original renderer. Gold and LawnGreen are
#: replaced by darker amber and green: at 1.40 and 1.33 contrast against white
#: text they were illegible, against 3.0 for large text under WCAG.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 99, 71),    # Tomato
    (60, 179, 113),   # MediumSeaGreen
    (65, 105, 225),   # RoyalBlue
    (176, 118, 0),    # Amber          (was Gold 255,215,0)
    (186, 85, 211),   # MediumOrchid
    (0, 206, 209),    # DarkTurquoise
    (255, 140, 0),    # DarkOrange
    (46, 139, 46),    # ForestGreen    (was LawnGreen 124,252,0)
    (255, 105, 180),  # HotPink
    (75, 0, 130),     # Indigo
)

TEXT_COLOR = (255, 255, 255)
#: Single colour for ground-truth overlays, so they read as one layer rather
#: than as a second set of predictions. Contrast against white text is 4.33.
GT_COLOR = (46, 139, 46)
BOX_WIDTH = 2
CHIP_PAD = 8
TEXT_PAD = 4
GAP = 5

#: Tried in order; the first that loads wins. Arial first to match the original.
FONT_CANDIDATES = ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                   "LiberationSans-Regular.ttf", "FreeSans.ttf")


def load_font(size: int = 20) -> ImageFont.ImageFont:
    """A TrueType font at ``size``, falling back to PIL's bitmap default."""
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    logger.warning("no TrueType font found; falling back to the PIL default "
                   "(font_size is ignored)")
    return ImageFont.load_default()


def select(scores: Sequence[float], *, score_threshold: float | None = None,
           top_k: int | None = None) -> np.ndarray:
    """Indices to draw, highest score first.

    Returns them in descending-score order so the chip of a more confident
    detection is drawn last and therefore sits on top of its neighbours.
    """
    s = np.asarray(scores, dtype=float)
    idx = np.argsort(-s, kind="stable")
    if score_threshold is not None:
        idx = idx[s[idx] >= score_threshold]
    if top_k is not None:
        idx = idx[:top_k]
    return idx


def _overlaps(a: tuple[int, int, int, int],
              placed: Sequence[tuple[int, int, int, int]]) -> bool:
    return any(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
               for b in placed)


def _place_chip(x: int, y: int, x2: int, y2: int, w: int, h: int,
                img_w: int, img_h: int,
                placed: Sequence[tuple[int, int, int, int]]
                ) -> tuple[int, int]:
    """Top-left for a ``w`` x ``h`` chip on box ``(x, y, x2, y2)``.

    Candidates are tried in order of preference -- above the box first, as in
    the original renderer -- and the first that clears every chip already
    placed wins. When a box is in a dense cluster and nothing clears, the chip
    is slid downwards in short steps; if even that fails it falls back to the
    preferred spot, so a chip is never dropped.
    """
    cands = [
        (x, y - h - GAP),            # above, left-aligned: the default
        (x, y + GAP),                # tucked inside the top
        (x, y2 + GAP),               # below the box
        (x2 - w, y - h - GAP),       # above, right-aligned
        (x2 - w, y2 + GAP),          # below, right-aligned
        (x, y2 - h - GAP),           # inside the bottom
    ]
    cands += [(x, y - h - GAP + k * (h + 2)) for k in range(1, 7)]

    for cx, cy in cands:
        cx = max(0, min(int(cx), img_w - w))
        cy = max(0, min(int(cy), img_h - h))
        if not _overlaps((cx, cy, cx + w, cy + h), placed):
            return cx, cy
    cx = max(0, min(x, img_w - w))
    cy = max(0, min(y - h - GAP, img_h - h))
    return cx, cy


def draw_detections(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
    *,
    score_threshold: float | None = None,
    top_k: int | None = None,
    font_size: int = 20,
    show_score: bool = True,
    colour: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Return a copy of ``image`` with the selected detections drawn on it.

    ``colour`` overrides the per-detection palette with a single colour, which
    is what a ground-truth overlay wants: the boxes are one annotation layer,
    not a ranked list, and cycling hues would imply a distinction that is not
    there.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = load_font(font_size)
    scores = list(scores) if scores is not None else [1.0] * len(boxes)
    idx = select(scores, score_threshold=score_threshold, top_k=top_k)

    # Boxes first, so no outline is drawn over a chip.
    plan, placed = [], []
    for rank, i in enumerate(idx):
        c = colour or PALETTE[rank % len(PALETTE)]
        x, y, x2, y2 = (int(round(float(v))) for v in boxes[i])
        draw.rectangle([(x, y), (x2, y2)], outline=c, width=BOX_WIDTH)

        text = f"{labels[i]} ({scores[i]:.2f})" if show_score else str(labels[i])
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        cw, ch = right - left + CHIP_PAD, bottom - top + CHIP_PAD
        # Placed in descending-score order, so the most confident detection
        # gets its preferred position and weaker ones move out of its way.
        cx, cy = _place_chip(x, y, x2, y2, cw, ch, out.width, out.height, placed)
        placed.append((cx, cy, cx + cw, cy + ch))
        plan.append((cx, cy, cw, ch, text, c))

    # Chips painted weakest first, so the strongest sits on top of any
    # unavoidable overlap in a dense cluster.
    for cx, cy, cw, ch, text, chip in plan[::-1]:
        draw.rectangle([(cx, cy), (cx + cw, cy + ch)], fill=chip)
        draw.text((cx + TEXT_PAD, cy + TEXT_PAD), text, font=font, fill=TEXT_COLOR)
    return out
