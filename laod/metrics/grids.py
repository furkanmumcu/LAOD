"""Threshold grids for CAAP, SNAP and SemLoc-AP.

Each metric summarises a sweep of thresholds into LO / MI / HI intervals plus a
macro average. Which thresholds those intervals contain is the single most
consequential choice in any of them, so the grids are declared here rather than
spelled inline at call sites.

CAAP and SNAP share one interval structure -- the same three disjoint bands over
ten thresholds from 0.50 to 0.95 -- so an interval means the same thing in both,
and SemLoc-AP's two axes are exactly those two grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def key(t: float) -> float:
    """Round a threshold so float arithmetic cannot fracture dict keys."""
    return round(float(t), 4)


@dataclass(frozen=True, slots=True)
class Grid:
    """A named set of LO / MI / HI threshold intervals."""

    name: str
    lo: tuple[float, ...]
    mi: tuple[float, ...]
    hi: tuple[float, ...]
    macro: tuple[float, ...]
    note: str = ""

    @property
    def all_thresholds(self) -> tuple[float, ...]:
        seen = dict.fromkeys(key(t) for t in (*self.lo, *self.mi, *self.hi, *self.macro))
        return tuple(seen)

    def summarise(self, per_threshold: Mapping[float, float]) -> dict[str, float]:
        def mean(ts: Sequence[float]) -> float:
            missing = [t for t in (key(x) for x in ts) if t not in per_threshold]
            if missing:
                raise KeyError(f"grid {self.name!r}: no value at threshold(s) {missing}")
            return float(np.mean([per_threshold[key(t)] for t in ts]))

        return {"LO": mean(self.lo), "MI": mean(self.mi),
                "HI": mean(self.hi), "MACRO": mean(self.macro)}


def _arange(start: float, stop: float, step: float = 0.05) -> tuple[float, ...]:
    """Half-open threshold range, e.g. ``(0.50, 1.00)`` -> 0.50 .. 0.95."""
    return tuple(key(x) for x in np.arange(start, stop, step))


# --------------------------------------------------------------------------
# CAAP
# --------------------------------------------------------------------------

CAAP = Grid(
    name="caap",
    lo=(0.50, 0.55, 0.60),
    mi=(0.65, 0.70, 0.75, 0.80),
    hi=(0.85, 0.90, 0.95),
    macro=tuple(key(x) for x in np.arange(0.50, 1.00, 0.05)),
    note="Disjoint intervals matching the paper's prose; HI stops at the "
         "highest attainable threshold.",
)

# --------------------------------------------------------------------------
# SNAP
# --------------------------------------------------------------------------

SNAP = Grid(
    name="snap",
    lo=(0.50, 0.55, 0.60),
    mi=(0.65, 0.70, 0.75, 0.80),
    hi=(0.85, 0.90, 0.95),
    macro=_arange(0.50, 1.00),
    note="The SNAP grid. Ten cosine thresholds, 0.50 to 0.95, summarised into "
         "three disjoint intervals -- the same values and the same reading as "
         "the CAAP IoU grid, so the two metrics are directly comparable "
         "interval by interval.\n\n"
         "The original implementation derived these from np.arange and "
         "inherited its float quirk: LO held four values rather than three, "
         "LO and MI both contained 0.65, and HI ran to tau=1.00, which "
         "requires near-identical embeddings and deflated every SNAP_HI by "
         "about 22%%. Those three defects are fixed here. The macro average "
         "never included 1.00, so it is unaffected -- only the interval "
         "columns differ from the original, and every published number uses "
         "this grid.",
)

SNAP_YEN = Grid(
    name="yen",
    lo=(0.60,), mi=(0.70, 0.80), hi=(0.90,),
    macro=(0.60, 0.70, 0.80, 0.90),
    note="The four fixed thresholds used by the external reproduction; its "
         "macro average is the number that paper reports.",
)

# Calibrated against the empirical null: thresholds are chosen per dataset and
# per CLIP checkpoint to hit a target false-match rate against unrelated
# ground-truth label pairs. These are the COCO-80 / ViT-B-32 values; see
# laod.metrics.snap.calibrate_thresholds to derive them elsewhere.
SNAP_V2_TARGET_FMR = {"LO": 0.05, "MI": 0.01, "HI": 0.001}

SNAP_CALIBRATED = Grid(
    name="calibrated",
    lo=(0.20,), mi=(0.38,), hi=(0.55,),
    macro=(0.20, 0.38, 0.55),
    note="Mean-centred embeddings, thresholds calibrated to 5%% / 1%% / 0.1%% "
         "false-match rate on COCO-80 with ViT-B/32. Derive per dataset with "
         "calibrate_thresholds() rather than reusing these numbers.",
)

CAAP_GRIDS: dict[str, Grid] = {g.name: g for g in (CAAP,)}
SNAP_GRIDS: dict[str, Grid] = {g.name: g for g in
                               (SNAP, SNAP_YEN, SNAP_CALIBRATED)}
