"""Threshold grids for CAAP and SNAP.

Both metrics summarise a sweep of thresholds into LO / MI / HI intervals plus a
macro average. Which thresholds those intervals contain is the single most
consequential choice in either metric, so every variant is named and declared
here rather than being spelled inline at call sites.

A note on the legacy grids, because they are surprising. The original scripts
built intervals with ``np.arange(0.50, 0.65, 0.05)`` and friends. Floating point
makes ``(0.65 - 0.50) / 0.05`` evaluate to 3.0000000000000004, so ``arange``
rounds the length up and emits **four** values where the inline comments claimed
three. Two consequences carried into the published table:

* ``LO`` and ``MI`` overlap -- 0.65 is the last element of one and the first of
  the other.
* ``HI`` ends at 1.00, a threshold nothing can satisfy (CAAP@1.00 = 0.0001),
  which pulls the published CAAP_HI from 0.1105 down to 0.0829.

The ``*_LEGACY`` grids reproduce this faithfully; they are what regenerates the
original paper's numbers. The ``*_V2`` grids are the corrected form.
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
    """Reproduce the legacy ``np.arange`` call, float quirk included."""
    return tuple(key(x) for x in np.arange(start, stop, step))


# --------------------------------------------------------------------------
# CAAP
# --------------------------------------------------------------------------

CAAP_LEGACY = Grid(
    name="legacy",
    lo=_arange(0.50, 0.65),          # (0.50, 0.55, 0.60, 0.65)
    mi=_arange(0.65, 0.85),          # (0.65, 0.70, 0.75, 0.80)
    hi=_arange(0.85, 1.00),          # (0.85, 0.90, 0.95, 1.00)
    macro=tuple(key(x) for x in np.arange(0.50, 1.00, 0.05)),
    note="Reproduces the original paper's Table 1 exactly, overlapping LO/MI "
         "boundary and unreachable IoU=1.00 in HI included.",
)

CAAP_V2 = Grid(
    name="v2",
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

SNAP_LEGACY = Grid(
    name="legacy",
    lo=_arange(0.50, 0.65),
    mi=_arange(0.65, 0.85),
    hi=_arange(0.85, 1.00),
    macro=_arange(0.50, 1.00),
    note="The original SNAP reused the CAAP IoU arrays as cosine thresholds. "
         "On raw CLIP text embeddings every threshold at or below 0.65 admits "
         "~100%% of unrelated label pairs, so LO and MI are not measuring "
         "naming. Kept for reproduction only.",
)

SNAP_DISJOINT = Grid(
    name="disjoint",
    lo=(0.50, 0.55, 0.60),
    mi=(0.65, 0.70, 0.75, 0.80),
    hi=(0.85, 0.90, 0.95),
    macro=_arange(0.50, 1.00),
    note="The legacy thresholds with its three construction defects removed: "
         "LO holds three values rather than four, LO and MI no longer overlap "
         "at 0.65, and HI drops tau=1.00. That last one matters most -- it "
         "requires near-identical embeddings, scores ~0.03 against 0.19-0.24 "
         "for the rest of HI, and deflates every SNAP_HI by about 22%%. The "
         "threshold values themselves are unchanged, so the intervals keep "
         "their intuitive reading.",
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

SNAP_V2_COCO = Grid(
    name="v2",
    lo=(0.20,), mi=(0.38,), hi=(0.55,),
    macro=(0.20, 0.38, 0.55),
    note="Mean-centred embeddings, thresholds calibrated to 5%% / 1%% / 0.1%% "
         "false-match rate on COCO-80 with ViT-B/32. Derive per dataset with "
         "calibrate_thresholds() rather than reusing these numbers.",
)

CAAP_GRIDS: dict[str, Grid] = {g.name: g for g in (CAAP_LEGACY, CAAP_V2)}
SNAP_GRIDS: dict[str, Grid] = {g.name: g for g in
                               (SNAP_LEGACY, SNAP_DISJOINT, SNAP_YEN,
                                SNAP_V2_COCO)}
