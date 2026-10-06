#!/usr/bin/env python
"""Write fitted thresholds into the detector registry's v2 preset.

The sweep fits a threshold per (LLM, detector) pair, but the registry holds one
value per detector, so a pair-level disagreement has to be resolved explicitly
rather than silently. Under ``maxDets``, every pair lands on its grid floor --
the cap, not the threshold, decides how many detections survive -- so agreement
is expected and a disagreement is worth stopping for.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

sys.path.insert(0, ".")

REGISTRY = "laod/models/registry.py"


def fitted(paths: list[str]) -> dict[str, dict[float, int]]:
    """detector -> {threshold: how many (dataset, llm) pairs chose it}."""
    picks: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in paths:
        with open(p) as fh:
            rows = list(csv.DictReader(fh))
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["llm"], r["detector"])].append(r)
        for (_, det), rs in by.items():
            best = max(rs, key=lambda r: float(r["objective"]))
            picks[det][float(best["threshold"])] += 1
    return {d: dict(c) for d, c in picks.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", nargs="+", required=True,
                    help="tuning sweep CSVs produced with --max_dets")
    ap.add_argument("--allow_disagreement", action="store_true",
                    help="take the most common value instead of stopping")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # A pattern that matches nothing is a missing input, not an empty set:
    # silently fitting on two of three datasets would hide a disagreement.
    paths, empty = [], []
    for g in args.sweeps:
        hits = sorted(glob.glob(g))
        (paths.extend(hits) if hits else empty.append(g))
    if empty:
        raise SystemExit(f"no sweep file matched: {empty}\n"
                         f"matched {len(paths)} of {len(args.sweeps)} patterns; "
                         f"refusing to fit on a partial set.")
    print("reading:", *[f"\n  {p}" for p in paths], "\n")

    picks = fitted(paths)
    chosen: dict[str, float] = {}
    for det, counts in sorted(picks.items()):
        if len(counts) > 1 and not args.allow_disagreement:
            raise SystemExit(
                f"{det}: pairs disagree on the threshold {counts}. Re-run with "
                f"--allow_disagreement to take the majority, or add a "
                f"per-dataset override.")
        value = max(counts, key=counts.get)
        chosen[det] = value
        print(f"  {det:<13} -> {value:g}   ({counts[value]}/{sum(counts.values())} pairs"
              + (f", others {counts}" if len(counts) > 1 else "") + ")")

    src = open(REGISTRY, encoding="utf-8").read()
    for det, value in chosen.items():
        # Rewrite only the v2 row of this detector's entry.
        pat = re.compile(
            rf'("{re.escape(det)}".*?v2_defaults=\{{"(conf|score_threshold)": )'
            r'([0-9.]+)', re.S)
        m = pat.search(src)
        if not m:
            raise SystemExit(f"could not find a v2_defaults entry for {det}")
        if float(m.group(3)) == value:
            print(f"  {det:<13} registry already at {value:g}")
            continue
        src = src[:m.start(3)] + f"{value:g}" + src[m.end(3):]
        print(f"  {det:<13} registry {m.group(3)} -> {value:g}")

    if args.dry_run:
        print("\ndry run; registry not written")
        return 0
    open(REGISTRY, "w", encoding="utf-8").write(src)
    print(f"\nwrote {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
