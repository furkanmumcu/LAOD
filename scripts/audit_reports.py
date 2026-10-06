#!/usr/bin/env python
"""Verify the reports against their sources, and the conventions against code.

Checks the things that have actually gone wrong in this project rather than a
generic sweep: a table disagreeing with the CSV it was built from, a cell keyed
so loosely that two stages collide, a default silently not applied, a metric
scored on a different grid or encoder than the prose claims.
"""

from __future__ import annotations

import collections
import csv
import inspect
import json
import os
import re
import sys

sys.path.insert(0, ".")

import numpy as np

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def main() -> int:
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()

    # The code checks always run; the table checks need generated reports.
    needed = ["reports/RESULTS.md", "reports/INTERVALS.md",
              "results/main_coco.csv", "results/snap_sbert.csv"]
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        print("\nreport checks will be skipped -- not generated yet:")
        for f in missing:
            print(f"  missing {f}")
        print("  run scripts/build_final_report.py and "
              "scripts/build_intervals_report.py first\n")

    print("\n=== 1. metric defaults (code, not prose) ===")
    from laod.metrics.caap import CAAPEvaluator
    from laod.metrics.snap import SNAPEvaluator, TextEmbedder
    from laod.metrics.uap import UAPEvaluator
    for cls in (CAAPEvaluator, SNAPEvaluator, UAPEvaluator):
        p = inspect.signature(cls.__init__).parameters
        check(f"{cls.__name__} max_dets=100", p["max_dets"].default == 100,
              f"got {p['max_dets'].default}")
        check(f"{cls.__name__} match_order='score'",
              p["match_order"].default == "score", f"got {p['match_order'].default}")
    e = TextEmbedder()
    check("SNAP default encoder is Sentence-BERT", e.backend == "sbert", e.backend)
    check("SNAP default template is bare", e.template == "{}", repr(e.template))

    print("\n=== 2. CAAP and SNAP share one interval structure ===")
    from laod.metrics.grids import CAAP_V2, SNAP_DISJOINT
    for k in ("lo", "mi", "hi"):
        a, b = getattr(CAAP_V2, k), getattr(SNAP_DISJOINT, k)
        check(f"{k.upper()} identical", a == b, f"CAAP {a} vs SNAP {b}")
    check("no threshold of 1.00 in either",
          1.0 not in CAAP_V2.hi and 1.0 not in SNAP_DISJOINT.hi)
    check("LO and MI are disjoint",
          not (set(CAAP_V2.lo) & set(CAAP_V2.mi)))

    print("\n=== 3. the SBERT pass used that grid ===")
    src = open("scripts/snap_sbert_all.py").read()
    check("snap_sbert_all uses SNAP_DISJOINT", "grid=SNAP_DISJOINT" in src)
    check("no SNAP_LEGACY left in it", "SNAP_LEGACY" not in src)

    if missing:
        print("\n(sections 4-8 skipped: reports not generated)")
        return 1 if FAILS else 0

    print("\n=== 4. stage keys cannot collide ===")
    sb = load("results/snap_sbert.csv")
    for stage in ("stage1", "stage2", "stage3", "stage4", "vlm"):
        rows = [r for r in sb if r["stage"] == stage]
        keys = [(r["llm"], r["detector"], r["prompt"], r["dataset"])
                for r in rows]
        check(f"{stage}: {len(rows)} rows, keys unique",
              len(keys) == len(set(keys)))
    s1 = {(r["llm"], r["detector"]) for r in sb if r["stage"] == "stage1"}
    s2 = {(r["llm"], r["detector"]) for r in sb if r["stage"] == "stage2"}
    check("stage1 and stage2 overlap on (llm,detector) -- so stage must key them",
          bool(s1 & s2), f"{len(s1 & s2)} shared pairs")

    print("\n=== 5. report tables match their source CSVs ===")
    txt = open("reports/INTERVALS.md").read()
    DET = {"YOLO-World": "yolo-world", "G-DINO-T": "gdino-tiny",
           "G-DINO-B": "gdino-base", "OWLv2": "owlv2-base"}
    for title, stage, main in (("Stage 1", "stage1", "main_coco.csv"),
                               ("Stage 3", "stage3", "main_lvis.csv")):
        sec = txt.split(f"## {title}")[1].split("<details>")[0]
        for metric, blk_key, src in (("CAAP", "**CAAP by interval", None),
                                     ("SNAP", "**SNAP by interval", "sbert")):
            if blk_key not in sec:
                continue
            blk = sec.split(blk_key)[1].split("**", 1)[0]
            bad = 0
            for line in blk.splitlines():
                m = re.match(r"\| ([A-Za-z0-9-]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
                if not m or m.group(1) not in DET:
                    continue
                det = DET[m.group(1)]
                if src == "sbert":
                    rows = [r for r in sb if r["stage"] == stage and r["detector"] == det]
                    cols = ("snap_lo", "snap_mi", "snap_hi")
                else:
                    rows = [r for r in load(f"results/{main}") if r["detector"] == det]
                    cols = ("caap_lo", "caap_mi", "caap_hi")
                for i, c in enumerate(cols, start=2):
                    if abs(np.mean([r[c] for r in rows]) - float(m.group(i))) > 1e-4:
                        bad += 1
            check(f"{title} {metric} table matches source", bad == 0,
                  f"{bad} mismatched values")

    print("\n=== 6. COCO-OOD SNAP is suppressed everywhere ===")
    for f in ("reports/RESULTS.md", "reports/INTERVALS.md"):
        t = open(f).read()
        check(f"{os.path.basename(f)} explains the omission",
              "nothing to name" in t)
    ood = [r for r in sb if r["stage"] == "stage4"]
    check("and the data agrees it is meaningless",
          all(abs(r["gain_lo"]) < 0.01 for r in ood),
          f"max |gain| {max(abs(r['gain_lo']) for r in ood):.4f}")

    print("\n=== 7. VLM ranking convention ===")
    for f in ("phase3_vlm_coco.csv", "phase3_vlm_coco_ood.csv"):
        rows = load(f"results/{f}")
        for m in {r["model"] for r in rows}:
            g = {s: next(r for r in rows if r["model"] == m and r["score_source"] == s)
                 for s in ("logprob_label", "order")}
            same = all(g["logprob_label"][k] == g["order"][k]
                       for k in g["order"] if k != "score_source")
            check(f"{f} {m}: logprob_label is a duplicate of order", same,
                  "documented, reports use logprob_bbox")
    r = open("reports/RESULTS.md").read()
    check("RESULTS states every VLM metric uses logprob_bbox",
          "Every metric here is ranked by `logprob_bbox`" in r)

    print("\n=== 8. no stale CLIP SNAP leaking into the reports ===")
    sb1 = {(x["llm"], x["detector"]): x for x in sb if x["stage"] == "stage1"}
    clip = load("results/main_coco.csv")
    r = open("reports/RESULTS.md").read()
    leaked = [f"{c['llm']}+{c['detector']}" for c in clip
              if f"| {c['snap_maxdets']:.4f} " in r
              and abs(c["snap_maxdets"] - sb1[(c["llm"], c["detector"])]["snap_macro"]) > 1e-4]
    check("no CLIP SNAP value appears in RESULTS.md", not leaked, str(leaked[:3]))

    print("\n" + ("=" * 60))
    print(f"{len(FAILS)} failures" if FAILS else "ALL CHECKS PASSED")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
