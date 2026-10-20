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
import glob
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
    from laod.metrics.grids import CAAP, SNAP
    for k in ("lo", "mi", "hi"):
        a, b = getattr(CAAP, k), getattr(SNAP, k)
        check(f"{k.upper()} identical", a == b, f"CAAP {a} vs SNAP {b}")
    check("no threshold of 1.00 in either",
          1.0 not in CAAP.hi and 1.0 not in SNAP.hi)
    check("LO and MI are disjoint",
          not (set(CAAP.lo) & set(CAAP.mi)))

    print("\n=== 3. there is one SNAP grid, and the sweeps use it ===")
    from laod.metrics.grids import SNAP, SNAP_GRIDS
    import laod.metrics.grids as _g
    # The legacy/disjoint split is gone: one grid, named once. A second
    # SNAP grid reappearing is how LO/MI/HI columns drift apart between
    # tables that are meant to be comparable.
    check("no SNAP_LEGACY / SNAP_DISJOINT symbol survives",
          not hasattr(_g, "SNAP_LEGACY") and not hasattr(_g, "SNAP_DISJOINT"))
    check("SNAP intervals are disjoint",
          not (set(SNAP.lo) & set(SNAP.mi) or set(SNAP.mi) & set(SNAP.hi)))
    check("SNAP HI stops at 0.95", 1.0 not in SNAP.hi)
    for f in ("scripts/snap_sbert_all.py", "scripts/semloc_all.py",
              "scripts/maxdets_compare.py", "scripts/report_stage.py",
              "scripts/rescore_vlm.py", "scripts/run_vlm_detect.py"):
        src = open(f).read()
        check(f"{os.path.basename(f)} names no other SNAP grid",
              "SNAP_LEGACY" not in src and "SNAP_DISJOINT" not in src)

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

    print("\n=== 6. COCO-OOD carries no SNAP or SemLoc-AP ===")
    from laod.data.annotations import has_semantic_labels
    check("the policy names coco_ood as unnamed",
          not has_semantic_labels("coco_ood"))
    check("and still names coco and lvis",
          has_semantic_labels("coco") and has_semantic_labels("lvis"))
    for f in ("reports/RESULTS.md", "reports/INTERVALS.md"):
        t = open(f).read()
        check(f"{os.path.basename(f)} explains the omission",
              "nothing to name" in t or "no naming to score" in t)
    # The prose check above passed for a year while the CSV shipped a SNAP
    # column anyway, so check the artefact itself, not only what is said
    # about it.
    for f in sorted(glob.glob("results/*coco_ood*.csv")):
        with open(f) as fh:
            cols = next(csv.reader(fh))
        snapcols = [c for c in cols if c.startswith("snap")]
        check(f"{os.path.basename(f)} has no snap* column",
              not snapcols, str(snapcols))
    # snap_sbert.csv is the deliberate exception: its stage4 rows ARE the
    # shuffle control that establishes the exclusion, carried alongside
    # chance_* and gain_* columns and displayed by no report. Check it still
    # reads as a null result rather than quietly becoming a number.
    s4 = [r for r in sb if r["stage"] == "stage4"]
    check("the COCO-OOD control in snap_sbert.csv is still null",
          s4 and all(abs(r["snap_macro"]) < 1e-6 for r in s4),
          f"max {max((abs(r['snap_macro']) for r in s4), default=-1):.6f}")

    # Every entry point that can be pointed at a dataset must consult the
    # policy. This is the check that would have caught phase3_vlm_coco_ood.csv.
    for f in ("scripts/maxdets_compare.py", "scripts/semloc_all.py",
              "scripts/report_stage.py", "scripts/rescore_vlm.py",
              "scripts/run_vlm_detect.py", "run_benchmark.py",
              "eval_metrics_only.py"):
        check(f"{os.path.basename(f)} consults has_semantic_labels",
              "has_semantic_labels" in open(f).read())
    if os.path.exists("results/semloc.csv"):
        check("semloc.csv has no coco_ood row",
              not [r for r in load("results/semloc.csv")
                   if r["dataset"] == "coco_ood"])
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

    # The VLM SNAP column exists in two places, and they drifted: the reports
    # read snap_sbert.csv's vlm rows, so phase3_vlm_coco.csv's own snap column
    # went stale across the CLIP -> Sentence-BERT switch without anything
    # failing. Pin them together.
    sbv = {x["llm"]: x for x in sb
           if x["stage"] == "vlm" and x["dataset"] == "coco"}
    for row in load("results/phase3_vlm_coco.csv"):
        if row["score_source"] != "logprob_bbox":
            continue
        want = sbv.get(row["model"])
        if want:
            check(f"phase3_vlm_coco {row['model']}: SNAP agrees with snap_sbert",
                  abs(row["snap"] - want["snap_macro"]) < 1e-4,
                  f"{row['snap']:.4f} vs {want['snap_macro']:.4f}")

    print("\n=== 8. no stale CLIP SNAP leaking into the reports ===")
    sb1 = {(x["llm"], x["detector"]): x for x in sb if x["stage"] == "stage1"}
    clip = load("results/main_coco.csv")
    r = open("reports/RESULTS.md").read()
    leaked = [f"{c['llm']}+{c['detector']}" for c in clip
              if f"| {c['snap_maxdets']:.4f} " in r
              and abs(c["snap_maxdets"] - sb1[(c["llm"], c["detector"])]["snap_macro"]) > 1e-4]
    check("no CLIP SNAP value appears in RESULTS.md", not leaked, str(leaked[:3]))

    print("\n=== 9. SemLoc-AP ===")
    if not os.path.exists("results/semloc.csv"):
        check("results/semloc.csv present", False, "run scripts/semloc_all.py")
    else:
        from laod.metrics.grids import CAAP, SNAP
        from laod.metrics.semloc import (IOU_THRESHOLDS, SEM_THRESHOLDS,
                                         SemLocEvaluator)
        p = inspect.signature(SemLocEvaluator.__init__).parameters
        check("SemLocEvaluator max_dets=100", p["max_dets"].default == 100,
              f"got {p['max_dets'].default}")
        check("SemLocEvaluator match_order='score'",
              p["match_order"].default == "score",
              f"got {p['match_order'].default}")

        # SemLoc-AP is only interpretable against CAAP and SNAP if its two
        # axes are the same thresholds those two are reported on. If either
        # grid is ever retuned, this is the check that should fail.
        check("IoU axis equals the CAAP grid",
              tuple(IOU_THRESHOLDS) == tuple(CAAP.macro),
              f"{IOU_THRESHOLDS} vs {CAAP.macro}")
        check("cosine axis equals the SNAP grid",
              tuple(SEM_THRESHOLDS) == tuple(SNAP.macro),
              f"{SEM_THRESHOLDS} vs {SNAP.macro}")

        sl = load("results/semloc.csv")
        check("every cell used Sentence-BERT",
              {r["encoder"] for r in sl} == {
                  "sentence-transformers/all-MiniLM-L6-v2"},
              str({r["encoder"] for r in sl}))
        # invalid_labels counts detections whose label was *excluded* from
        # semantic matching, so a nonzero count is the guard working rather
        # than a defect -- MiniLM will embed the empty string and put it
        # within 0.410 of a real name, which is exactly why the exclusion is
        # explicit. What would be a defect is a large rate: before the
        # Grounding DINO span fix ~96% of its retained boxes carried an empty
        # name, so this bound is the tripwire for that regression returning.
        # Measured at 20 of 16,205,448 detections, i.e. 1.2e-6.
        inv = sum(float(r["invalid_labels"]) for r in sl)
        tot = sum(float(r["detections"]) for r in sl)
        rate = inv / tot if tot else 0.0
        check("invalid labels are a negligible fraction (and were excluded)",
              rate < 1e-4, f"{inv:.0f}/{tot:.0f} = {rate:.2e}")

        if os.path.exists("results/semloc_matrices.json"):
            mats = json.load(open("results/semloc_matrices.json"))
            shapes = {(len(m["matrix"]), len(m["matrix"][0]))
                      for m in mats.values()}
            check("every stored matrix is 10x10", shapes == {(10, 10)},
                  str(shapes))
            keyed = [r["run"] for r in sl if r["run"] not in mats]
            check("every scored cell has a stored matrix", not keyed,
                  str(keyed[:3]))
            # Monotonicity is a diagnostic, not an assumption: SemLoc-AP is
            # not monotone by construction, so a rise is reported rather than
            # asserted away. It has been 0 on every cell measured so far.
            rises = [r["run"] for r in sl
                     if float(r["rises_iou"]) or float(r["rises_sem"])]
            check("no non-monotone cell (diagnostic, not a guarantee)",
                  not rises, f"{len(rises)} cells rise: {rises[:3]}")

        if os.path.exists("reports/RESULTS.md"):
            r = open("reports/RESULTS.md").read()
            by_run = {x["run"]: x for x in sl}
            leaked = [k for k, v in by_run.items()
                      if v["stage"] == "stage1"
                      and f"| {float(v['semloc']):.4f} " not in r]
            check("RESULTS.md SemLoc table matches semloc.csv",
                  not leaked, f"{len(leaked)} absent: {leaked[:3]}")

    print("\n" + ("=" * 60))
    print(f"{len(FAILS)} failures" if FAILS else "ALL CHECKS PASSED")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
