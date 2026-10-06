#!/usr/bin/env python
"""U-AP / U-F1 / U-PRE / U-REC for every COCO-OOD pipeline cell.

Stage 4 reported CAAP only; the unknown-object trio is what Table 2 of the
paper compares against OWOD/UOD baselines and against the Phase 3 VLMs.
Offline pass over stored detections -- no inference.

Each cell is reported at its own max-F1 operating point (thresholds are not
comparable across detectors) and at a fixed cut, for continuity with stage 4.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.uap import UAPEvaluator


def max_f1(gt, preds, n_points: int = 200):
    """Best F1 over a quantile grid of the observed scores."""
    pooled = [p.scores for p in preds if len(p.scores)]
    grid = (np.unique(np.quantile(np.concatenate(pooled),
                                  np.linspace(0, 0.99, n_points)))
            if pooled else np.zeros(1))
    best = None
    for t in grid:
        r = UAPEvaluator(gt, score_threshold=float(t)).evaluate(preds)
        if best is None or r.u_f1 > best[0].u_f1:
            best = (r, float(t))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--out", default="results/phase4_uap_coco_ood.csv")
    ap.add_argument("--fixed_threshold", type=float, default=0.3)
    ap.add_argument("--n_points", type=int, default=200,
                    help="quantile grid resolution for the max-F1 sweep; "
                         "max-F1 over a coarse grid is a lower bound")
    args = ap.parse_args()

    gt = apply_split(load_ground_truth(PATHS.coco_ood_ann, "coco_ood"), "test")
    print(f"{len(gt)} images | {sum(len(g) for g in gt)} unknown objects")

    dirs = sorted(d for d in glob.glob(os.path.join(args.runs, "coco_ood-test__*"))
                  if os.path.isdir(d))
    rows = []
    for d in dirs:
        cfg, recs = load_run(d)
        preds = [r.to_predictions() for r in recs]
        best, t_best = max_f1(gt, preds, args.n_points)
        fixed = UAPEvaluator(gt, score_threshold=args.fixed_threshold).evaluate(preds)
        n_det = sum(len(r.scores) for r in recs) / max(len(recs), 1)
        rows.append(dict(
            llm=cfg.llm, detector=cfg.detector, prompt=cfg.prompt,
            images=len(recs), det_per_img=round(n_det, 2),
            u_ap=round(best.u_ap, 4),
            u_f1_max=round(best.u_f1, 4), u_pre_max=round(best.u_precision, 4),
            u_rec_max=round(best.u_recall, 4), u_threshold_max=round(t_best, 5),
            u_f1_fixed=round(fixed.u_f1, 4), u_pre_fixed=round(fixed.u_precision, 4),
            u_rec_fixed=round(fixed.u_recall, 4),
            fixed_threshold=args.fixed_threshold))
        print(f"  {cfg.llm:14s} {cfg.detector:12s} U-AP {best.u_ap:.4f} | "
              f"maxF1 {best.u_f1:.3f}@{t_best:.3g} "
              f"(P {best.u_precision:.3f} R {best.u_recall:.3f}) | "
              f"{n_det:.1f} det/img", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
