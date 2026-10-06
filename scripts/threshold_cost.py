#!/usr/bin/env python
"""What the detector confidence threshold buys, and what it costs.

The original LAOD ran YOLO-World at Ultralytics' stock ``conf=0.25``; v2's
holdout tuning selected ``conf=0.001``, a ~13x increase in detections per
image. This script scores one stored run at a range of score cuts to separate
the two effects: average precision, which integrates over the ranking, against
precision/recall/F1 at an operating point.

A run stored at a low threshold is a superset of the same run at a higher one,
so every cut here is an exact filter of detections already on disk -- no
re-inference, and no approximation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.predictions import ImagePredictions
from laod.io.run_store import load_run
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP_V2
from laod.metrics.uap import UAPEvaluator

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}
CUTS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2,
        0.25, 0.3, 0.4, 0.5]


def filtered(recs, cut: float) -> list[ImagePredictions]:
    out = []
    for r in recs:
        s = np.asarray(r.scores, dtype=float)
        keep = s >= cut
        out.append(ImagePredictions(
            image_id=r.image_id,
            boxes=np.asarray(r.boxes, dtype=np.float32).reshape(-1, 4)[keep],
            scores=s[keep],
            labels=[l for l, k in zip(r.pred_labels, keep) if k]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="outputs/runs/coco-test__qwen35-9b__yolo-world__original")
    ap.add_argument("--dataset", choices=sorted(ANN), default="coco")
    ap.add_argument("--out", default="results/threshold_cost.csv")
    args = ap.parse_args()

    gt = apply_split(load_ground_truth(getattr(PATHS, ANN[args.dataset]),
                                       args.dataset), "test")
    cfg, recs = load_run(args.run)
    n_gt = sum(len(g) for g in gt)
    print(f"{cfg.llm} + {cfg.detector} | {len(recs)} images | {n_gt} objects\n")

    ev = CAAPEvaluator(gt, CAAP_V2.all_thresholds)
    rows = []
    print(f"{'cut':>7} {'det/img':>8} {'CAAP':>7} {'F1':>7} {'prec':>7} {'rec':>7}")
    for cut in CUTS:
        preds = filtered(recs, cut)
        caap = CAAP_V2.summarise(ev.evaluate(preds, progress=False).per_threshold)["MACRO"]
        u = UAPEvaluator(gt, score_threshold=cut).evaluate(preds)
        n = sum(len(p.scores) for p in preds) / len(preds)
        rows.append(dict(llm=cfg.llm, detector=cfg.detector, dataset=args.dataset,
                         cut=cut, det_per_img=round(n, 2), caap=round(caap, 4),
                         f1=round(u.u_f1, 4), precision=round(u.u_precision, 4),
                         recall=round(u.u_recall, 4)))
        print(f"{cut:>7.3f} {n:>8.1f} {caap:>7.4f} {u.u_f1:>7.4f} "
              f"{u.u_precision:>7.4f} {u.u_recall:>7.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
