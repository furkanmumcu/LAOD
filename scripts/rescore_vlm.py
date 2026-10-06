#!/usr/bin/env python
"""Re-score the stored Phase 3 VLM runs. No inference.

The VLM runs were written before the matcher was corrected, and their output is
only 9-28% score-sorted, so they are affected by the same defect as Grounding
DINO and OWLv2. Everything needed is in the run store, so this is an offline
pass: attach each score source, score, and rewrite the CSVs.
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
from laod.metrics.grids import CAAP_V2, SNAP_LEGACY
from laod.metrics.uap import UAPEvaluator

#: Declared before the original runs; preserved here unchanged.
RANKING = {"caap": "logprob_bbox", "snap": "logprob_label", "uap": "logprob_bbox"}
SOURCES = ("logprob_bbox", "logprob_label", "order")
ANN = {"coco": "coco_ann", "coco_ood": "coco_ood_ann"}
MODELS = ("qwen25-vl-7b", "internvl3-8b")


def preds_for(recs, source: str) -> list[ImagePredictions]:
    """Predictions ranked by one of the three score sources.

    Only ``logprob_bbox`` is stored as ``scores``; the others are recovered by
    re-parsing the saved reply, which is exactly how the original run produced
    them -- so this reproduces that path rather than approximating it.
    """
    from laod.models.vlm_detector import parse_detections
    out = []
    for r in recs:
        boxes = np.asarray(r.boxes, np.float32).reshape(-1, 4)
        if source == RANKING["caap"]:
            s = np.asarray(r.scores, dtype=np.float32)
        else:
            res = parse_detections(r.raw_response or "", 1.0, 1.0,
                                   r.width or 1, r.height or 1)
            s = (res.as_arrays(source)[1]
                 if len(res.detections) == len(r.pred_labels)
                 else np.asarray(r.scores, dtype=np.float32))
        out.append(ImagePredictions(r.image_id, boxes, s, list(r.pred_labels)))
    return out


def max_f1(gt, preds, n_points: int = 200):
    pooled = [p.scores for p in preds if len(p.scores)]
    grid = (np.unique(np.quantile(np.concatenate(pooled), np.linspace(0, 0.99, n_points)))
            if pooled else np.zeros(1))
    best = None
    for t in grid:
        r = UAPEvaluator(gt, score_threshold=float(t)).evaluate(preds)
        if best is None or r.u_f1 > best[0].u_f1:
            best = (r, float(t))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(ANN), default="coco")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fixed_threshold", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/phase3_vlm_{args.dataset}.csv"

    gt = apply_split(load_ground_truth(getattr(PATHS, ANN[args.dataset]),
                                       args.dataset), "test")
    ev = CAAPEvaluator(gt, CAAP_V2.all_thresholds)
    from laod.metrics.snap import SNAPEvaluator, TextEmbedder
    sev = SNAPEvaluator(gt, TextEmbedder(device=args.device), grid=SNAP_LEGACY)
    print(f"{len(gt)} images | {sum(len(g) for g in gt)} objects\n")

    rows = []
    for model in MODELS:
        d = f"outputs/runs/{args.dataset}-vlm-none__{model}__none__grounding"
        if not os.path.isdir(d):
            print(f"  missing {d}; skipped")
            continue
        cfg, recs = load_run(d)
        # A failure is the parser rejecting the reply, not the model simply
        # finding nothing: many images parse cleanly and return an empty list.
        fails = sum(1 for r in recs if not (r.parse_flags or {}).get("parse_ok", 1))
        for source in SOURCES:
            p = preds_for(recs, source)
            caap = CAAP_V2.summarise(ev.evaluate(p, progress=False).per_threshold)
            snap = sev.evaluate(p, control=False, progress=False).summary()
            u, t = max_f1(gt, p)
            fx = UAPEvaluator(gt, score_threshold=args.fixed_threshold).evaluate(p)
            nd = sum(len(x.scores) for x in p) / max(len(p), 1)
            rows.append(dict(
                model=model, dataset=args.dataset, subset="none", score_source=source,
                images=len(recs), caap=round(caap["MACRO"], 4),
                caap_lo=round(caap["LO"], 4), caap_mi=round(caap["MI"], 4),
                caap_hi=round(caap["HI"], 4), snap=round(snap["MACRO"], 4),
                snap_lo=round(snap["LO"], 4), snap_mi=round(snap["MI"], 4),
                snap_hi=round(snap["HI"], 4), u_ap=round(u.u_ap, 4),
                u_f1_max=round(u.u_f1, 4), u_pre_max=round(u.u_precision, 4),
                u_rec_max=round(u.u_recall, 4), u_threshold_max=round(t, 5),
                u_f1_fixed=round(fx.u_f1, 4), u_pre_fixed=round(fx.u_precision, 4),
                u_rec_fixed=round(fx.u_recall, 4),
                fixed_threshold=args.fixed_threshold,
                det_per_img=round(nd, 2), parse_failures=fails))
            print(f"  {model:<14}{source:<14} CAAP {caap['MACRO']:.4f}  "
                  f"SNAP {snap['MACRO']:.4f}  U-AP {u.u_ap:.4f}  "
                  f"maxF1 {u.u_f1:.3f}  {nd:.1f} det/img", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
