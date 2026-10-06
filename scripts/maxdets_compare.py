#!/usr/bin/env python
"""Re-score stored runs under COCO's ``maxDets`` cap.

``pycocotools`` keeps only the top 100 detections per image by score; our CAAP
has no such cap, so cells that emit more than 100 are scored on detections the
standard protocol would discard. This re-scores every cell both ways so the
difference is a measured number rather than an assumption.

Offline pass over stored detections -- no inference.
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
from laod.io.predictions import ImagePredictions
from laod.io.run_store import load_run
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP_V2, SNAP_LEGACY

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}


def capped(recs, max_dets: int | None) -> list[ImagePredictions]:
    """Top-``max_dets`` detections per image by score, as COCOeval does."""
    out = []
    for r in recs:
        s = np.asarray(r.scores, dtype=float)
        b = np.asarray(r.boxes, dtype=np.float32).reshape(-1, 4)
        keep = np.arange(len(s))
        if max_dets is not None and len(s) > max_dets:
            keep = np.argsort(-s, kind="stable")[:max_dets]
            keep.sort()          # preserve the stored order within the kept set
        out.append(ImagePredictions(image_id=r.image_id, boxes=b[keep],
                                    scores=s[keep],
                                    labels=[r.pred_labels[i] for i in keep]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(ANN), default="coco")
    ap.add_argument("--pattern", default=None,
                    help="run-directory glob; defaults to the dataset's test runs")
    ap.add_argument("--max_dets", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--snap", action="store_true",
                    help="also score SNAP, which matches on label similarity "
                         "alone and so has more to gain from extra detections")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pattern = args.pattern or f"outputs/runs/{args.dataset}-test__*"
    out = args.out or f"results/maxdets_{args.dataset}.csv"

    gt = apply_split(load_ground_truth(getattr(PATHS, ANN[args.dataset]),
                                       args.dataset), "test")
    ev = CAAPEvaluator(gt, CAAP_V2.all_thresholds)
    sev = None
    if args.snap:
        from laod.metrics.snap import SNAPEvaluator, TextEmbedder
        sev = SNAPEvaluator(gt, TextEmbedder(device=args.device), grid=SNAP_LEGACY)
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    print(f"{len(gt)} images | {sum(len(g) for g in gt)} objects | "
          f"{len(dirs)} cells | cap {args.max_dets}\n")

    rows = []
    head = (f"{'LLM':<14}{'detector':<13}{'det/img':>8}{'capped':>8}"
            f"{'CAAP':>9}{'CAAP@100':>10}{'delta':>8}")
    if sev:
        head += f"{'SNAP':>9}{'SNAP@100':>10}{'delta':>8}"
    print(head)
    for d in dirs:
        cfg, recs = load_run(d)
        n_raw = sum(len(r.scores) for r in recs) / max(len(recs), 1)
        full_s = CAAP_V2.summarise(
            ev.evaluate(capped(recs, None), progress=False).per_threshold)
        full = full_s["MACRO"]
        cap = capped(recs, args.max_dets)
        n_cap = sum(len(p.scores) for p in cap) / max(len(cap), 1)
        cap_s = CAAP_V2.summarise(ev.evaluate(cap, progress=False).per_threshold)
        capped_caap = cap_s["MACRO"]
        delta = (capped_caap / full - 1) * 100
        row = dict(llm=cfg.llm, detector=cfg.detector, dataset=args.dataset,
                   det_per_img=round(n_raw, 2), det_per_img_capped=round(n_cap, 2),
                   caap=round(full, 4), caap_maxdets=round(capped_caap, 4),
                   caap_lo=round(cap_s["LO"], 4), caap_mi=round(cap_s["MI"], 4),
                   caap_hi=round(cap_s["HI"], 4),
                   delta_pct=round(delta, 2), max_dets=args.max_dets)
        line = (f"{cfg.llm:<14}{cfg.detector:<13}{n_raw:>8.1f}{n_cap:>8.1f}"
                f"{full:>9.4f}{capped_caap:>10.4f}{delta:>+7.2f}%")
        if sev:
            sf = sev.evaluate(capped(recs, None), control=False,
                              progress=False).summary()["MACRO"]
            ss = sev.evaluate(cap, control=False, progress=False).summary()
            sc = ss["MACRO"]
            sd = (sc / sf - 1) * 100
            row.update(snap=round(sf, 4), snap_maxdets=round(sc, 4),
                       snap_lo=round(ss["LO"], 4), snap_mi=round(ss["MI"], 4),
                       snap_hi=round(ss["HI"], 4), snap_delta_pct=round(sd, 2))
            line += f"{sf:>9.4f}{sc:>10.4f}{sd:>+7.2f}%"
        rows.append(row)
        print(line, flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    aff = [r for r in rows if r["det_per_img"] > args.max_dets]
    if aff:
        d = [r["delta_pct"] for r in aff]
        print(f"\naffected cells: {len(d)}/{len(rows)} | CAAP mean "
              f"{np.mean(d):+.2f}% | worst {min(d):+.2f}%")
        if sev:
            sd = [r["snap_delta_pct"] for r in aff]
            print(f"{'':>17}{len(sd)}/{len(rows)} | SNAP mean "
                  f"{np.mean(sd):+.2f}% | worst {min(sd):+.2f}%")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
