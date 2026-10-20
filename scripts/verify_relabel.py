#!/usr/bin/env python
"""Prove the Grounding DINO relabel changed labels and nothing else.

Compares every re-detected run against its predecessor on three levels:

1. **Bit identity of boxes and scores.** The relabel takes both straight from
   the same post-processing call as before, so any difference at all means the
   model input or thresholds moved and the run is not comparable.
2. **CAAP and U-AP.** Neither metric reads a label -- CAAP is class-agnostic and
   U-AP matches on IoU alone -- so both must come out identical. This is a
   weaker check than (1) and is run anyway, because it is the invariance the
   paper actually depends on.
3. **What did change**: empty-label rate and how many labels moved.

Exits non-zero if anything in (1) or (2) differs.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
sys.path.insert(0, ".")
import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.uap import UAPEvaluator
from laod.metrics.grids import CAAP_V2

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}


def dataset_of(slug: str) -> str:
    if slug.startswith("coco_ood"): return "coco_ood"
    if slug.startswith("lvis"):     return "lvis"
    return "coco"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="outputs/runs")
    ap.add_argument("--new", default="outputs/runs_spanfix")
    ap.add_argument("--out", default="results/relabel_verification.csv")
    args = ap.parse_args()

    news = sorted(d for d in glob.glob(f"{args.new}/*") if os.path.isdir(d))
    if not news:
        raise SystemExit(f"nothing in {args.new}")
    gt_cache: dict[str, list] = {}
    rows, failures = [], []

    for nd in news:
        slug = os.path.basename(nd)
        od = os.path.join(args.old, slug)
        if not os.path.isdir(od):
            failures.append(f"{slug}: no predecessor in {args.old}"); continue
        ds = dataset_of(slug)
        if ds not in gt_cache:
            gt_cache[ds] = apply_split(
                load_ground_truth(getattr(PATHS, ANN[ds]), ds), "test")
        gt = gt_cache[ds]

        _, orecs = load_run(od); _, nrecs = load_run(nd)
        obyid = {r.image_id: r for r in orecs}
        nbyid = {r.image_id: r for r in nrecs}
        shared = sorted(set(obyid) & set(nbyid))
        bit_ok = len(obyid) == len(nbyid) == len(shared)
        n_lab = n_moved = n_empty_o = n_empty_n = 0
        for i in shared:
            o, n = obyid[i], nbyid[i]
            ob = np.asarray(o.boxes, np.float64); nb = np.asarray(n.boxes, np.float64)
            osc = np.asarray(o.scores, np.float64); nsc = np.asarray(n.scores, np.float64)
            if ob.shape != nb.shape or not np.array_equal(ob, nb): bit_ok = False
            if osc.shape != nsc.shape or not np.array_equal(osc, nsc): bit_ok = False
            n_lab += len(o.pred_labels)
            n_empty_o += sum(1 for x in o.pred_labels if not str(x).strip())
            n_empty_n += sum(1 for x in n.pred_labels if not str(x).strip())
            n_moved += sum(1 for a, b in zip(o.pred_labels, n.pred_labels) if a != b)

        op = [r.to_predictions() for r in orecs]
        np_ = [r.to_predictions() for r in nrecs]
        cev = CAAPEvaluator(gt, CAAP_V2.all_thresholds)
        c_o = CAAP_V2.summarise(cev.evaluate(op, progress=False).per_threshold)["MACRO"]
        c_n = CAAP_V2.summarise(cev.evaluate(np_, progress=False).per_threshold)["MACRO"]
        uev = UAPEvaluator(gt)
        u_o = uev.evaluate(op).u_ap
        u_n = uev.evaluate(np_).u_ap

        caap_ok = c_o == c_n
        uap_ok = u_o == u_n
        if not (bit_ok and caap_ok and uap_ok):
            failures.append(f"{slug}: bit={bit_ok} caap={caap_ok} uap={uap_ok} "
                            f"(CAAP {c_o!r} vs {c_n!r}; U-AP {u_o!r} vs {u_n!r})")
        rows.append(dict(
            run=slug, images=len(shared), detections=n_lab,
            boxes_scores_bit_identical=bit_ok,
            caap_old=f"{c_o:.10f}", caap_new=f"{c_n:.10f}", caap_identical=caap_ok,
            uap_old=f"{u_o:.10f}", uap_new=f"{u_n:.10f}", uap_identical=uap_ok,
            empty_pct_old=round(100 * n_empty_o / max(n_lab, 1), 3),
            empty_pct_new=round(100 * n_empty_n / max(n_lab, 1), 3),
            labels_changed_pct=round(100 * n_moved / max(n_lab, 1), 2)))
        r = rows[-1]
        print(f"{slug:<48} bits {'OK ' if bit_ok else 'DIFF'} "
              f"CAAP {'=' if caap_ok else '!'} U-AP {'=' if uap_ok else '!'}  "
              f"empty {r['empty_pct_old']:6.2f}% -> {r['empty_pct_new']:5.2f}%",
              flush=True)

    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} run(s) verified -> {args.out}")
    if failures:
        print(f"\n!!! {len(failures)} FAILURE(S) -- do not swap the runs:")
        for f in failures: print("   ", f)
        return 1
    print("all runs: boxes and scores bit-identical, CAAP and U-AP unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
