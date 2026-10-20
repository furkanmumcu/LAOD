#!/usr/bin/env python
"""Score SemLoc-AP for every configuration that already has CAAP and SNAP.

Offline pass over the stored runs -- no detection, no LLM. The same encoder,
the same maxDets = 100 cap and the same score-ordered matcher as SNAP, so the
three metrics describe one identical detection set.

COCO-OOD is deliberately excluded: it annotates every object as
``unknow object``, so there is no semantic naming to score there and SNAP's
shuffle control already returns a gain of exactly 0.0000 on all 28 of its cells.

Writes one row per cell with the macro average and both marginals, the full
10x10 matrix per cell as JSON, and the monotonicity diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, ".")

from laod.device import pin_cuda_device

DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:0"))

import numpy as np

from laod.config import PATHS
from laod.data.annotations import has_semantic_labels, load_ground_truth
from laod.data.splits import (ABLATION_LVIS_PATH, apply_split,
                              load_ablation_subset)
from laod.io.run_store import load_run
from laod.metrics.semloc import SemLocEvaluator
from laod.metrics.snap import TextEmbedder

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}

#: (stage, dataset, run-dir glob, subset) -- mirrors snap_sbert_all.py, minus
#: COCO-OOD. Every group here has CAAP and SNAP reported for the same cells.
#:
#: The omission of COCO-OOD is enforced below rather than left to this list:
#: SemLoc-AP requires a ground-truth name to compare against, and a group
#: added here by mistake would otherwise be scored without complaint.
GROUPS = [
    ("stage1",      "coco", "coco-test__*",               None),
    ("stage2",      "coco", "coco-test-ablation__*",      "ablation"),
    ("stage3",      "lvis", "lvis-test__*",               None),
    ("stage2_lvis", "lvis", "lvis-test-ablation__*",      "ablation"),
    ("fixed_vocab", "coco", "coco-test-fixed80__*",       None),
    ("vlm",         "coco", "coco-vlm-none__*",           None),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--template", default="{}",
                    help="bare labels, as SNAP uses: a shared prefix injects a "
                         "common component and raises the unrelated-pair floor")
    ap.add_argument("--device", default=DEV)
    ap.add_argument("--out", default="results/semloc.csv")
    ap.add_argument("--matrices", default="results/semloc_matrices.json")
    ap.add_argument("--groups", nargs="+", default=None)
    args = ap.parse_args()

    emb = TextEmbedder(args.model, backend="sbert", device=args.device,
                       template=args.template)
    rows, matrices, gt_cache = [], {}, {}
    t0 = time.time()

    for stage, dataset, pattern, subset in GROUPS:
        if args.groups and stage not in args.groups:
            continue
        if not has_semantic_labels(dataset):
            print(f"[{stage}] skipped: {dataset} annotates every object under "
                  f"a single category, so the semantic criterion has nothing "
                  f"to test", file=sys.stderr)
            continue
        dirs = sorted(d for d in glob.glob(f"outputs/runs/{pattern}")
                      if os.path.isdir(d))
        if not dirs:
            print(f"[{stage}] no runs match {pattern}", file=sys.stderr)
            continue
        ck = (dataset, subset)
        if ck not in gt_cache:
            gt = apply_split(load_ground_truth(getattr(PATHS, ANN[dataset]),
                                               dataset), "test")
            if subset == "ablation":
                sub = load_ablation_subset(
                    ABLATION_LVIS_PATH if dataset == "lvis" else None)
                gt = [g for g in gt if g.image_id in sub]
            gt_cache[ck] = gt
        gt = gt_cache[ck]
        ev = SemLocEvaluator(gt, emb)
        print(f"\n=== {stage} | {dataset} | {len(gt)} images | {len(dirs)} cells ===",
              flush=True)

        for d in dirs:
            cfg, recs = load_run(d)
            preds = [r.to_predictions() for r in recs]
            res = ev.evaluate(preds, progress=False)
            mono = res.monotonicity()
            name = os.path.basename(d)
            matrices[name] = {
                "iou_thresholds": list(res.iou_thresholds),
                "sem_thresholds": list(res.sem_thresholds),
                "matrix": res.matrix().round(6).tolist(),
                "stage": stage, "dataset": dataset,
                "llm": cfg.llm, "detector": cfg.detector, "prompt": cfg.prompt,
            }
            row = dict(
                stage=stage, dataset=dataset, run=name, llm=cfg.llm,
                detector=cfg.detector, prompt=cfg.prompt, images=len(recs),
                detections=res.n_pred, n_gt=res.n_gt,
                invalid_labels=res.n_invalid_labels,
                semloc=round(res.macro, 4),
                semloc_iou50=round(float(np.mean(
                    [res.per_pair[(0.50, u)] for u in res.sem_thresholds])), 4),
                semloc_tau50=round(float(np.mean(
                    [res.per_pair[(t, 0.50)] for t in res.iou_thresholds])), 4),
                semloc_5050=round(res.per_pair[(0.50, 0.50)], 4),
                semloc_7575=round(res.per_pair[(0.75, 0.75)], 4),
                monotone_iou=mono["monotone_in_iou"],
                monotone_sem=mono["monotone_in_sem"],
                rises_iou=mono["increases_in_iou"], rises_sem=mono["increases_in_sem"],
                largest_rise=round(mono["largest_increase"], 6),
                encoder=args.model,
            )
            rows.append(row)
            print(f"  {cfg.llm:<14}{cfg.detector:<13}{cfg.prompt:<14} "
                  f"SemLoc {row['semloc']:.4f}  "
                  f"(.50/.50 {row['semloc_5050']:.4f}, .75/.75 "
                  f"{row['semloc_7575']:.4f})  mono IoU/tau "
                  f"{mono['monotone_in_iou']}/{mono['monotone_in_sem']}",
                  flush=True)

    if not rows:
        raise SystemExit("no cells scored")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(matrices, open(args.matrices, "w"), indent=1)
    nm_i = sum(1 for r in rows if not r["monotone_iou"])
    nm_s = sum(1 for r in rows if not r["monotone_sem"])
    print(f"\n{len(rows)} cells in {(time.time()-t0)/60:.1f} min")
    print(f"non-monotone in IoU: {nm_i}/{len(rows)} | in tau: {nm_s}/{len(rows)}")
    if nm_s or nm_i:
        worst = max(rows, key=lambda r: r["largest_rise"])
        print(f"largest rise on tightening a threshold: {worst['largest_rise']:.6f} "
              f"({worst['run']})")
    print(f"wrote {args.out} and {args.matrices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
