#!/usr/bin/env python
"""Score SNAP for every configuration using Sentence-BERT instead of CLIP.

Same runs, same corrected matcher, same maxDets = 100 -- only the text encoder
changes. CLIP's text embeddings sit in a narrow cone, so unrelated labels have
a cosine floor of +0.567 and a threshold of 0.50 admits 100% of them; SNAP_LO
and SNAP_MI therefore score identically under randomly shuffled labels.
MiniLM, trained contrastively for sentence similarity, puts unrelated COCO
pairs at a median of +0.274 and admits 2% at the same threshold.

Thresholds come from ``SNAP_DISJOINT``: the same 0.50-0.95 values as before,
with the legacy grid's three construction defects removed -- LO holds three
values rather than four, LO and MI no longer both contain 0.65, and HI drops
tau=1.00, which needs near-identical embeddings and deflated every SNAP_HI by
about 22%.

The label-shuffle control is scored alongside every cell, because the value
alone cannot show whether a threshold is doing anything: only the gap to its
shuffled twin can.

Labels come from the stored runs, so this is an offline pass -- no detection,
no LLM.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import (ABLATION_LVIS_PATH, apply_split,
                              load_ablation_subset)
from laod.io.run_store import load_run
from laod.metrics.grids import SNAP_DISJOINT
from laod.metrics.snap import SNAPEvaluator, TextEmbedder

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

#: (dataset, run-dir glob, subset) -- the stage each group of runs belongs to.
GROUPS = [
    ("stage1", "coco",     "coco-test__*",            None),
    ("stage2", "coco",     "coco-test-ablation__*",   "ablation"),
    ("stage3", "lvis",     "lvis-test__*",            None),
    ("stage2_lvis", "lvis", "lvis-test-ablation__*",   "ablation"),
    ("stage4", "coco_ood", "coco_ood-test__*",        None),
    ("vlm",    "coco",     "coco-vlm-none__*",        None),
    ("vlm",    "coco_ood", "coco_ood-vlm-none__*",    None),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MINILM)
    ap.add_argument("--template", default="{}",
                    help="bare labels by default: a shared prefix such as "
                         "'a photo of {}' injects a common component and "
                         "raises the unrelated-pair floor from .27 to .39")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/snap_sbert.csv")
    ap.add_argument("--groups", nargs="+", default=None)
    args = ap.parse_args()

    emb = TextEmbedder(args.model, backend="sbert", device=args.device,
                       template=args.template)
    rows, t0 = [], time.time()
    gt_cache: dict[tuple, list] = {}

    for stage, dataset, pattern, subset in GROUPS:
        if args.groups and stage not in args.groups:
            continue
        dirs = sorted(d for d in glob.glob(f"outputs/runs/{pattern}")
                      if os.path.isdir(d))
        if not dirs:
            print(f"[{stage}] no runs match {pattern}")
            continue
        key = (dataset, subset)
        if key not in gt_cache:
            gt = apply_split(load_ground_truth(getattr(PATHS, ANN[dataset]),
                                               dataset), "test")
            if subset == "ablation":
                # Per-dataset draw. The COCO subset shares only 40 of 500 ids
                # with the LVIS one, so the wrong file drops 92% of the images
                # and quietly deflates every score rather than failing.
                sub = load_ablation_subset(
                    ABLATION_LVIS_PATH if dataset == "lvis" else None)
                gt = [g for g in gt if g.image_id in sub]
            gt_cache[key] = gt
        gt = gt_cache[key]
        ev = SNAPEvaluator(gt, emb, grid=SNAP_DISJOINT)

        print(f"\n=== {stage} | {dataset} | {len(dirs)} runs | "
              f"{len(gt)} images ===")
        print(f"{'cell':<46}{'LO':>8}{'MI':>8}{'HI':>8}{'macro':>8}"
              f"{'gainLO':>9}{'gainMI':>9}{'gainHI':>9}")
        for d in dirs:
            cfg, recs = load_run(d)
            preds = [r.to_predictions() for r in recs]
            res = ev.evaluate(preds, control=True, progress=False)
            s = res.summary()
            chance = {k: float(np.mean([res.chance.get(round(t, 4), np.nan)
                                        for t in taus]))
                      for k, taus in (("LO", SNAP_DISJOINT.lo),
                                      ("MI", SNAP_DISJOINT.mi),
                                      ("HI", SNAP_DISJOINT.hi))}
            macro_chance = float(np.mean([v for t, v in res.chance.items()
                                          if t <= 0.95]))
            rows.append(dict(
                stage=stage, dataset=dataset, run=os.path.basename(d),
                llm=cfg.llm, detector=cfg.detector, prompt=cfg.prompt,
                images=len(recs),
                snap_lo=round(s["LO"], 4), snap_mi=round(s["MI"], 4),
                snap_hi=round(s["HI"], 4), snap_macro=round(s["MACRO"], 4),
                chance_lo=round(chance["LO"], 4), chance_mi=round(chance["MI"], 4),
                chance_hi=round(chance["HI"], 4),
                chance_macro=round(macro_chance, 4),
                gain_lo=round(s["LO"] - chance["LO"], 4),
                gain_mi=round(s["MI"] - chance["MI"], 4),
                gain_hi=round(s["HI"] - chance["HI"], 4),
                gain_macro=round(s["MACRO"] - macro_chance, 4),
                encoder=args.model, template=args.template))
            r = rows[-1]
            print(f"{os.path.basename(d)[:45]:<46}{r['snap_lo']:>8.4f}"
                  f"{r['snap_mi']:>8.4f}{r['snap_hi']:>8.4f}{r['snap_macro']:>8.4f}"
                  f"{r['gain_lo']:>+9.4f}{r['gain_mi']:>+9.4f}{r['gain_hi']:>+9.4f}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows, {(time.time()-t0)/60:.0f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
