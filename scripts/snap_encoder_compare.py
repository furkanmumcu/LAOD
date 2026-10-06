#!/usr/bin/env python
"""Does a different text encoder make SNAP's low thresholds mean anything?

SNAP values are not comparable across encoders -- the same tau is a different
constraint in a different embedding geometry. What *is* comparable is the gain
over a label-shuffle control: identical boxes and scores, predicted label
strings permuted dataset-wide. A threshold that carries naming information
scores above its shuffled twin; one that does not, does not.

Reports that gain per interval, so the question "are SNAP_LO and SNAP_MI
informative under this encoder" is answered directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.grids import SNAP_LEGACY
from laod.metrics.snap import SNAPEvaluator, TextEmbedder

#: name -> (backend, model, template, centre)
ENCODERS = {
    "clip-raw":     ("clip",  "ViT-B/32", "a photo of {}", False),
    "clip-centred": ("clip",  "ViT-B/32", "a photo of {}", True),
    "minilm":       ("sbert", "sentence-transformers/all-MiniLM-L6-v2", "{}", False),
    "mpnet":        ("sbert", "sentence-transformers/all-mpnet-base-v2", "{}", False),
}
CELLS = ["coco-test__qwen35-9b__yolo-world__original",
         "coco-test__qwen35-9b__owlv2-base__original",
         "coco-test__gemma4-12b__gdino-base__original"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=1500,
                    help="images per cell; the control only needs enough to "
                         "separate signal from chance, not the full split")
    ap.add_argument("--out", default="results/snap_encoder_compare.json")
    args = ap.parse_args()

    gt_all = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), "test")
    gt = gt_all[: args.limit]
    keep = {g.image_id for g in gt}
    print(f"{len(gt)} images | {sum(len(g) for g in gt)} objects | "
          f"{len(args.cells)} cells x {len(args.encoders)} encoders\n")

    out = {"images": len(gt), "cells": {}}
    for cell in args.cells:
        cfg, recs = load_run(f"outputs/runs/{cell}")
        preds = [r.to_predictions() for r in recs if r.image_id in keep]
        tag = f"{cfg.llm} + {cfg.detector}"
        print(f"=== {tag} ===")
        print(f"{'encoder':<14}{'interval':<10}{'SNAP':>9}{'chance':>9}{'gain':>9}   verdict")
        out["cells"][tag] = {}
        for name in args.encoders:
            backend, model, template, centre = ENCODERS[name]
            emb = TextEmbedder(model, backend=backend, device=args.device,
                               template=template)
            res = SNAPEvaluator(gt, emb, grid=SNAP_LEGACY, center=centre).evaluate(
                preds, control=True, progress=False)
            real, chance = res.summary(), {}
            for k, taus in (("LO", SNAP_LEGACY.lo), ("MI", SNAP_LEGACY.mi),
                            ("HI", SNAP_LEGACY.hi)):
                chance[k] = float(np.mean([res.chance.get(round(t, 4), np.nan)
                                           for t in taus]))
            out["cells"][tag][name] = {
                "snap": {k: round(real[k], 4) for k in ("LO", "MI", "HI", "MACRO")},
                "chance": {k: round(v, 4) for k, v in chance.items()},
                "per_tau": {str(t): [round(res.per_threshold[t], 4),
                                     round(res.chance.get(t, float("nan")), 4)]
                            for t in sorted(res.per_threshold)}}
            for k in ("LO", "MI", "HI"):
                g = real[k] - chance[k]
                verdict = ("vacuous" if g < 0.01 else
                           "weak" if g < 0.05 else "informative")
                print(f"{name if k=='LO' else '':<14}{k:<10}{real[k]:>9.4f}"
                      f"{chance[k]:>9.4f}{g:>+9.4f}   {verdict}")
            print(flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
