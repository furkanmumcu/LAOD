#!/usr/bin/env python
"""Render ground-truth boxes for the qualitative figure.

A companion to ``render_detections.py``: same images, same geometry, so the
two can be placed side by side. Ground truth is drawn in one colour because it
is a single annotation layer rather than a ranked list, and carries no scores.

COCO-OOD annotates every object under one label (``unknow object`` [sic]);
that is rendered as ``unknown``, which is what it means.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, ".")

from PIL import Image

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.viz import GT_COLOR, draw_detections

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}
#: COCO-OOD's single category, as it is spelled in the annotation file.
OOD_LABEL = "unknow object"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(ANN), required=True)
    ap.add_argument("--images", type=int, nargs="+", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--font_size", type=int, default=20)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    out_dir = args.out_dir or f"figures/qualitative/{args.dataset}/gt"

    gt = {g.image_id: g for g in apply_split(
        load_ground_truth(getattr(PATHS, ANN[args.dataset]), args.dataset), "test")}
    # file names live in the run store, which is the only place image id ->
    # file name is recorded for every split
    _, recs = load_run(
        f"outputs/runs/{args.dataset}-test__qwen35-9b__yolo-world__original")
    names = {r.image_id: r.file_name for r in recs}

    os.makedirs(out_dir, exist_ok=True)
    for iid in args.images:
        g = gt.get(iid)
        if g is None or iid not in names:
            print(f"  {iid}: not in this split")
            continue
        img = Image.open(os.path.join(str(PATHS.images), names[iid]))
        labels = ["unknown" if l == OOD_LABEL else l for l in g.labels]
        out = draw_detections(img, g.boxes, labels, None, top_k=len(g.boxes),
                              font_size=args.font_size, show_score=False,
                              colour=GT_COLOR)
        path = os.path.join(out_dir, f"{iid}{args.suffix}.png")
        out.save(path)
        print(f"  {path}  ({len(g.boxes)} boxes, "
              f"{len(set(labels))} distinct labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
