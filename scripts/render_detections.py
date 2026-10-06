#!/usr/bin/env python
"""Render stored detections onto images, one standalone file per image.

    python scripts/render_detections.py --images 453722 457559 --score 0.25

Boxes come straight from a run store, so what is drawn is what was scored.
Which boxes are drawn is a presentation choice -- see ``--score``/``--top_k``.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, ".")

from PIL import Image

from laod.config import PATHS
from laod.io.run_store import load_run
from laod.viz import draw_detections, select


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/runs/coco-test__qwen35-9b__yolo-world__original")
    ap.add_argument("--images", type=int, nargs="+", required=True,
                    help="COCO image ids")
    ap.add_argument("--out_dir", default=None,
                    help="defaults to figures/qualitative/<dataset>/pred, "
                         "alongside the ground-truth renders")
    ap.add_argument("--dataset", default=None,
                    help="only used to pick the default output directory")
    ap.add_argument("--score", type=float, default=None,
                    help="draw detections at or above this confidence")
    ap.add_argument("--top_k", type=int, default=None,
                    help="draw at most this many, highest confidence first")
    ap.add_argument("--font_size", type=int, default=20)
    ap.add_argument("--no_score", action="store_true",
                    help="label only, without the confidence in parentheses")
    ap.add_argument("--suffix", default="", help="appended to each output name")
    args = ap.parse_args()

    if args.score is None and args.top_k is None:
        ap.error("pass --score and/or --top_k; drawing every detection at a "
                 "benchmark operating point is unreadable")

    cfg, recs = load_run(args.run)
    ds = args.dataset or os.path.basename(args.run).split("-")[0]
    out_dir = args.out_dir or os.path.join("figures", "qualitative", ds, "pred")
    want = set(args.images)
    found = {r.image_id: r for r in recs if r.image_id in want}
    missing = sorted(want - set(found))
    if missing:
        print(f"not in this run: {missing}")

    os.makedirs(out_dir, exist_ok=True)
    for iid in args.images:
        r = found.get(iid)
        if r is None:
            continue
        img = Image.open(os.path.join(str(PATHS.images), r.file_name))
        n = len(select(r.scores, score_threshold=args.score, top_k=args.top_k))
        out = draw_detections(img, r.boxes, r.pred_labels, r.scores,
                              score_threshold=args.score, top_k=args.top_k,
                              font_size=args.font_size,
                              show_score=not args.no_score)
        # keyed by image id alone, so gt/<id>.png and pred/<id>.png pair up
        path = os.path.join(out_dir, f"{iid}{args.suffix}.png")
        out.save(path)
        print(f"  {path}  ({n} of {len(r.boxes)} detections, "
              f"{img.width}x{img.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
