#!/usr/bin/env python
"""Choose images for the qualitative figure, one list per dataset.

Each dataset is meant to show a different thing, so the selection criteria
differ rather than being "whatever scored well":

* **COCO** -- the generated vocabulary exceeding the 80 annotated categories,
  so prefer images where many proposed names have no COCO counterpart.
* **LVIS** -- fine-grained and long-tail concepts, so prefer images whose
  ground truth uses rare categories and whose proposals are specific.
* **COCO-OOD** -- objects outside the known vocabulary, so prefer images with
  several annotated unknown objects that the pipeline actually found.

Readability is a constraint everywhere: landscape, a workable number of
confident detections, and no single object filling the frame.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.viz import select

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}
#: LVIS marks rare categories 'r', common 'c', frequent 'f'.
RARE = {"r", "c"}


def lvis_rarity() -> dict[str, str]:
    d = json.load(open(PATHS.lvis_ann))
    return {c["name"].replace("_", " ").lower(): c.get("frequency", "f")
            for c in d["categories"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(ANN), required=True)
    ap.add_argument("--llm", default="qwen35-9b")
    ap.add_argument("--detector", default="yolo-world")
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min_shown", type=int, default=4)
    ap.add_argument("--max_shown", type=int, default=11)
    args = ap.parse_args()

    gt = {g.image_id: g for g in apply_split(
        load_ground_truth(getattr(PATHS, ANN[args.dataset]), args.dataset), "test")}
    _, recs = load_run(
        f"outputs/runs/{args.dataset}-test__{args.llm}__{args.detector}__original")
    rarity = lvis_rarity() if args.dataset == "lvis" else {}

    cand = []
    for r in recs:
        g = gt.get(r.image_id)
        if g is None or not r.boxes:
            continue
        ar = r.width / max(r.height, 1)
        if not (1.2 <= ar <= 1.7):                      # landscape, uncropped
            continue
        idx = select(r.scores, score_threshold=args.score)
        if not (args.min_shown <= len(idx) <= args.max_shown):
            continue
        shown = {r.pred_labels[i].lower() for i in idx}
        if len(shown) < 3:                              # avoid one repeated name
            continue
        boxes = np.asarray(r.boxes, float)[idx]
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                / max(r.width * r.height, 1))
        if area.max() > 0.92:                           # one box swallowing the frame
            continue

        gtl = {l.lower() for l in g.labels}
        if args.dataset == "coco":
            score = len(shown - gtl) + 0.5 * len(shown)
        elif args.dataset == "lvis":
            rare = sum(1 for l in gtl if rarity.get(l, "f") in RARE)
            score = 2.0 * rare + len(shown & gtl) + 0.3 * len(shown)
        else:
            score = 2.0 * len(g.boxes) + len(shown)
        cand.append((score, r.image_id, len(idx), len(shown),
                     sorted(shown)[:7], sorted(gtl)[:5]))

    cand.sort(reverse=True)
    print(f"{args.dataset}: {len(cand)} candidates; top {args.n}\n")
    ids = []
    for s, iid, n, nu, shown, g in cand[: args.n]:
        ids.append(iid)
        print(f"  {iid:>8}  {n:>2} shown / {nu} names | gt: {', '.join(g)}")
        print(f"            {', '.join(shown)}")
    print("\n" + " ".join(str(i) for i in ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
