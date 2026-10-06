#!/usr/bin/env python3
"""Score every run directory of a stage and write a detailed markdown report.

Reads stored runs only -- no inference -- so a report can be regenerated or a
metric changed without touching the GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
from laod.device import pin_cuda_device

DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:1"))

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP_LEGACY, CAAP_V2, SNAP_LEGACY

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}


def fmt(rows, headers, align=None):
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="glob over outputs/runs")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset", choices=["ablation"], default=None,
                    help="restrict ground truth to a frozen subset. Required when "
                         "the runs cover a subset: scoring 500-image runs against "
                         "4,500 images of ground truth caps recall at ~1/9 and "
                         "deflates CAAP by roughly 7x.")
    ap.add_argument("--snap", action="store_true", help="also compute SNAP")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dirs = sorted(glob.glob(str(PATHS.outputs / "runs" / args.pattern)))
    if not dirs:
        print(f"no runs match {args.pattern}", file=sys.stderr)
        return 1

    gt_all = load_ground_truth(getattr(PATHS, ANN[args.dataset]), args.dataset)
    gt = apply_split(gt_all, args.split)
    if args.subset == "ablation":
        # Per-dataset draw: scoring LVIS runs against the COCO subset would
        # leave almost no ground truth and deflate every number.
        from laod.data.splits import ABLATION_LVIS_PATH, load_ablation_subset
        sub = load_ablation_subset(
            ABLATION_LVIS_PATH if args.dataset == "lvis" else None)
        gt = [g for g in gt if g.image_id in sub]
        if not gt:
            raise SystemExit(f"no {args.dataset} ground truth survived the "
                             f"ablation subset -- wrong subset file?")
    ev = CAAPEvaluator(gt, CAAP_V2.all_thresholds)
    n_gt = sum(len(g) for g in gt)

    cells = []
    for d in dirs:
        cfg, recs = load_run(d)
        preds = [r.to_predictions() for r in recs]
        res = ev.evaluate(preds, progress=False)
        s = CAAP_V2.summarise(res.per_threshold)
        vocab = {}
        flags = {}
        for r in recs:
            for l in r.pred_labels:
                vocab[l] = vocab.get(l, 0) + 1
            for k, v in (r.parse_flags or {}).items():
                if k != "llm_seconds":
                    flags[k] = flags.get(k, 0) + v
        n_det = sum(len(r.scores) for r in recs)
        cell = {
            "dir": os.path.basename(d), "llm": cfg.llm, "detector": cfg.detector,
            "prompt": cfg.prompt, "images": len(recs), "detections": n_det,
            "det_per_img": n_det / max(len(recs), 1),
            "labels_per_img": st.mean([len(r.labels) for r in recs]) if recs else 0,
            "unique_labels": len(vocab),
            "caap_lo": s["LO"], "caap_mi": s["MI"], "caap_hi": s["HI"],
            "caap": s["MACRO"], "caap50": res.per_threshold[0.50],
            "params": cfg.detector_params, "flags": flags,
        }
        if args.snap:
            from laod.metrics.snap import SNAPEvaluator, TextEmbedder
            emb = TextEmbedder(device=DEV)
            sn = SNAPEvaluator(gt, emb, grid=SNAP_LEGACY).evaluate(
                preds, control=True, progress=False)
            sm = sn.summary()
            cell.update(snap_lo=sm["LO"], snap_mi=sm["MI"], snap_hi=sm["HI"],
                        snap_macro=sm["MACRO"],
                        snap_gain_85=sn.gain(0.85) or 0.0)
        cells.append(cell)
        print(f"  scored {cell['dir']}  CAAP {cell['caap']:.4f}", flush=True)

    llms = sorted({c["llm"] for c in cells})
    dets = sorted({c["detector"] for c in cells})
    prompts = sorted({c["prompt"] for c in cells})

    L = [f"# {args.title or args.stage}", "",
         f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
         f"{len(cells)} cells · {args.dataset} `{args.split}` split · "
         f"{len(gt)} images · {n_gt:,} ground-truth objects*", ""]

    L += ["## Per-cell results", "",
          fmt([[c["llm"], c["detector"], c["prompt"], f"{c['labels_per_img']:.1f}",
                f"{c['det_per_img']:.1f}", f"{c['caap_lo']:.4f}", f"{c['caap_mi']:.4f}",
                f"{c['caap_hi']:.4f}", f"**{c['caap']:.4f}**", f"{c['unique_labels']:,}"]
               for c in sorted(cells, key=lambda c: -c["caap"])],
              ["LLM", "detector", "prompt", "lab/img", "det/img",
               "CAAP_LO", "CAAP_MI", "CAAP_HI", "CAAP@.5:.95", "vocab"]), ""]

    if args.snap and "snap_macro" in cells[0]:
        L += ["## SNAP (legacy grid, with label-shuffle control)", "",
              fmt([[c["llm"], c["detector"], f"{c['snap_lo']:.4f}", f"{c['snap_mi']:.4f}",
                    f"{c['snap_hi']:.4f}", f"{c['snap_macro']:.4f}",
                    f"{c['snap_gain_85']:+.4f}"]
                   for c in sorted(cells, key=lambda c: -c.get("snap_macro", 0))],
                  ["LLM", "detector", "SNAP_LO", "SNAP_MI", "SNAP_HI", "macro",
                   "gain@0.85"]),
              "",
              "`gain@0.85` is SNAP minus its label-shuffled baseline. SNAP_LO and "
              "SNAP_MI are reported for continuity with the original table; the "
              "shuffle control shows how much of each is real signal.", ""]

    if len(dets) > 1:
        L += ["## CAAP@.5:.95 by LLM × detector", "",
              fmt([[l] + [f"{next((c['caap'] for c in cells if c['llm']==l and c['detector']==d), float('nan')):.4f}"
                          for d in dets]
                   + [f"**{st.mean([c['caap'] for c in cells if c['llm']==l]):.4f}**"]
                   for l in sorted(llms, key=lambda l: -st.mean([c["caap"] for c in cells if c["llm"]==l]))]
                  + [["**mean**"] + [f"**{st.mean([c['caap'] for c in cells if c['detector']==d]):.4f}**" for d in dets] + [""]],
                  ["LLM"] + dets + ["mean"]), ""]

    if len(prompts) > 1:
        L += ["## Prompt effect", "",
              fmt([[l] + [f"{next((c['caap'] for c in cells if c['llm']==l and c['prompt']==p), float('nan')):.4f}"
                          for p in prompts]
                   for l in sorted(llms)],
                  ["LLM"] + prompts), "",
              fmt([[l] + [f"{next((c['labels_per_img'] for c in cells if c['llm']==l and c['prompt']==p), float('nan')):.1f}"
                          for p in prompts]
                   for l in sorted(llms)],
                  ["LLM (labels/image)"] + prompts), ""]

    xs = [st.mean([c["labels_per_img"] for c in cells if c["llm"] == l]) for l in llms]
    ys = [st.mean([c["caap"] for c in cells if c["llm"] == l]) for l in llms]
    if len(xs) > 2 and st.pstdev(xs) > 0 and st.pstdev(ys) > 0:
        mx, my = st.mean(xs), st.mean(ys)
        r = (sum((a-mx)*(b-my) for a, b in zip(xs, ys))
             / ((sum((a-mx)**2 for a in xs) ** .5) * (sum((b-my)**2 for b in ys) ** .5)))
        L += ["## Label density vs accuracy", "",
              f"Correlation between mean labels/image and mean CAAP@.5:.95 "
              f"across {len(llms)} LLMs: **r = {r:+.3f}**", ""]

    anomalies = []
    for c in cells:
        if c["images"] != len(gt):
            anomalies.append(f"`{c['dir']}` has {c['images']} images, expected {len(gt)}")
        if c["det_per_img"] < 1:
            anomalies.append(f"`{c['dir']}` produced only {c['det_per_img']:.2f} det/img")
        for k, v in c["flags"].items():
            if k == "long_phrase" and v > c["images"] * 0.1:
                anomalies.append(f"`{c['dir']}` {v} long-phrase labels "
                                 f"({v/c['images']:.2f}/image)")
    L += ["## Integrity", "",
          f"- cells scored: **{len(cells)}**",
          f"- images per cell: {sorted({c['images'] for c in cells})}",
          f"- total detections: **{sum(c['detections'] for c in cells):,}**",
          f"- detector parameters: " +
          ", ".join(f"`{d}` {next(c['params'] for c in cells if c['detector']==d)}"
                    for d in dets), ""]
    L += (["### Anomalies", ""] + [f"- {a}" for a in anomalies] + [""]) if anomalies \
        else ["No anomalies detected.", ""]

    out = args.out or (PATHS.root / "reports" / f"{args.stage}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Preserve any hand-written sections appended after the generated ones, so
    # regenerating a report (e.g. to add SNAP) does not silently discard
    # analysis written against an earlier pass.
    keep = ""
    if out.is_file():
        prev = out.read_text(encoding="utf-8")
        marker = "\n## Holdout validation"
        if marker in prev:
            keep = "\n" + prev.split(marker, 1)[1].join([marker, ""])[:0] + marker + \
                   prev.split(marker, 1)[1]
    out.write_text("\n".join(L) + keep, encoding="utf-8")
    json.dump(cells, open(out.with_suffix(".json"), "w"), indent=1, default=str)
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
