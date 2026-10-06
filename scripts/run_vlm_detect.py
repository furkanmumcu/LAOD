#!/usr/bin/env python3
"""Phase 3: a VLM used directly as a detector, with no OVOD.

LAOD splits perception into naming and localising. A grounding-capable VLM does
both in one pass, which makes it the baseline the two-stage design should be
measured against. Both systems are scored by identical metric code on identical
images.

Scoring convention, fixed in advance so it is not chosen after seeing results:

    CAAP, U-AP   ranked by ``logprob_bbox``   (localisation confidence)
    SNAP         ranked by ``logprob_label``  (naming confidence)
    sensitivity  ``order`` reported alongside, never used for a headline

Operating point for U-PRE / U-REC / U-F1: a VLM's exp(log-probability) and a
detector's confidence are different quantities, so applying one numeric cut to
both is equal in form but not in meaning. Each system is therefore reported at
its own **max-F1 operating point**, found by sweeping thresholds -- a standard
way to compare detectors whose scores are not commensurable. A fixed cut is
also reported for continuity with stage 4.

Everything is written to a run store and the run resumes, because a full COCO
pass is ~2.8 h per model and a crash must not cost it.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from laod.config import PATHS
from laod.data import coco_loader
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split, load_ablation_subset
from laod.io.predictions import ImagePredictions
from laod.io.run_store import ImageRecord, RunConfig, RunStore, file_digest
from laod.metrics.caap import CAAP_ALL, CAAPEvaluator
from laod.metrics.grids import SNAP_LEGACY
from laod.metrics.uap import UAPEvaluator
from laod.models.vlm_detector import GROUNDING_PROMPT, QwenVLDetector, parse_detections

#: metric -> the score source it is ranked by. Declared, not discovered.
RANKING = {"caap": "logprob_bbox", "snap": "logprob_label", "uap": "logprob_bbox"}
SOURCES = ("logprob_bbox", "logprob_label", "order")
ANN = {"coco": "coco_ann", "coco_ood": "coco_ood_ann"}


def max_f1(gt, preds, grid=None):
    """Best F1 over a threshold sweep, with the threshold that achieved it."""
    if grid is None:
        all_scores = np.concatenate([p.scores for p in preds if len(p.scores)]) \
            if any(len(p.scores) for p in preds) else np.zeros(1)
        grid = np.unique(np.quantile(all_scores, np.linspace(0, 0.99, 40)))
    best = None
    for t in grid:
        r = UAPEvaluator(gt, score_threshold=float(t)).evaluate(preds)
        if best is None or r.u_f1 > best[0].u_f1:
            best = (r, float(t))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--dataset", choices=sorted(ANN), default="coco")
    ap.add_argument("--subset", choices=["none", "ablation"], default="none",
                    help="'none' uses the whole test split, which is what the "
                         "reported numbers must come from")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--coord_mode", choices=["grid", "norm1000"], default="grid",
                    help="how the model expresses coordinates. Qwen-VL emits "
                         "pixels in its resized frame ('grid'); InternVL emits "
                         "0-1000 normalised values ('norm1000'). Declaring it "
                         "wrong yields boxes that are plausible but uniformly "
                         "mis-sized, so each model's mode is verified against "
                         "ground truth before a run.")
    ap.add_argument("--fixed_threshold", type=float, default=0.3,
                    help="secondary operating point, for continuity with stage 4")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    gt = apply_split(load_ground_truth(getattr(PATHS, ANN[args.dataset]),
                                       args.dataset), "test")
    if args.subset == "ablation":
        sub = load_ablation_subset()
        gt = [g for g in gt if g.image_id in sub]
    if args.limit:
        gt = gt[: args.limit]
    ids = {g.image_id for g in gt}
    _, ds = coco_loader.load()
    imgs = {it["image_id"]: it for i in range(len(ds)) for it in [ds[i]]
            if it["image_id"] in ids}
    print(f"{args.key} / {args.dataset} / subset={args.subset}: {len(gt)} images, "
          f"{sum(len(g.boxes) for g in gt):,} GT objects", flush=True)

    cfg = RunConfig(dataset=f"{args.dataset}-vlm-{args.subset}", llm=args.key,
                    detector="none", prompt="grounding", llm_model_id=args.model,
                    detector_model_id="(direct)",
                    generation_params={"max_new_tokens": args.max_new_tokens,
                                       "do_sample": False},
                    annotation_file=str(getattr(PATHS, ANN[args.dataset])),
                    annotation_md5=file_digest(getattr(PATHS, ANN[args.dataset])),
                    notes=f"VLM direct detection, no OVOD, "
                          f"coord_mode={args.coord_mode}")
    store = RunStore(PATHS.outputs / "runs", cfg, resume=not args.no_resume)
    todo = store.missing([g.image_id for g in gt])
    print(f"  {len(gt) - len(todo)} already done, {len(todo)} to generate", flush=True)

    if todo:
        det = QwenVLDetector(args.model, device=args.device,
                             max_new_tokens=args.max_new_tokens,
                             coord_mode=args.coord_mode)
        pending = set(todo)
        t0, n = time.time(), 0
        for g in gt:
            if g.image_id not in pending:
                continue
            item = imgs[g.image_id]
            try:
                r = det.detect(item["image"], GROUNDING_PROMPT)
            except Exception as exc:
                print(f"  generation failed on {g.image_id}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            b, s, l = r.as_arrays(RANKING["caap"])
            store.append(ImageRecord(
                image_id=g.image_id, file_name=item["file_name"],
                width=item["width"], height=item["height"],
                raw_response=r.raw, labels=l, boxes=b.tolist(), scores=s.tolist(),
                pred_labels=l, gt_labels=list(g.labels),
                parse_flags={"parse_ok": int(r.parse_ok), "n_det": len(l)}))
            n += 1
            if n % 250 == 0:
                print(f"  {n}/{len(todo)}  {(time.time()-t0)/n:.2f}s/img", flush=True)
        del det
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    # ---- scoring, from the store so it never depends on what is in memory ----
    _, recs = __import__("laod.io.run_store", fromlist=["load_run"]).load_run(store.dir)
    byid = {r.image_id: r for r in recs}
    gt = [g for g in gt if g.image_id in byid]
    print(f"  scoring {len(gt)} images", flush=True)

    def preds_for(src):
        out = []
        for g in gt:
            r = byid[g.image_id]
            boxes = np.array(r.boxes, np.float32).reshape(-1, 4)
            if src == RANKING["caap"]:
                s = np.array(r.scores, np.float32)
            else:
                # re-parse the stored reply: every score source is recoverable
                res = parse_detections(r.raw_response or "", 1.0, 1.0,
                                       r.width or 1, r.height or 1)
                s = (res.as_arrays(src)[1] if len(res.detections) == len(r.pred_labels)
                     else np.array(r.scores, np.float32))
            out.append(ImagePredictions(g.image_id, boxes, s, list(r.pred_labels)))
        return out

    ev = CAAPEvaluator(gt, CAAP_ALL)
    from laod.metrics.snap import SNAPEvaluator, TextEmbedder
    sev = SNAPEvaluator(gt, TextEmbedder(device=args.device), grid=SNAP_LEGACY)

    rows = []
    for src in SOURCES:
        p = preds_for(src)
        caap = ev.evaluate(p, progress=False)
        snap = sev.evaluate(p, control=False, progress=False).summary()
        u_fix = UAPEvaluator(gt, score_threshold=args.fixed_threshold).evaluate(p)
        u_best, t_best = max_f1(gt, p)
        nd = sum(len(x.scores) for x in p) / max(len(p), 1)
        pf = sum(1 for g in gt if byid[g.image_id].parse_flags.get("parse_ok") == 0)
        tag = " <- headline" if src in RANKING.values() else ""
        print(f"  [{src:<14}] CAAP {caap.macro:.4f} SNAP {snap['MACRO']:.4f} "
              f"U-AP {u_best.u_ap:.4f} | maxF1 {u_best.u_f1:.3f}@{t_best:.3g} "
              f"(P {u_best.u_precision:.3f} R {u_best.u_recall:.3f}) | "
              f"F1@{args.fixed_threshold} {u_fix.u_f1:.3f} | {nd:.1f} det/img{tag}",
              flush=True)
        rows.append(dict(
            model=args.key, dataset=args.dataset, subset=args.subset,
            score_source=src, images=len(gt),
            caap=round(caap.macro, 4), caap_lo=round(caap.lo, 4),
            caap_mi=round(caap.mi, 4), caap_hi=round(caap.hi, 4),
            snap=round(snap["MACRO"], 4), snap_lo=round(snap["LO"], 4),
            snap_mi=round(snap["MI"], 4), snap_hi=round(snap["HI"], 4),
            u_ap=round(u_best.u_ap, 4),
            u_f1_max=round(u_best.u_f1, 4), u_pre_max=round(u_best.u_precision, 4),
            u_rec_max=round(u_best.u_recall, 4), u_threshold_max=round(t_best, 5),
            u_f1_fixed=round(u_fix.u_f1, 4), u_pre_fixed=round(u_fix.u_precision, 4),
            u_rec_fixed=round(u_fix.u_recall, 4), fixed_threshold=args.fixed_threshold,
            det_per_img=round(nd, 2), parse_failures=pf))

    out = PATHS.root / "results" / f"phase3_vlm_{args.dataset}.csv"
    exists = out.exists()
    with out.open("a" if exists else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)   run dir: {store.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
