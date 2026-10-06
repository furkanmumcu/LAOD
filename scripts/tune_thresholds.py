#!/usr/bin/env python3
"""Fit a confidence threshold per (LLM, detector) pair on the holdout split.

Thresholding is a pure score filter on detector output, verified identical to
running the detector at that threshold directly. So each pair is run **once** at
the lowest threshold of interest and every higher value is evaluated by
filtering -- 28 detector passes instead of ~180, and the grid can be dense for
free.

Fits only on the frozen 500-image tuning split; nothing here touches the images
any reported number comes from.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, ".")
from laod.device import pin_cuda_device

DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:1"))

import numpy as np

from laod.config import PATHS
from laod.data import coco_loader
from laod.data.splits import apply_split
from laod.io.label_cache import LabelCache
from laod.io.predictions import ImagePredictions
from laod.metrics.caap import CAAP_ALL, CAAPEvaluator
from laod.models.detector import build_detector
from laod.models.label_parser import parse_labels
from laod.models.registry import ACTIVE_LLMS, DETECTORS, PROMPTS

PROMPT = "default"
RESULTS = PATHS.root / "results"

# Each benchmark gets its own fit. COCO-OOD in particular is nothing like
# COCO-Val -- one category against 80, 3.3 ground-truth objects per image
# against 7.4 -- so a COCO threshold has no reason to transfer. The holdout for
# every dataset is the same frozen COCO val2017 id list, which is already
# excluded from all three test splits, so tuning here costs no test images.
LOADERS = {"coco": "coco_loader", "lvis": "lvis_loader", "coco_ood": "coco_ood_loader"}

# (parameter name, threshold the detector is actually run at, grid to evaluate)
SWEEP = {
    "yolo-world": ("conf", 0.001,
                   [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
                    0.20, 0.25, 0.30, 0.40, 0.50]),
    "gdino-tiny": ("score_threshold", 0.03,
                   [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                    0.40, 0.45, 0.50]),
    "gdino-base": ("score_threshold", 0.03,
                   [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                    0.40, 0.45, 0.50]),
    "owlv2-base": ("score_threshold", 0.01,
                   [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175,
                    0.20, 0.25, 0.30, 0.40]),
}


def _cap(p: ImagePredictions, max_dets: int) -> ImagePredictions:
    """Top-``max_dets`` detections by score, as COCOeval keeps them."""
    if not max_dets or len(p.scores) <= max_dets:
        return p
    keep = np.argsort(-np.asarray(p.scores, float), kind="stable")[:max_dets]
    keep.sort()
    return ImagePredictions(p.image_id, p.boxes[keep], p.scores[keep],
                            [p.labels[i] for i in keep])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(LOADERS), default="coco")
    ap.add_argument("--llm", nargs="+", default=["all"])
    ap.add_argument("--metric", choices=["caap", "uap"], default="caap",
                    help="objective to maximise; COCO-OOD reports U-AP, so it "
                         "must be tuned on U-AP rather than CAAP")
    ap.add_argument("--max_dets", type=int, default=100,
                    help="keep only the top-N detections per image by score, "
                         "as COCOeval does, BEFORE scoring. The reported "
                         "metric is capped, so the objective must be capped "
                         "too; fitting against the uncapped score chases "
                         "detections the protocol discards. 0 disables.")
    ap.add_argument("--detector", nargs="+", default=["all"],
                    help="restrict the sweep to these detectors; the default "
                         "fits all four")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    import importlib
    loader = importlib.import_module(f"laod.data.{LOADERS[args.dataset]}")
    llm_keys = list(ACTIVE_LLMS) if args.llm == ["all"] else args.llm
    gt_all, ds = loader.load()
    gt = apply_split(gt_all, "tune")
    keep = {g.image_id for g in gt}
    items = [ds[i] for i in range(len(gt_all)) if gt_all[i].image_id in keep]
    # CAAP columns are recorded for every run regardless of the objective, so
    # the evaluator is always built; `score` selects what is maximised.
    ev = CAAPEvaluator(gt, CAAP_ALL)
    score = None
    if args.metric == "uap":
        from laod.metrics.uap import UAPEvaluator
        uev = UAPEvaluator(gt)
        score = lambda preds: uev.evaluate(preds).u_ap
    print(f"[{args.dataset}] tuning split: {len(gt)} images, "
          f"{sum(len(g) for g in gt)} ground-truth objects, "
          f"objective={args.metric}\n")

    # ---- stage 1: one label set per LLM, cached -----------------------------
    labels: dict[str, list[list[str]]] = {}
    for key in llm_keys:
        cache = LabelCache(PATHS.label_cache, f"{args.dataset}_tune", key, PROMPT)
        todo = cache.missing([g.image_id for g in gt])
        if todo:
            from laod.models.llm_agent import build_llm_agent
            print(f"  generating labels: {key} ({len(todo)} images)", flush=True)
            agent = build_llm_agent(key, device_map=DEV)
            t0 = time.time()
            for it in items:
                if it["image_id"] in cache:
                    continue
                cache.put(it["image_id"],
                          parse_labels(agent.generate(it["image"], PROMPTS[PROMPT]),
                                       "strict"))
            print(f"    {(time.time()-t0)/max(len(todo),1):.2f}s/image", flush=True)
            del agent
            _free()
        labels[key] = [cache.get(it["image_id"]).labels for it in items]
        n = sum(map(len, labels[key])) / len(items)
        print(f"  {key:<14} {n:>5.1f} labels/image", flush=True)

    # ---- stage 2: one detector pass per pair, swept by filtering ------------
    rows = []
    wanted = list(SWEEP) if args.detector == ["all"] else args.detector
    unknown = [d for d in wanted if d not in SWEEP]
    if unknown:
        raise SystemExit(f"unknown detector(s) {unknown}; have {sorted(SWEEP)}")
    for det_key in wanted:
        param, run_at, grid = SWEEP[det_key]
        print(f"\n=== {det_key} (run once at {param}={run_at}, "
              f"{len(grid)} thresholds by filtering) ===", flush=True)
        detector = build_detector(det_key, device=DEV, **{param: run_at})
        for llm_key in llm_keys:
            t0 = time.time()
            raw = []
            for it, lab in zip(items, labels[llm_key]):
                detector.set_labels(lab)
                raw.append(detector.detect(it["image"]))
            elapsed = time.time() - t0
            for thr in grid:
                preds = [
                    _cap(ImagePredictions(it["image_id"], b[s >= thr], s[s >= thr],
                                          [l for l, ok in zip(lab, s >= thr) if ok]),
                         args.max_dets)
                    for it, (b, s, lab) in zip(items, raw)
                ]
                if args.metric == "uap":
                    obj = score(preds)
                    res = ev.evaluate(preds, progress=False)
                else:
                    res = ev.evaluate(preds, progress=False)
                    obj = res.macro
                rows.append({
                    "objective": round(obj, 4),
                    "llm": llm_key, "detector": det_key, "param": param,
                    "threshold": thr,
                    "detections_per_image": round(sum(len(p) for p in preds) / len(items), 2),
                    "caap_50_95": round(res.macro, 4),
                    "caap_50": round(res.per_threshold[0.50], 4),
                    "caap_lo": round(res.lo, 4), "caap_hi": round(res.hi, 4),
                })
            best = max((r for r in rows if r["llm"] == llm_key
                        and r["detector"] == det_key), key=lambda r: r["objective"])
            print(f"  {llm_key:<14} best {param}={best['threshold']:<6} "
                  f"{args.metric} {best['objective']:.4f}  "
                  f"({best['detections_per_image']:.1f} det/img, {elapsed:.0f}s)",
                  flush=True)
        del detector
        _free()

    RESULTS.mkdir(parents=True, exist_ok=True)
    # The dataset always stays in the name: a suffix alone let two datasets
    # with the same suffix overwrite each other.
    tag = f"{args.dataset}{args.suffix}"
    with (RESULTS / f"tuning_sweep_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tuned = {}
    for llm_key in llm_keys:
        for det_key in wanted:
            sub = [r for r in rows if r["llm"] == llm_key and r["detector"] == det_key]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["objective"])
            tuned[f"{llm_key}|{det_key}"] = {
                SWEEP[det_key][0]: best["threshold"],
                "objective": best["objective"], "metric": args.metric,
                "caap_50_95": best["caap_50_95"],
                "detections_per_image": best["detections_per_image"],
            }
    (RESULTS / f"tuned_thresholds_{tag}.json").write_text(json.dumps(tuned, indent=1))
    print(f"\nwrote {RESULTS}/tuning_sweep_{tag}.csv ({len(rows)} rows) and "
          f"{RESULTS}/tuned_thresholds_{tag}.json ({len(tuned)} pairs)")
    return 0


def _free() -> None:
    try:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
