#!/usr/bin/env python
"""Score the vocabulary-feedback experiment and produce its diagnostics.

Three layers, in increasing specificity to the research question:

1. **Benchmark metrics** -- CAAP, SNAP (Sentence-BERT) and SemLoc-AP, computed
   by the current authoritative evaluators on D0 and D1.
2. **Vocabulary diagnostics** -- what the revision did: sizes, removals,
   renames, additions, and what happened to zero-detection concepts.
3. **Semantic recovery diagnostics** -- whether a revision brought a ground-truth
   concept into semantic coverage that C0 lacked, or dropped one it had. These
   are diagnostics, not benchmark metrics.

Offline: every number comes from stored runs. No inference.
"""

from __future__ import annotations

import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from laod.device import pin_cuda_device

DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:0"))

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP, SNAP
from laod.metrics.semloc import SemLocEvaluator
from laod.metrics.snap import SNAPEvaluator, TextEmbedder

CONDITIONS = ("baseline", "selfreflect", "feedback_refine", "feedback_full")
#: Fixed threshold for the headline coverage table; the full sweep is also run.
COVER_TAU = 0.60
TAU_GRID = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def run_dir(split: str, condition: str, llm: str, det: str) -> Path:
    if condition == "baseline":
        return (PATHS.outputs / "runs" /
                f"coco-{split}-fb-baseline__{llm}__{det}__default")
    return PATHS.outputs / "runs" / f"coco-{split}-fb-{condition}__{llm}__{det}__default"


def coverage(vocab_emb, gt_emb, tau: float) -> np.ndarray:
    """Which ground-truth concepts are within ``tau`` of some vocabulary entry."""
    if not len(vocab_emb) or not len(gt_emb):
        return np.zeros(len(gt_emb), bool)
    return (gt_emb @ vocab_emb.T).max(axis=1) >= tau


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["tune", "test"], default="tune")
    ap.add_argument("--configs", nargs="+",
                    default=["qwen35-9b+yolo-world", "gemma4-e4b+yolo-world",
                             "qwen35-9b+owlv2-base"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir or f"results/feedback_{'dev' if args.split=='tune' else 'final'}")
    outdir.mkdir(parents=True, exist_ok=True)

    gt = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), args.split)
    gmap = {g.image_id: g for g in gt}
    emb = TextEmbedder(device=DEV)
    cev = CAAPEvaluator(gt, CAAP.all_thresholds)
    sev = SNAPEvaluator(gt, emb, grid=SNAP)
    lev = SemLocEvaluator(gt, emb)
    print(f"{len(gt)} images | {sum(len(g) for g in gt)} GT objects | split={args.split}\n")

    metric_rows, vocab_rows, recov_rows, conf_rows = [], [], [], []
    for cfgname in args.configs:
        llm, det = cfgname.split("+")
        for cond in CONDITIONS:
            d = run_dir(args.split, cond, llm, det)
            if not d.is_dir():
                print(f"  MISSING {d.name}", file=sys.stderr); continue
            cfg, recs = load_run(d)
            recs = [r for r in recs if r.image_id in gmap]
            preds = [r.to_predictions() for r in recs]
            caap = CAAP.summarise(cev.evaluate(preds, progress=False).per_threshold)
            snap = sev.evaluate(preds, control=False, progress=False).summary()
            sl = lev.evaluate(preds, progress=False)
            n_det = sum(len(p.scores) for p in preds) / max(len(preds), 1)
            metric_rows.append(dict(
                config=cfgname, llm=llm, detector=det, condition=cond,
                images=len(recs), det_per_img=round(n_det, 2),
                caap=round(caap["MACRO"], 4), snap=round(snap["MACRO"], 4),
                semloc=round(sl.macro, 4),
                vocab_per_img=round(float(np.mean([len(r.labels) for r in recs])), 2)))
            print(f"  {cfgname:<26}{cond:<18}CAAP {caap['MACRO']:.4f}  "
                  f"SNAP {snap['MACRO']:.4f}  SemLoc {sl.macro:.4f}  "
                  f"det/img {n_det:6.1f}", flush=True)

            if cond == "baseline":
                continue

            # ---- vocabulary + recovery diagnostics -----------------------
            trace = {}
            tp = d / "feedback.jsonl"
            if tp.is_file():
                for line in tp.open(encoding="utf-8"):
                    t = json.loads(line); trace[int(t["image_id"])] = t
            base = {r.image_id: r for r in
                    load_run(run_dir(args.split, "baseline", llm, det))[1]}
            d1_labels = {r.image_id: set(
                str(l).strip().lower() for l in r.pred_labels if str(l).strip())
                for r in recs}

            agg = defaultdict(float)
            cov_gain = {t: [0, 0] for t in TAU_GRID}      # [recovered, lost]
            n_img = 0
            removed_conf, kept_conf = [], []
            for r in recs:
                t = trace.get(r.image_id)
                if t is None:
                    continue
                n_img += 1
                c0, c1 = set(t["c0"]), set(t["c1"])
                added, removed = c1 - c0, c0 - c1
                agg["n_c0"] += len(c0); agg["n_c1"] += len(c1)
                agg["kept"] += len(c0 & c1); agg["removed"] += len(removed)
                agg["added"] += len(added); agg["renames"] += len(t["rename"])
                agg["parse_fail"] += int(not t["parse_ok"])
                agg["fallback1"] += int(t["fallback_level"] == 1)
                agg["noop"] += int(t["c0"] == t["c1"])
                agg["violations"] += len(t["violations"])
                agg["rejected"] += len(t["rejected"])
                agg["truncated"] += int(any("truncated" in v for v in t["violations"]))
                # zero-detection concepts and what happened to them
                b = base.get(r.image_id)
                if b is not None:
                    got = set(str(l).strip().lower() for l in b.pred_labels
                              if str(l).strip())
                    zero = {c for c in c0 if c not in got}
                    agg["zero_det"] += len(zero)
                    agg["zero_det_removed"] += len(zero - c1)
                    agg["zero_det_kept"] += len(zero & c1)
                    # did the model key on raw confidence rather than presence?
                    conf = {}
                    for l, s in zip(b.pred_labels, b.scores):
                        k = str(l).strip().lower()
                        if k:
                            conf[k] = max(conf.get(k, 0.0), float(s))
                    for c in c0:
                        (removed_conf if c in removed else kept_conf).append(
                            conf.get(c, 0.0))
                agg["added_detected"] += len(added & d1_labels.get(r.image_id, set()))
                # semantic coverage of the image's GT categories
                gtl = sorted({str(l).strip().lower()
                              for l in gmap[r.image_id].labels if str(l).strip()})
                if gtl and (c0 or c1):
                    ge = emb.encode(gtl)
                    e0 = emb.encode(sorted(c0)) if c0 else np.zeros((0, ge.shape[1]))
                    e1 = emb.encode(sorted(c1)) if c1 else np.zeros((0, ge.shape[1]))
                    for tau in TAU_GRID:
                        k0, k1 = coverage(e0, ge, tau), coverage(e1, ge, tau)
                        cov_gain[tau][0] += int((~k0 & k1).sum())
                        cov_gain[tau][1] += int((k0 & ~k1).sum())
            if not n_img:
                continue
            vocab_rows.append(dict(
                config=cfgname, condition=cond, images=n_img,
                c0_per_img=round(agg["n_c0"] / n_img, 2),
                c1_per_img=round(agg["n_c1"] / n_img, 2),
                kept_per_img=round(agg["kept"] / n_img, 2),
                removed_per_img=round(agg["removed"] / n_img, 2),
                added_per_img=round(agg["added"] / n_img, 2),
                renames_per_img=round(agg["renames"] / n_img, 2),
                zero_det_per_img=round(agg["zero_det"] / n_img, 2),
                pct_zero_det_removed=round(
                    100 * agg["zero_det_removed"] / max(agg["zero_det"], 1), 1),
                pct_added_detected=round(
                    100 * agg["added_detected"] / max(agg["added"], 1), 1),
                noop_pct=round(100 * agg["noop"] / n_img, 1),
                parse_fail=int(agg["parse_fail"]),
                fallback_recovered=int(agg["fallback1"]),
                violations=int(agg["violations"]), rejected=int(agg["rejected"]),
                truncated_images=int(agg["truncated"])))
            for tau in TAU_GRID:
                rec, lost = cov_gain[tau]
                recov_rows.append(dict(config=cfgname, condition=cond, tau=tau,
                                       recovered=rec, lost=lost, net=rec - lost))
            if removed_conf or kept_conf:
                conf_rows.append(dict(
                    config=cfgname, condition=cond,
                    n_removed=len(removed_conf), n_kept=len(kept_conf),
                    mean_maxconf_removed=round(float(np.mean(removed_conf)), 4)
                    if removed_conf else None,
                    mean_maxconf_kept=round(float(np.mean(kept_conf)), 4)
                    if kept_conf else None,
                    median_maxconf_removed=round(float(np.median(removed_conf)), 4)
                    if removed_conf else None,
                    median_maxconf_kept=round(float(np.median(kept_conf)), 4)
                    if kept_conf else None))

    def write(rows, name):
        if not rows:
            return
        with (outdir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {outdir / name}")
    write(metric_rows, "metrics.csv")
    write(vocab_rows, "vocabulary.csv")
    write(recov_rows, "recovery.csv")
    write(conf_rows, "confidence_usage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
