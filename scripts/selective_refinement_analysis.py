#!/usr/bin/env python
"""Robustness analysis for selective refinement. Entirely offline.

The analysis needs ~9,000 gate evaluations, each requiring 120 pooled average
precisions (10 CAAP + 10 SNAP + 100 SemLoc-AP). Recomputing those the ordinary
way would be hours, so this module evaluates them exactly but quickly:

* every detection from D0 and D1 is placed in **one globally score-sorted
  array**, tagged with its image and which run it came from. A gate is then a
  boolean mask over that array, and masking preserves the sort, so no gate ever
  re-sorts anything.
* 101-point AP needs, for each recall level, the best precision at or beyond it.
  Recall only changes at a true positive, so the whole curve is determined by
  the *ranks of the true positives*. Those are found by looking up the
  already-known TP rows of each threshold rather than scanning all 61k
  detections, which is what makes the sweep affordable.

The fast path is checked against the authoritative evaluators before use; if it
disagreed it would be worthless.
"""

from __future__ import annotations

import argparse, csv, json, os, sys
from pathlib import Path

sys.path.insert(0, ".")

from laod.device import pin_cuda_device

DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:0"))

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.predictions import cap_detections
from laod.io.run_store import load_run
from laod.metrics.common import box_iou_matrix, greedy_match
from laod.metrics.semloc import IOU_THRESHOLDS, SEM_THRESHOLDS, is_valid_label
from laod.metrics.snap import TextEmbedder

D0_RUN = "coco-tune-fb-baseline__qwen35-9b__yolo-world__default"
D1_RUN = "coco-tune-fb-feedback_full__qwen35-9b__yolo-world__default"
MAX_DETS = 100
SEED = 20261015
LEVELS = np.linspace(0.0, 1.0, 101)

CAAP_KEYS = [("caap", t) for t in IOU_THRESHOLDS]
SNAP_KEYS = [("snap", t) for t in SEM_THRESHOLDS]
SEMLOC_KEYS = [("semloc", (th, ta)) for th in IOU_THRESHOLDS for ta in SEM_THRESHOLDS]
ALL_KEYS = CAAP_KEYS + SNAP_KEYS + SEMLOC_KEYS


def build_tables(gt, p0, p1, emb, order):
    """Global score-sorted detection table with a flag column per threshold."""
    rows_score, rows_img, rows_run, rows_flags = [], [], [], []
    gt_by_id = {g.image_id: g for g in gt}
    for run_id, pmap in ((0, p0), (1, p1)):
        preds = cap_detections([pmap[i].to_predictions() for i in order], MAX_DETS)
        for p in preds:
            g = gt_by_id[p.image_id]
            iou = box_iou_matrix(p.boxes, g.boxes)
            if len(p.labels) and len(g.labels):
                valid = np.array([is_valid_label(l) for l in p.labels], bool)
                sim = np.full((len(p.labels), len(g.labels)), -np.inf)
                keep = np.flatnonzero(valid)
                if keep.size:
                    sim[keep] = (emb.encode([p.labels[i] for i in keep])
                                 @ emb.encode(list(g.labels)).T)
            else:
                sim = np.zeros((len(p.labels), len(g.labels)))
            cols = []
            for kind, key in ALL_KEYS:
                if kind == "caap":
                    m, thr = iou, key
                elif kind == "snap":
                    m, thr = sim, key
                else:
                    th, ta = key
                    m, thr = np.where(sim >= ta, iou, -np.inf), th
                cols.append(greedy_match(m, thr, scores=p.scores, order="score"))
            n = len(p.scores)
            rows_score.append(np.asarray(p.scores, float))
            rows_img.append(np.full(n, p.image_id))
            rows_run.append(np.full(n, run_id, np.int8))
            rows_flags.append(np.stack(cols, axis=1) if n else
                              np.zeros((0, len(ALL_KEYS)), bool))
    score = np.concatenate(rows_score)
    img = np.concatenate(rows_img)
    run = np.concatenate(rows_run)
    flags = np.concatenate(rows_flags, axis=0)
    o = np.argsort(-score, kind="stable")
    return score[o], img[o], run[o], flags[o]


class FastScorer:
    """Exact pooled AP for an arbitrary per-image choice between D0 and D1."""

    def __init__(self, score, img, run, flags, n_gt, order):
        self.run = run
        self.n_gt = n_gt
        self.flags = flags
        self.index = {i: k for k, i in enumerate(order)}
        self.img_idx = np.array([self.index[i] for i in img])
        # TP row positions per threshold column, computed once.
        self.tp_rows = [np.flatnonzero(flags[:, c]) for c in range(flags.shape[1])]
        self.targets = np.ceil(LEVELS * n_gt).astype(int)

    def score_gate(self, fired_mask: np.ndarray) -> np.ndarray:
        """``fired_mask``: bool per image, in ``order``. Returns AP per column."""
        want = fired_mask[self.img_idx].astype(np.int8)
        sel = want == self.run
        rank = np.cumsum(sel)                      # 1-indexed rank of selected rows
        out = np.empty(self.flags.shape[1])
        for c, rows in enumerate(self.tp_rows):
            keep = rows[sel[rows]]
            if keep.size == 0:
                out[c] = 0.0
                continue
            pos = rank[keep]                       # rank of each TP within the gate
            k = np.arange(1, pos.size + 1)
            env = np.maximum.accumulate((k / pos)[::-1])[::-1]
            t = self.targets
            ok = (t >= 1) & (t <= pos.size)
            vals = np.zeros(t.size)
            vals[ok] = env[t[ok] - 1]
            vals[t == 0] = env[0]
            out[c] = vals.mean()
        return out

    @staticmethod
    def summarise(ap: np.ndarray) -> tuple[float, float, float]:
        n1, n2 = len(CAAP_KEYS), len(CAAP_KEYS) + len(SNAP_KEYS)
        return (float(ap[:n1].mean()), float(ap[n1:n2].mean()), float(ap[n2:].mean()))


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--outdir", default="results/selective_refinement")
    ap_.add_argument("--n_random", type=int, default=1000)
    args = ap_.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    gt = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), "tune")
    n_gt = sum(len(g) for g in gt)
    order = [g.image_id for g in gt]
    _, r0 = load_run(PATHS.outputs / "runs" / D0_RUN)
    _, r1 = load_run(PATHS.outputs / "runs" / D1_RUN)
    p0 = {r.image_id: r for r in r0}
    p1 = {r.image_id: r for r in r1}
    emb = TextEmbedder(device=DEV)
    print(f"{len(order)} images | {n_gt} GT objects | building detection table...",
          flush=True)
    score, img, run, flags = build_tables(gt, p0, p1, emb, order)
    print(f"  {score.size:,} detections, {flags.shape[1]} threshold columns")
    sc = FastScorer(score, img, run, flags, n_gt, order)

    # ---- verify against the authoritative evaluators ----------------------
    none_ = np.zeros(len(order), bool); all_ = np.ones(len(order), bool)
    b = sc.summarise(sc.score_gate(none_))
    a = sc.summarise(sc.score_gate(all_))
    REF = {"Standard LAOD": (0.2809, 0.2957, 0.1712),
           "Always Refine": (0.2807, 0.3030, 0.1730)}
    print("\n=== fast path vs the authoritative evaluators ===")
    ok = True
    for nm, got in (("Standard LAOD", b), ("Always Refine", a)):
        exp = REF[nm]
        d = max(abs(g - e) for g, e in zip(got, exp))
        ok &= d < 5e-4
        print(f"  {nm:<16} got {got[0]:.4f}/{got[1]:.4f}/{got[2]:.4f}  "
              f"expected {exp[0]:.4f}/{exp[1]:.4f}/{exp[2]:.4f}  maxdiff {d:.2e}"
              f"  {'OK' if d < 5e-4 else 'MISMATCH'}")
    if not ok:
        raise SystemExit("fast scorer disagrees with the authoritative evaluators")

    stats = {}
    for i in order:
        s = np.sort(np.asarray(p0[i].scores, float))[::-1]
        stats[i] = {"s_max": float(s[0]) if s.size else 0.0,
                    "s_mean_top3": float(s[:3].mean()) if s.size else 0.0,
                    "s_mean_top5": float(s[:5].mean()) if s.size else 0.0}
    STATS = ("s_max", "s_mean_top3", "s_mean_top5")
    vals = {k: np.array([stats[i][k] for i in order]) for k in STATS}
    rank_lo = {k: np.argsort(vals[k], kind="stable") for k in STATS}  # ascending

    def gate_lowest(stat, rate):
        n = int(round(rate * len(order)))
        m = np.zeros(len(order), bool)
        m[rank_lo[stat][:n]] = True
        return m

    def gate_highest(stat, rate):
        n = int(round(rate * len(order)))
        m = np.zeros(len(order), bool)
        if n:
            m[rank_lo[stat][-n:]] = True
        return m

    results = {"baseline": b, "always": a}

    # ---- 1. random-gate control ------------------------------------------
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rng = np.random.default_rng(SEED)
    rand_rows, pct_rows = [], []
    print(f"\n=== 1. random-gate control ({args.n_random} seeds per rate, "
          f"base seed {SEED}) ===")
    rand_dist = {}
    for rate in rates:
        n = int(round(rate * len(order)))
        acc = np.empty((args.n_random, 3))
        for s in range(args.n_random):
            m = np.zeros(len(order), bool)
            m[rng.choice(len(order), size=n, replace=False)] = True
            acc[s] = sc.summarise(sc.score_gate(m))
        rand_dist[rate] = acc
        for j, mname in enumerate(("CAAP", "SNAP", "SemLoc")):
            v = acc[:, j]
            rand_rows.append(dict(rate=rate, metric=mname, mean=round(v.mean(), 5),
                                  std=round(v.std(ddof=1), 5),
                                  p2_5=round(np.percentile(v, 2.5), 5),
                                  p50=round(np.percentile(v, 50), 5),
                                  p97_5=round(np.percentile(v, 97.5), 5),
                                  min=round(v.min(), 5), max=round(v.max(), 5)))
        print(f"  rate {rate:.0%} done", flush=True)

    # empirical percentile of each confidence gate within its random distribution
    for stat in STATS:
        for rate in rates:
            g = sc.summarise(sc.score_gate(gate_lowest(stat, rate)))
            for j, mname in enumerate(("CAAP", "SNAP", "SemLoc")):
                v = rand_dist[rate][:, j]
                pct_rows.append(dict(stat=stat, rate=rate, metric=mname,
                                     gate=round(g[j], 5),
                                     random_mean=round(v.mean(), 5),
                                     ci_lo=round(np.percentile(v, 2.5), 5),
                                     ci_hi=round(np.percentile(v, 97.5), 5),
                                     percentile=round(100 * (v < g[j]).mean(), 1)))

    # ---- 2. low vs high complements --------------------------------------
    comp_rows = []
    for stat in STATS:
        for rate in rates:
            lo = sc.summarise(sc.score_gate(gate_lowest(stat, rate)))
            hi = sc.summarise(sc.score_gate(gate_highest(stat, rate)))
            rm = rand_dist[rate].mean(axis=0)
            comp_rows.append(dict(stat=stat, rate=rate,
                                  low_caap=round(lo[0], 5), low_snap=round(lo[1], 5),
                                  low_semloc=round(lo[2], 5),
                                  rand_caap=round(rm[0], 5), rand_snap=round(rm[1], 5),
                                  rand_semloc=round(rm[2], 5),
                                  high_caap=round(hi[0], 5), high_snap=round(hi[1], 5),
                                  high_semloc=round(hi[2], 5),
                                  ordering_holds=bool(lo[2] > rm[2] > hi[2])))

    # ---- 3/4. fine rate sweep --------------------------------------------
    sweep_rows = []
    for stat in STATS:
        for r in np.arange(0.0, 1.0001, 0.05):
            g = sc.summarise(sc.score_gate(gate_lowest(stat, float(r))))
            sweep_rows.append(dict(stat=stat, rate=round(float(r), 2),
                                   caap=round(g[0], 5), snap=round(g[1], 5),
                                   semloc=round(g[2], 5)))
    fine_rows = []
    for stat in ("s_max", "s_mean_top3"):
        for r in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
            g = sc.summarise(sc.score_gate(gate_lowest(stat, r)))
            fine_rows.append(dict(stat=stat, rate=r, caap=round(g[0], 5),
                                  snap=round(g[1], 5), semloc=round(g[2], 5)))

    # ---- 5. threshold-wise behaviour -------------------------------------
    thr_rows = []
    confs = {"Standard LAOD": none_, "Always Refine": all_,
             "max-conf p50": gate_lowest("s_max", 0.50),
             "top3-mean p50": gate_lowest("s_mean_top3", 0.50)}
    n1, n2 = len(CAAP_KEYS), len(CAAP_KEYS) + len(SNAP_KEYS)
    for nm, m in confs.items():
        apv = sc.score_gate(m)
        for t, v in zip(IOU_THRESHOLDS, apv[:n1]):
            thr_rows.append(dict(method=nm, metric="CAAP", threshold=t, ap=round(v, 5)))
        for t, v in zip(SEM_THRESHOLDS, apv[n1:n2]):
            thr_rows.append(dict(method=nm, metric="SNAP", threshold=t, ap=round(v, 5)))
        sl = apv[n2:].reshape(len(IOU_THRESHOLDS), len(SEM_THRESHOLDS))
        for t, v in zip(IOU_THRESHOLDS, sl.mean(axis=1)):
            thr_rows.append(dict(method=nm, metric="SemLoc_marg_IoU", threshold=t,
                                 ap=round(float(v), 5)))
        for t, v in zip(SEM_THRESHOLDS, sl.mean(axis=0)):
            thr_rows.append(dict(method=nm, metric="SemLoc_marg_sem", threshold=t,
                                 ap=round(float(v), 5)))

    def w(rows, name):
        with (out / name).open("w", newline="") as fh:
            ww = csv.DictWriter(fh, fieldnames=list(rows[0])); ww.writeheader(); ww.writerows(rows)
        print(f"wrote {out/name}")
    w(rand_rows, "random_control.csv"); w(pct_rows, "gate_percentiles.csv")
    w(comp_rows, "low_vs_high.csv"); w(sweep_rows, "rate_sweep.csv")
    w(fine_rows, "fine_sweep.csv"); w(thr_rows, "threshold_wise.csv")
    json.dump({"seed": SEED, "n_random": args.n_random,
               "baseline": b, "always": a,
               "thresholds": {s: {f"p{int(r*100)}": float(np.percentile(vals[s], r*100))
                                  for r in rates} for s in STATS}},
              (out / "analysis_meta.json").open("w"), indent=1)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
