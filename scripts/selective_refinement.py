#!/usr/bin/env python
"""Selective refinement: refine only when the first pass looks unreliable.

Always-refine did not pay for itself. The hypothesis here is that it might if
it were invoked selectively -- most images keep their one-pass output, and the
MLLM is consulted only when a detector-side signal suggests the first pass went
badly.

The experiment is entirely offline. D0 and D1 already exist for all 500 images
from the feedback experiment, so no MLLM call is made: a gate only *selects*
between two cached results.

The evaluation exploits the same idea one level down. All three metrics match
per image and then pool, so the per-image match flags are computed **once** for
D0 and once for D1; a gate is then a choice of which image's flags enter the
pooled ranking. Scoring twenty gates costs barely more than scoring one.

The gate sees only first-pass detector confidence. No ground truth, no IoU, no
metric value, no annotation count.
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
from laod.metrics.common import (average_precision, box_iou_matrix,
                                 greedy_match, LEGACY_EPS)
from laod.metrics.grids import CAAP, SNAP
from laod.metrics.semloc import IOU_THRESHOLDS, SEM_THRESHOLDS, is_valid_label
from laod.metrics.snap import TextEmbedder

D0_RUN = "coco-tune-fb-baseline__qwen35-9b__yolo-world__default"
D1_RUN = "coco-tune-fb-feedback_full__qwen35-9b__yolo-world__default"
MAX_DETS = 100
#: Confidence threshold used *inside* the count gate (N_tau). Unrelated to the
#: detector's 0.001 retention threshold, which only decides what is emitted.
COUNT_TAUS = (0.05, 0.10, 0.20, 0.30)


def gate_stats(scores: np.ndarray) -> dict:
    """First-pass confidence summary for one image. Detector-side only."""
    s = np.sort(np.asarray(scores, float))[::-1]
    n = int(s.size)
    out = {"n": n,
           "s_max": float(s[0]) if n else 0.0,
           "s_mean_top3": float(s[:3].mean()) if n else 0.0,
           "s_mean_top5": float(s[:5].mean()) if n else 0.0}
    for t in COUNT_TAUS:
        out[f"n_above_{t:g}"] = int((s >= t).sum())
    return out


def per_image_flags(gt, preds, emb):
    """Match flags for every threshold, for one prediction set.

    Returned once per run and reused by every gate, since a gate changes which
    image contributes which flags but never the flags themselves.
    """
    caap, snap, semloc, scores = {}, {}, {}, {}
    gt_by_id = {g.image_id: g for g in gt}
    for p in cap_detections(preds, MAX_DETS):
        g = gt_by_id[p.image_id]
        scores[p.image_id] = np.asarray(p.scores, float)
        iou = box_iou_matrix(p.boxes, g.boxes)
        caap[p.image_id] = {t: greedy_match(iou, t, scores=p.scores, order="score")
                            for t in IOU_THRESHOLDS}
        if len(p.labels) and len(g.labels):
            valid = np.array([is_valid_label(l) for l in p.labels], bool)
            sim = np.full((len(p.labels), len(g.labels)), -np.inf)
            keep = np.flatnonzero(valid)
            if keep.size:
                pe = emb.encode([p.labels[i] for i in keep])
                ge = emb.encode(list(g.labels))
                sim[keep] = pe @ ge.T
        else:
            sim = np.zeros((len(p.labels), len(g.labels)))
        snap[p.image_id] = {t: greedy_match(sim, t, scores=p.scores, order="score")
                            for t in SEM_THRESHOLDS}
        semloc[p.image_id] = {
            (th, ta): greedy_match(np.where(sim >= ta, iou, -np.inf), th,
                                   scores=p.scores, order="score")
            for th in IOU_THRESHOLDS for ta in SEM_THRESHOLDS}
    return {"scores": scores, "caap": caap, "snap": snap, "semloc": semloc}


def score_gate(order, fired, F0, F1, n_gt):
    """CAAP / SNAP / SemLoc-AP for one gate, from the cached flags."""
    pick = [(F1 if fired.get(i) else F0) for i in order]
    sc = np.concatenate([F["scores"][i] for F, i in zip(pick, order)])

    def ap_over(key, keys):
        out = {}
        for k in keys:
            fl = np.concatenate([F[key][i][k] for F, i in zip(pick, order)])
            out[k] = average_precision(sc, fl, n_gt, eps=LEGACY_EPS)
        return out

    caap = CAAP.summarise(ap_over("caap", IOU_THRESHOLDS))["MACRO"]
    snap = SNAP.summarise(ap_over("snap", SEM_THRESHOLDS))["MACRO"]
    pairs = [(th, ta) for th in IOU_THRESHOLDS for ta in SEM_THRESHOLDS]
    sl = float(np.mean(list(ap_over("semloc", pairs).values())))
    n_det = float(np.mean([len(F["scores"][i]) for F, i in zip(pick, order)]))
    return caap, snap, sl, n_det


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results/selective_refinement")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    gt = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), "tune")
    n_gt = sum(len(g) for g in gt)
    order = [g.image_id for g in gt]
    _, r0 = load_run(PATHS.outputs / "runs" / D0_RUN)
    _, r1 = load_run(PATHS.outputs / "runs" / D1_RUN)
    p0 = {r.image_id: r for r in r0}
    p1 = {r.image_id: r for r in r1}
    emb = TextEmbedder(device=DEV)
    print(f"{len(gt)} images | {n_gt} GT objects | D0={D0_RUN} | D1={D1_RUN}\n")

    print("computing per-image match flags (once per run)...", flush=True)
    F0 = per_image_flags(gt, [p0[i].to_predictions() for i in order], emb)
    F1 = per_image_flags(gt, [p1[i].to_predictions() for i in order], emb)

    # ---- gate statistics, from the UNCAPPED first pass --------------------
    stats = {i: gate_stats(np.asarray(p0[i].scores, float)) for i in order}
    with (out / "gate_stats.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id"] + list(stats[order[0]]))
        w.writeheader()
        for i in order:
            w.writerow({"image_id": i, **stats[i]})

    qs = (5, 10, 20, 30, 40, 50)
    print("\n=== first-pass confidence distribution (500 images) ===")
    print(f"{'statistic':<14}" + "".join(f"{f'p{q}':>9}" for q in qs))
    quant = {}
    for key in ("s_max", "s_mean_top3", "s_mean_top5"):
        v = np.array([stats[i][key] for i in order])
        quant[key] = {q: float(np.percentile(v, q)) for q in qs}
        print(f"{key:<14}" + "".join(f"{quant[key][q]:>9.4f}" for q in qs))
    n_zero = sum(1 for i in order if stats[i]["n"] == 0)
    print(f"\nimages with zero first-pass detections: {n_zero}")
    print("detections above a gate threshold, mean per image: " +
          "  ".join(f"tau={t:g}: {np.mean([stats[i][f'n_above_{t:g}'] for i in order]):.1f}"
                    for t in COUNT_TAUS))

    # ---- gate definitions -------------------------------------------------
    gates: list[tuple[str, str, dict]] = []
    gates.append(("Standard LAOD", "none", {i: False for i in order}))
    gates.append(("Always Refine", "all", {i: True for i in order}))
    gates.append(("No detections only", "A", {i: stats[i]["n"] == 0 for i in order}))
    for q in qs:
        t = quant["s_max"][q]
        gates.append((f"Max-conf < {t:.3f} (p{q})", "B",
                      {i: stats[i]["n"] == 0 or stats[i]["s_max"] < t for i in order}))
    for k in (3, 5):
        for q in qs:
            t = quant[f"s_mean_top{k}"][q]
            gates.append((f"Top-{k} mean < {t:.3f} (p{q})", "C",
                          {i: stats[i]["n"] == 0 or stats[i][f"s_mean_top{k}"] < t
                           for i in order}))
    for ct in COUNT_TAUS:
        for nmin in (1, 2, 3, 5):
            gates.append((f"N(s>={ct:g}) < {nmin}", "D",
                          {i: stats[i][f"n_above_{ct:g}"] < nmin for i in order}))
    rows = []
    print("\n=== gate sweep ===")
    print(f"{'method':<34}{'family':<8}{'ref%':>7}{'calls':>7}{'CAAP':>9}{'SNAP':>9}{'SemLoc':>9}{'det/img':>9}")
    for name, fam, fired in gates:
        nref = sum(fired.values())
        caap, snap, sl, nd = score_gate(order, fired, F0, F1, n_gt)
        rows.append(dict(method=name, family=fam, n_refined=nref,
                         refined_pct=round(100 * nref / len(order), 1),
                         calls_per_image=round(1 + nref / len(order), 3),
                         caap=round(caap, 4), snap=round(snap, 4),
                         semloc=round(sl, 4), det_per_img=round(nd, 1)))
        r = rows[-1]
        print(f"{name:<34}{fam:<8}{r['refined_pct']:>6.1f}%{r['calls_per_image']:>7.2f}"
              f"{caap:>9.4f}{snap:>9.4f}{sl:>9.4f}{nd:>9.1f}", flush=True)

    # ---- combined gates, built from whatever worked above ------------------
    best_c = max((r for r in rows if r["family"] == "C"), key=lambda r: r["semloc"])
    best_d = max((r for r in rows if r["family"] == "D"), key=lambda r: r["semloc"])
    t_mean = quant["s_mean_top5"][30]
    for ct, nmin in ((0.10, 2), (0.20, 3)):
        name = f"N==0 OR top5<{t_mean:.3f} OR N(s>={ct:g})<{nmin}"
        fired = {i: (stats[i]["n"] == 0 or stats[i]["s_mean_top5"] < t_mean
                     or stats[i][f"n_above_{ct:g}"] < nmin) for i in order}
        nref = sum(fired.values())
        caap, snap, sl, nd = score_gate(order, fired, F0, F1, n_gt)
        rows.append(dict(method=name, family="E", n_refined=nref,
                         refined_pct=round(100 * nref / len(order), 1),
                         calls_per_image=round(1 + nref / len(order), 3),
                         caap=round(caap, 4), snap=round(snap, 4),
                         semloc=round(sl, 4), det_per_img=round(nd, 1)))
        r = rows[-1]
        print(f"{name:<34}{'E':<8}{r['refined_pct']:>6.1f}%{r['calls_per_image']:>7.2f}"
              f"{caap:>9.4f}{snap:>9.4f}{sl:>9.4f}{nd:>9.1f}", flush=True)

    with (out / "gates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(quant, (out / "quantiles.json").open("w"), indent=1)
    print(f"\nwrote {out}/gates.csv, gate_stats.csv, quantiles.json")

    # ---- per-image diagnosis (retrospective; never used by a gate) --------
    TH, TA = 0.50, 0.60
    diag = []
    for i in order:
        tp0 = int(F0["semloc"][i][(TH, TA)].sum())
        tp1 = int(F1["semloc"][i][(TH, TA)].sum())
        c0 = int(F0["caap"][i][TH].sum()); c1 = int(F1["caap"][i][TH].sum())
        diag.append(dict(image_id=i, **stats[i],
                         n_det0=len(F0["scores"][i]), n_det1=len(F1["scores"][i]),
                         semloc_tp0=tp0, semloc_tp1=tp1, d_semloc_tp=tp1 - tp0,
                         caap_tp0=c0, caap_tp1=c1, d_caap_tp=c1 - c0,
                         outcome=("helped" if tp1 > tp0 else
                                  "hurt" if tp1 < tp0 else "unchanged")))
    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag[0])); w.writeheader(); w.writerows(diag)

    helped = [d for d in diag if d["outcome"] == "helped"]
    hurt = [d for d in diag if d["outcome"] == "hurt"]
    unch = [d for d in diag if d["outcome"] == "unchanged"]
    print(f"\n=== retrospective per-image outcome (joint TP at IoU {TH}, tau {TA}) ===")
    print(f"  helped {len(helped)}  hurt {len(hurt)}  unchanged {len(unch)}")
    print(f"\n{'group':<12}{'n':>5}{'s_max':>9}{'top3':>9}{'top5':>9}{'N':>8}{'dTP':>7}")
    for nm, grp in (("helped", helped), ("hurt", hurt), ("unchanged", unch)):
        if not grp: continue
        print(f"{nm:<12}{len(grp):>5}"
              f"{np.mean([g['s_max'] for g in grp]):>9.4f}"
              f"{np.mean([g['s_mean_top3'] for g in grp]):>9.4f}"
              f"{np.mean([g['s_mean_top5'] for g in grp]):>9.4f}"
              f"{np.mean([g['n'] for g in grp]):>8.1f}"
              f"{np.mean([g['d_semloc_tp'] for g in grp]):>+7.2f}")

    # ORACLE -- diagnostic upper bound only, never a method.
    oracle = {d["image_id"]: d["d_semloc_tp"] > 0 for d in diag}
    caap, snap, sl, nd = score_gate(order, oracle, F0, F1, n_gt)
    print(f"\n[ORACLE, not a method] refine iff it raises joint TP: "
          f"{100*sum(oracle.values())/len(order):.1f}% refined  "
          f"CAAP {caap:.4f}  SNAP {snap:.4f}  SemLoc {sl:.4f}")
    with (out / "oracle.json").open("w") as fh:
        json.dump({"refined_pct": round(100*sum(oracle.values())/len(order), 1),
                   "caap": round(caap, 4), "snap": round(snap, 4),
                   "semloc": round(sl, 4),
                   "note": "diagnostic upper bound; uses ground truth; NOT a method"}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
