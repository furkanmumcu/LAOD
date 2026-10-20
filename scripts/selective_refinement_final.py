#!/usr/bin/env python
"""Threshold-wise stability, expanded help/hurt diagnostics, oracle headroom."""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from laod.device import pin_cuda_device
DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:0"))
import numpy as np
from selective_refinement_analysis import (ALL_KEYS, CAAP_KEYS, D0_RUN, D1_RUN,
                                           SNAP_KEYS, FastScorer, build_tables)
from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.semloc import IOU_THRESHOLDS, SEM_THRESHOLDS
from laod.metrics.snap import TextEmbedder

OUT = Path("results/selective_refinement"); OUT.mkdir(parents=True, exist_ok=True)
BEST_RATE = 0.60

gt = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), "tune")
n_gt = sum(len(g) for g in gt); order = [g.image_id for g in gt]
gmap = {g.image_id: g for g in gt}
_, r0 = load_run(PATHS.outputs / "runs" / D0_RUN)
_, r1 = load_run(PATHS.outputs / "runs" / D1_RUN)
p0 = {r.image_id: r for r in r0}; p1 = {r.image_id: r for r in r1}
emb = TextEmbedder(device=DEV)
score, img, run, flags = build_tables(gt, p0, p1, emb, order)
sc = FastScorer(score, img, run, flags, n_gt, order)

smax = np.array([max(p0[i].scores) if len(p0[i].scores) else 0.0 for i in order])
thr = float(np.percentile(smax, BEST_RATE * 100))
gate = smax < thr
print(f"frozen gate: s_max < {thr:.4f}  -> fires on {gate.sum()}/{len(order)} "
      f"({100*gate.mean():.1f}%)\n")

none_ = np.zeros(len(order), bool); all_ = np.ones(len(order), bool)
n1, n2 = len(CAAP_KEYS), len(CAAP_KEYS) + len(SNAP_KEYS)
rows = []
print("=== 5. threshold-wise behaviour ===")
cur = {}
for nm, m in (("Standard LAOD", none_), ("Always Refine", all_),
              (f"s_max<{thr:.3f} ({BEST_RATE:.0%})", gate)):
    ap = sc.score_gate(m); cur[nm] = ap
    sl = ap[n2:].reshape(len(IOU_THRESHOLDS), len(SEM_THRESHOLDS))
    for t, v in zip(IOU_THRESHOLDS, ap[:n1]):
        rows.append(dict(method=nm, metric="CAAP", threshold=t, ap=round(float(v), 5)))
    for t, v in zip(SEM_THRESHOLDS, ap[n1:n2]):
        rows.append(dict(method=nm, metric="SNAP", threshold=t, ap=round(float(v), 5)))
    for t, v in zip(IOU_THRESHOLDS, sl.mean(axis=1)):
        rows.append(dict(method=nm, metric="SemLoc_marg_IoU", threshold=t, ap=round(float(v), 5)))
    for t, v in zip(SEM_THRESHOLDS, sl.mean(axis=0)):
        rows.append(dict(method=nm, metric="SemLoc_marg_sem", threshold=t, ap=round(float(v), 5)))
with (OUT / "threshold_wise.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

base, gat = cur["Standard LAOD"], cur[f"s_max<{thr:.3f} ({BEST_RATE:.0%})"]
for label, lo, hi, ths in (("CAAP", 0, n1, IOU_THRESHOLDS),
                           ("SNAP", n1, n2, SEM_THRESHOLDS)):
    d = gat[lo:hi] - base[lo:hi]
    print(f"  {label}: improved at {int((d > 0).sum())}/{len(ths)} thresholds; "
          f"delta range {d.min():+.4f} to {d.max():+.4f}")
    print("     " + "  ".join(f"{t:.2f}:{x:+.4f}" for t, x in zip(ths, d)))
slb = base[n2:].reshape(10, 10); slg = gat[n2:].reshape(10, 10)
di, ds = slg.mean(1) - slb.mean(1), slg.mean(0) - slb.mean(0)
print(f"  SemLoc marginal over IoU: improved at {int((di>0).sum())}/10")
print("     " + "  ".join(f"{t:.2f}:{x:+.4f}" for t, x in zip(IOU_THRESHOLDS, di)))
print(f"  SemLoc marginal over tau: improved at {int((ds>0).sum())}/10")
print("     " + "  ".join(f"{t:.2f}:{x:+.4f}" for t, x in zip(SEM_THRESHOLDS, ds)))

# ---- 6. expanded help/hurt diagnostics ---------------------------------
TH, TA = 0.50, 0.60
ci = ALL_KEYS.index(("semloc", (TH, TA)))
tp_by = {}
for run_id, pm in ((0, p0), (1, p1)):
    sel = run == run_id
    for i in order:
        rows_i = sel & (img == i)
        tp_by[(run_id, i)] = int(flags[rows_i, ci].sum())

trace = {}
tp_path = PATHS.outputs / "runs" / D1_RUN / "feedback.jsonl"
for line in tp_path.open(encoding="utf-8"):
    t = json.loads(line); trace[int(t["image_id"])] = t

diag = []
for k, i in enumerate(order):
    t = trace.get(i, {})
    c0, c1 = set(t.get("c0", [])), set(t.get("c1", []))
    s0 = np.sort(np.asarray(p0[i].scores, float))[::-1]
    s1 = np.sort(np.asarray(p1[i].scores, float))[::-1]
    d = tp_by[(1, i)] - tp_by[(0, i)]
    diag.append(dict(
        image_id=i, s_max=float(s0[0]) if s0.size else 0.0,
        d_n_vocab=len(c1) - len(c0), n_added=len(c1 - c0), n_removed=len(c0 - c1),
        n_retained=len(c0 & c1),
        d_n_det=len(s1) - len(s0),
        d_s_max=float((s1[0] if s1.size else 0) - (s0[0] if s0.size else 0)),
        d_top3=float(s1[:3].mean() - s0[:3].mean()) if s0.size and s1.size else 0.0,
        d_top5=float(s1[:5].mean() - s0[:5].mean()) if s0.size and s1.size else 0.0,
        tp0=tp_by[(0, i)], tp1=tp_by[(1, i)], d_tp=d,
        outcome="helped" if d > 0 else "hurt" if d < 0 else "unchanged",
        added=";".join(sorted(c1 - c0)), removed=";".join(sorted(c0 - c1))))
with (OUT / "per_image_expanded.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(diag[0])); w.writeheader(); w.writerows(diag)

print("\n=== 6. help / hurt / unchanged, expanded ===")
groups = {g: [d for d in diag if d["outcome"] == g]
          for g in ("helped", "hurt", "unchanged")}
cols = ("d_n_vocab", "n_added", "n_removed", "n_retained", "d_n_det",
        "d_s_max", "d_top3", "d_top5", "s_max")
print(f"{'group':<11}{'n':>5}" + "".join(f"{c:>11}" for c in cols))
for g, grp in groups.items():
    if not grp: continue
    print(f"{g:<11}{len(grp):>5}" + "".join(
        f"{np.mean([x[c] for x in grp]):>11.3f}" for c in cols))

# ---- 7. oracle ----------------------------------------------------------
oracle = np.array([d["d_tp"] > 0 for d in diag])
o = FastScorer.summarise(sc.score_gate(oracle))
b = FastScorer.summarise(base); g = FastScorer.summarise(gat)
print(f"\n=== 7. oracle headroom (diagnostic only; uses ground truth) ===")
print(f"{'':<22}{'refined%':>10}{'CAAP':>9}{'SNAP':>9}{'SemLoc':>9}")
print(f"{'Standard LAOD':<22}{0.0:>9.1f}%{b[0]:>9.4f}{b[1]:>9.4f}{b[2]:>9.4f}")
print(f"{'selective (frozen)':<22}{100*gate.mean():>9.1f}%{g[0]:>9.4f}{g[1]:>9.4f}{g[2]:>9.4f}")
print(f"{'ORACLE':<22}{100*oracle.mean():>9.1f}%{o[0]:>9.4f}{o[1]:>9.4f}{o[2]:>9.4f}")
print(f"\n{'':<22}{'captured of oracle gain':>26}")
for j, nm in enumerate(("CAAP", "SNAP", "SemLoc")):
    gain, ceil = g[j] - b[j], o[j] - b[j]
    print(f"  {nm:<20}{gain:+.4f} of {ceil:+.4f}  = {100*gain/ceil:>5.1f}%")
json.dump({"frozen_threshold": round(thr, 4), "rate": round(float(gate.mean()), 4),
           "baseline": b, "selective": g, "oracle": o,
           "oracle_refined_pct": round(100 * float(oracle.mean()), 1)},
          (OUT / "final_summary.json").open("w"), indent=1)
print(f"\nwrote {OUT}/threshold_wise.csv, per_image_expanded.csv, final_summary.json")
