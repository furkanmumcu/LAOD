#!/usr/bin/env python
"""Three-configuration appendix analysis for selective refinement.

Entirely offline: D0 and D1 are cached for every configuration, so a method is
only a per-image choice between two stored results. All three configurations
use the **same 60% refinement budget** so the rows represent equal added
inference cost; no per-configuration rate is tuned.

Confidence scales are not comparable between YOLO-World and OWLv2, so the gate
is defined by the lowest-60% *ranking* within each configuration. The absolute
YOLO-World threshold is reported separately as a secondary diagnostic.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from laod.device import pin_cuda_device
DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:0"))
import numpy as np
from selective_refinement_analysis import (ALL_KEYS, CAAP_KEYS, SNAP_KEYS,
                                           FastScorer, build_tables)
from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP, SNAP
from laod.metrics.semloc import IOU_THRESHOLDS, SEM_THRESHOLDS, SemLocEvaluator
from laod.metrics.snap import SNAPEvaluator, TextEmbedder

OUT = Path("results/selective_refinement"); OUT.mkdir(parents=True, exist_ok=True)
CONFIGS = [("qwen35-9b", "yolo-world"), ("gemma4-e4b", "yolo-world"),
           ("qwen35-9b", "owlv2-base")]
RATE, SEED, N_RANDOM = 0.60, 20261015, 1000
ABS_THR = 0.9170           # YOLO-World absolute threshold, frozen earlier

gt = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), "tune")
n_gt = sum(len(g) for g in gt); order = [g.image_id for g in gt]
emb = TextEmbedder(device=DEV)
cev = CAAPEvaluator(gt, CAAP.all_thresholds)
sev = SNAPEvaluator(gt, emb, grid=SNAP)
lev = SemLocEvaluator(gt, emb)
n1, n2 = len(CAAP_KEYS), len(CAAP_KEYS) + len(SNAP_KEYS)

main_rows, ctrl_rows, curve_rows, diag_rows, abs_rows = [], [], [], [], []
maxdiff_all = 0.0

for llm, det in CONFIGS:
    name = f"{llm} + {det}"
    print(f"\n{'='*78}\n{name}\n{'='*78}", flush=True)
    d0 = f"coco-tune-fb-baseline__{llm}__{det}__default"
    d1 = f"coco-tune-fb-feedback_full__{llm}__{det}__default"
    _, r0 = load_run(PATHS.outputs / "runs" / d0)
    _, r1 = load_run(PATHS.outputs / "runs" / d1)
    p0 = {r.image_id: r for r in r0}; p1 = {r.image_id: r for r in r1}
    score, img, run, flags = build_tables(gt, p0, p1, emb, order)
    sc = FastScorer(score, img, run, flags, n_gt, order)
    none_ = np.zeros(len(order), bool); all_ = np.ones(len(order), bool)

    # verify the fast path against the authoritative evaluators, per config
    for tag, mask, pm in (("Standard LAOD", none_, p0), ("Always Refine", all_, p1)):
        fast = FastScorer.summarise(sc.score_gate(mask))
        preds = [pm[i].to_predictions() for i in order]
        auth = (CAAP.summarise(cev.evaluate(preds, progress=False).per_threshold)["MACRO"],
                sev.evaluate(preds, control=False, progress=False).summary()["MACRO"],
                lev.evaluate(preds, progress=False).macro)
        d = max(abs(a - b) for a, b in zip(fast, auth))
        maxdiff_all = max(maxdiff_all, d)
        print(f"  verify {tag:<15} fast {fast[0]:.4f}/{fast[1]:.4f}/{fast[2]:.4f}  "
              f"authoritative {auth[0]:.4f}/{auth[1]:.4f}/{auth[2]:.4f}  maxdiff {d:.1e}")
        if d >= 5e-4:
            raise SystemExit(f"fast path disagrees for {name}/{tag}")

    smax = np.array([max(p0[i].scores) if len(p0[i].scores) else 0.0 for i in order])
    rank = np.argsort(smax, kind="stable")          # ascending confidence
    k = int(round(RATE * len(order)))
    low = np.zeros(len(order), bool); low[rank[:k]] = True
    high = np.zeros(len(order), bool); high[rank[-k:]] = True

    base = FastScorer.summarise(sc.score_gate(none_))
    alw = FastScorer.summarise(sc.score_gate(all_))
    sel = FastScorer.summarise(sc.score_gate(low))
    hi = FastScorer.summarise(sc.score_gate(high))
    for meth, vals, pct, calls in (("Standard LAOD", base, 0.0, 1.00),
                                   ("Always Refine", alw, 100.0, 2.00),
                                   ("Selective Refinement", sel, 60.0, 1.60)):
        main_rows.append(dict(config=name, method=meth, refined_pct=pct,
                              mllm_calls=calls, ovod_passes=calls,
                              caap=round(vals[0], 4), snap=round(vals[1], 4),
                              semloc=round(vals[2], 4),
                              d_caap=round(vals[0] - base[0], 4),
                              d_snap=round(vals[1] - base[1], 4),
                              d_semloc=round(vals[2] - base[2], 4)))
    print(f"\n  {'method':<22}{'CAAP':>9}{'SNAP':>9}{'SemLoc':>9}")
    for m, v in (("Standard LAOD", base), ("Always Refine", alw),
                 ("Selective (low 60%)", sel), ("HIGH 60% (control)", hi)):
        print(f"  {m:<22}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>9.4f}")

    # ---- 1000-seed random control at the same 60% budget -----------------
    rng = np.random.default_rng(SEED)
    acc = np.empty((N_RANDOM, 3))
    for s in range(N_RANDOM):
        m = np.zeros(len(order), bool)
        m[rng.choice(len(order), size=k, replace=False)] = True
        acc[s] = FastScorer.summarise(sc.score_gate(m))
    for j, mn in enumerate(("CAAP", "SNAP", "SemLoc")):
        v = acc[:, j]
        ctrl_rows.append(dict(config=name, metric=mn,
                              low=round(sel[j], 5), high=round(hi[j], 5),
                              rand_mean=round(v.mean(), 5),
                              rand_std=round(v.std(ddof=1), 5),
                              rand_p2_5=round(np.percentile(v, 2.5), 5),
                              rand_p97_5=round(np.percentile(v, 97.5), 5),
                              low_percentile=round(100 * (v < sel[j]).mean(), 1),
                              ordering_low_rand_high=bool(sel[j] > v.mean() > hi[j])))
    print(f"\n  {'metric':<8}{'LOW':>9}{'rand mean':>11}{'HIGH':>9}"
          f"{'95% CI':>22}{'LOW pct':>9}  order")
    for r in ctrl_rows[-3:]:
        print(f"  {r['metric']:<8}{r['low']:>9.4f}{r['rand_mean']:>11.4f}{r['high']:>9.4f}"
              f"   [{r['rand_p2_5']:.4f}, {r['rand_p97_5']:.4f}]{r['low_percentile']:>8.1f}%"
              f"  {'LOW>RAND>HIGH' if r['ordering_low_rand_high'] else 'FAILS'}")

    # ---- rate curves ------------------------------------------------------
    for rt in np.arange(0.0, 1.0001, 0.10):
        kk = int(round(float(rt) * len(order)))
        lo_m = np.zeros(len(order), bool); lo_m[rank[:kk]] = True
        hi_m = np.zeros(len(order), bool)
        if kk: hi_m[rank[-kk:]] = True
        lo_v = FastScorer.summarise(sc.score_gate(lo_m))
        hi_v = FastScorer.summarise(sc.score_gate(hi_m))
        rnd = np.empty((50, 3)); rg = np.random.default_rng(SEED + 7)
        for s in range(50):
            m = np.zeros(len(order), bool)
            if kk: m[rg.choice(len(order), size=kk, replace=False)] = True
            rnd[s] = FastScorer.summarise(sc.score_gate(m))
        curve_rows.append(dict(config=name, rate=round(float(rt), 2),
                               low_caap=round(lo_v[0], 5), low_snap=round(lo_v[1], 5),
                               low_semloc=round(lo_v[2], 5),
                               rand_caap=round(rnd[:, 0].mean(), 5),
                               rand_snap=round(rnd[:, 1].mean(), 5),
                               rand_semloc=round(rnd[:, 2].mean(), 5),
                               high_caap=round(hi_v[0], 5), high_snap=round(hi_v[1], 5),
                               high_semloc=round(hi_v[2], 5)))

    # ---- absolute-threshold diagnostic, YOLO-World only ------------------
    if det == "yolo-world":
        m = smax < ABS_THR
        v = FastScorer.summarise(sc.score_gate(m))
        abs_rows.append(dict(config=name, threshold=ABS_THR,
                             fired=int(m.sum()), fired_pct=round(100 * m.mean(), 1),
                             caap=round(v[0], 4), snap=round(v[1], 4),
                             semloc=round(v[2], 4),
                             d_caap=round(v[0] - base[0], 4),
                             d_snap=round(v[1] - base[1], 4),
                             d_semloc=round(v[2] - base[2], 4)))

    # ---- help / hurt ------------------------------------------------------
    ci = ALL_KEYS.index(("semloc", (0.50, 0.60)))
    tp = {}
    for rid in (0, 1):
        selr = run == rid
        for i in order:
            tp[(rid, i)] = int(flags[selr & (img == i), ci].sum())
    out = np.array([tp[(1, i)] - tp[(0, i)] for i in order])
    grp = {"helped": out > 0, "hurt": out < 0, "unchanged": out == 0}
    for g, mask in grp.items():
        diag_rows.append(dict(config=name, group=g, n=int(mask.sum()),
                              mean_smax=round(float(smax[mask].mean()), 4) if mask.any() else None,
                              median_smax=round(float(np.median(smax[mask])), 4) if mask.any() else None))
    chg = out != 0
    corr_all = float(np.corrcoef(smax, out)[0, 1])
    corr_chg = (float(np.corrcoef(smax[chg], (out[chg] > 0).astype(float))[0, 1])
                if chg.sum() > 2 else float("nan"))
    diag_rows.append(dict(config=name, group="corr(s_max, dTP)", n=len(order),
                          mean_smax=round(corr_all, 4), median_smax=round(corr_chg, 4)))
    print(f"\n  helped {int((out>0).sum())}  hurt {int((out<0).sum())}  "
          f"unchanged {int((out==0).sum())}")
    print(f"  mean s_max  helped {smax[out>0].mean():.4f}  hurt {smax[out<0].mean():.4f}  "
          f"unchanged {smax[out==0].mean():.4f}")
    print(f"  corr(s_max, dTP) = {corr_all:+.3f} | corr(s_max, helped|changed) = {corr_chg:+.3f}")

def w(rows, fn):
    with (OUT / fn).open("w", newline="") as fh:
        ww = csv.DictWriter(fh, fieldnames=list(rows[0])); ww.writeheader(); ww.writerows(rows)
    print(f"wrote {OUT/fn}")
print()
w(main_rows, "appendix_main.csv"); w(ctrl_rows, "appendix_controls.csv")
w(curve_rows, "appendix_curves.csv"); w(diag_rows, "appendix_diagnostics.csv")
w(abs_rows, "appendix_abs_threshold.csv")
json.dump({"max_fastpath_discrepancy": maxdiff_all, "seed": SEED,
           "n_random": N_RANDOM, "rate": RATE}, (OUT / "appendix_meta.json").open("w"), indent=1)
print(f"\nmax fast-path discrepancy across all configs: {maxdiff_all:.2e}")
