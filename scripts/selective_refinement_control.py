"""Control: does a RANDOM gate at the same refinement rate do just as well?"""
import os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)
from laod.device import pin_cuda_device
DEV = pin_cuda_device(os.environ.get("LAOD_DEVICE", "cuda:0"))
import json, numpy as np
sys.path.insert(0, "scripts")
from selective_refinement import (D0_RUN, D1_RUN, per_image_flags, score_gate,
                                  gate_stats)
from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.snap import TextEmbedder

gt = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), "tune")
n_gt = sum(len(g) for g in gt); order = [g.image_id for g in gt]
_, r0 = load_run(PATHS.outputs / "runs" / D0_RUN)
_, r1 = load_run(PATHS.outputs / "runs" / D1_RUN)
p0 = {r.image_id: r for r in r0}; p1 = {r.image_id: r for r in r1}
emb = TextEmbedder(device=DEV)
F0 = per_image_flags(gt, [p0[i].to_predictions() for i in order], emb)
F1 = per_image_flags(gt, [p1[i].to_predictions() for i in order], emb)
stats = {i: gate_stats(np.asarray(p0[i].scores, float)) for i in order}
q = json.load(open("results/selective_refinement/quantiles.json"))

print(f"{'gate':<34}{'ref%':>7}{'CAAP':>9}{'SNAP':>9}{'SemLoc':>9}")
base = score_gate(order, {i: False for i in order}, F0, F1, n_gt)
print(f"{'Standard LAOD':<34}{0.0:>6.1f}%{base[0]:>9.4f}{base[1]:>9.4f}{base[2]:>9.4f}")
alw = score_gate(order, {i: True for i in order}, F0, F1, n_gt)
print(f"{'Always Refine':<34}{100.0:>6.1f}%{alw[0]:>9.4f}{alw[1]:>9.4f}{alw[2]:>9.4f}")
print()
for p in (30, 50):
    t = q["s_max"][str(p)]
    fired = {i: stats[i]["s_max"] < t for i in order}
    c = score_gate(order, fired, F0, F1, n_gt)
    print(f"{'CONFIDENCE max-conf p'+str(p):<34}{100*sum(fired.values())/500:>6.1f}%"
          f"{c[0]:>9.4f}{c[1]:>9.4f}{c[2]:>9.4f}")
    res = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        sel = set(rng.choice(order, size=int(round(p/100*500)), replace=False).tolist())
        res.append(score_gate(order, {i: i in sel for i in order}, F0, F1, n_gt))
    a = np.array([[r[0], r[1], r[2]] for r in res])
    print(f"{'  RANDOM same rate (10 seeds)':<34}{p:>6.1f}%"
          f"{a[:,0].mean():>9.4f}{a[:,1].mean():>9.4f}{a[:,2].mean():>9.4f}")
    print(f"{'  random min-max':<34}{'':>7}"
          f"{a[:,0].min():.4f}-{a[:,0].max():.4f} {a[:,1].min():.4f}-{a[:,1].max():.4f} "
          f"{a[:,2].min():.4f}-{a[:,2].max():.4f}")
    for k, nm in ((0, "CAAP"), (1, "SNAP"), (2, "SemLoc")):
        better = int((a[:, k] >= c[k]).sum())
        print(f"{'  random >= confidence gate on '+nm:<34}{better}/10")
    print()
