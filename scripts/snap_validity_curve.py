#!/usr/bin/env python
"""SNAP against its label-shuffle control, swept over the similarity threshold.

Produces the evidence that SNAP's published low/mid thresholds carry no naming
information: a shuffled-label copy of the same predictions scores identically.
Also records the cosine distribution over *unrelated* ground-truth label pairs,
raw and mean-centred, which is the cause.

Offline pass over stored detections plus one CLIP text-encoder pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.data.splits import apply_split
from laod.io.run_store import load_run
from laod.metrics.snap import SNAPEvaluator, TextEmbedder, _unit

ANN = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}


def pair_stats(emb: np.ndarray) -> dict:
    """Cosine similarity between every distinct pair of label embeddings."""
    sim = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    v = sim[iu]
    return {"n_pairs": int(v.size), "min": float(v.min()),
            "median": float(np.median(v)), "max": float(v.max()),
            "mean": float(v.mean()),
            "hist_edges": np.linspace(-1, 1, 81).tolist(),
            "hist_counts": np.histogram(v, bins=np.linspace(-1, 1, 81))[0].tolist(),
            "admits": {f"{t:.2f}": float((v >= t).mean())
                       for t in np.arange(0.05, 1.0, 0.05)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="outputs/runs/coco-test__qwen35-9b__yolo-world__original",
                    help="run directory to sweep; defaults to the strongest COCO cell")
    ap.add_argument("--dataset", choices=sorted(ANN), default="coco")
    ap.add_argument("--out", default="results/snap_validity.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau_min", type=float, default=0.05)
    ap.add_argument("--tau_max", type=float, default=0.95)
    ap.add_argument("--tau_step", type=float, default=0.05)
    args = ap.parse_args()

    gt = apply_split(load_ground_truth(getattr(PATHS, ANN[args.dataset]),
                                       args.dataset), "test")
    cfg, recs = load_run(args.run)
    preds = [r.to_predictions() for r in recs]
    taus = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    print(f"{cfg.llm} + {cfg.detector} | {len(recs)} images | "
          f"{sum(len(g) for g in gt)} objects | {len(taus)} thresholds")

    emb = TextEmbedder(device=args.device)
    out = {"run": os.path.basename(args.run), "llm": cfg.llm,
           "detector": cfg.detector, "dataset": args.dataset,
           "images": len(recs), "thresholds": taus.tolist(), "variants": {}}

    for center in (False, True):
        ev = SNAPEvaluator(gt, emb, thresholds=taus, center=center)
        res = ev.evaluate(preds, control=True, progress=False)
        name = "centred" if center else "raw"
        out["variants"][name] = {
            "snap": [res.per_threshold[round(t, 4)] for t in taus],
            "chance": [res.chance.get(round(t, 4), float("nan")) for t in taus],
        }
        gains = [s - c for s, c in zip(out["variants"][name]["snap"],
                                       out["variants"][name]["chance"])]
        best = int(np.argmax(gains))
        out["variants"][name]["best_tau"] = float(taus[best])
        out["variants"][name]["best_gain"] = float(gains[best])
        print(f"  {name:8s} best gain {gains[best]:+.4f} at tau {taus[best]:.2f} "
              f"(SNAP {out['variants'][name]['snap'][best]:.4f} vs "
              f"chance {out['variants'][name]['chance'][best]:.4f})")

    # Why: the cosine floor between unrelated ground-truth labels.
    vocab = sorted({l for g in gt for l in g.labels})
    raw = emb.encode_raw(vocab)
    out["label_pairs"] = {
        "vocab_size": len(vocab),
        "raw": pair_stats(_unit(raw)),
        "centred": pair_stats(_unit(raw - raw.mean(0, keepdims=True))),
    }
    for k in ("raw", "centred"):
        s = out["label_pairs"][k]
        print(f"  {k:8s} pairs: min {s['min']:+.3f} median {s['median']:+.3f} "
              f"max {s['max']:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
