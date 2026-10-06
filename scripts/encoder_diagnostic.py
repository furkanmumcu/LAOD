#!/usr/bin/env python
"""Where does an encoder put *unrelated* labels on the cosine scale?

SNAP matches a predicted label to a ground-truth label when their cosine
similarity clears a threshold. That only works if unrelated labels score low.
CLIP's text embeddings sit in a narrow cone, so they do not -- which is why a
threshold of 0.50 admits everything and SNAP_LO is reproducible by shuffling
the labels.

This measures the floor directly, before any SNAP is computed: embed a
vocabulary, take every distinct pair, and report the distribution plus the
fraction each candidate threshold would admit. Cheap enough to screen several
encoders at once.
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
from laod.metrics.snap import TextEmbedder, _unit

#: (label, backend, model, template)
ENCODERS = [
    ("clip-raw", "clip", "ViT-B/32", "a photo of {}"),
    ("clip-centred", "clip", "ViT-B/32", "a photo of {}"),
    ("minilm", "sbert", "sentence-transformers/all-MiniLM-L6-v2", "{}"),
    ("minilm-photo", "sbert", "sentence-transformers/all-MiniLM-L6-v2", "a photo of {}"),
    ("mpnet", "sbert", "sentence-transformers/all-mpnet-base-v2", "{}"),
    ("mpnet-photo", "sbert", "sentence-transformers/all-mpnet-base-v2", "a photo of {}"),
]
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]


def stats(emb: np.ndarray) -> dict:
    sim = emb @ emb.T
    v = sim[np.triu_indices(len(emb), k=1)]
    return {"n_pairs": int(v.size), "min": float(v.min()),
            "p05": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max()),
            "admits": {f"{t:.2f}": float((v >= t).mean()) for t in TAUS}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/encoder_diagnostic.json")
    args = ap.parse_args()

    ann = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}
    gt = load_ground_truth(getattr(PATHS, ann[args.dataset]), args.dataset)
    vocab = sorted({l for g in gt for l in g.labels})
    print(f"{args.dataset}: {len(vocab)} ground-truth labels, "
          f"{len(vocab)*(len(vocab)-1)//2:,} distinct pairs\n")

    out = {"dataset": args.dataset, "vocab_size": len(vocab), "encoders": {}}
    hdr = f"{'encoder':<14}{'min':>8}{'p05':>8}{'median':>9}{'p95':>8}{'max':>8}   "
    print(hdr + "  ".join(f"@{t:.2f}" for t in TAUS))
    for name, backend, model, template in ENCODERS:
        try:
            emb = TextEmbedder(model, backend=backend, device=args.device,
                               template=template)
            raw = emb.encode_raw(vocab)
        except Exception as exc:
            print(f"{name:<14} FAILED  {type(exc).__name__}: {str(exc)[:60]}")
            continue
        e = _unit(raw - raw.mean(0, keepdims=True)) if name.endswith("centred") \
            else _unit(raw)
        st = stats(e)
        out["encoders"][name] = dict(st, backend=backend, model=model,
                                     template=template)
        print(f"{name:<14}{st['min']:>+8.3f}{st['p05']:>+8.3f}{st['median']:>+9.3f}"
              f"{st['p95']:>+8.3f}{st['max']:>+8.3f}   "
              + "  ".join(f"{st['admits'][f'{t:.2f}']*100:4.0f}%" for t in TAUS),
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("\n'admits' = fraction of UNRELATED pairs a threshold would accept. "
          "A usable threshold needs this near 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
