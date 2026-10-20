#!/usr/bin/env python
"""Score the fixed-ontology reference with the authoritative evaluators.

Every image is given the complete benchmark ontology, so the detector never has
to discover what to look for -- but it must still decide which of those
categories are present, enumerate instances, localise them, rank them and
suppress duplicates. This is a reference for how much the autonomous
vocabulary-discovery requirement costs, **not** a ceiling.

maxDets = 100 is applied once by each evaluator, after all Grounding DINO text
chunks have been merged -- never per chunk.
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
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP, SNAP
from laod.metrics.semloc import IOU_THRESHOLDS, SEM_THRESHOLDS, SemLocEvaluator
from laod.metrics.snap import SNAPEvaluator, TextEmbedder

DETS = ["yolo-world", "gdino-base", "gdino-tiny", "owlv2-base"]
LABEL = {"yolo-world": "YOLO-World", "gdino-base": "Grounding DINO-B",
         "gdino-tiny": "Grounding DINO-T", "owlv2-base": "OWLv2"}
ANN = {"coco": "coco_ann", "lvis": "lvis_ann"}
VOCAB = {"coco": ("fixed80", "COCO-80"), "lvis": ("fixed1203", "LVIS-1203")}
#: Autonomous LAOD detector averages from the main experiments, for comparison.
AUTO = {
 "coco": {"yolo-world": (0.2882, 0.2899, 0.1779), "gdino-base": (0.2634, 0.2097, 0.1270),
          "gdino-tiny": (0.2522, 0.2179, 0.1342), "owlv2-base": (0.2129, 0.2159, 0.1249)},
 "lvis": {"yolo-world": (0.1645, 0.1687, 0.0913), "gdino-base": (0.1215, 0.1272, 0.0689),
          "gdino-tiny": (0.1451, 0.1408, 0.0797), "owlv2-base": (0.1963, 0.2159, 0.1172)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results/fixed_ontology")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    emb = TextEmbedder(device=DEV)
    meta, rows, cmp_rows = {}, [], []

    for ds in ("coco", "lvis"):
        gt = apply_split(load_ground_truth(getattr(PATHS, ANN[ds]), ds), "test")
        n_gt = sum(len(g) for g in gt)
        cev = CAAPEvaluator(gt, CAAP.all_thresholds)
        sev = SNAPEvaluator(gt, emb, grid=SNAP)
        lev = SemLocEvaluator(gt, emb)
        tag, vname = VOCAB[ds]
        print(f"\n=== {ds.upper()} | {len(gt)} images | {n_gt:,} GT objects | "
              f"{vname} ===", flush=True)
        print(f"{'detector':<20}{'CAAP':>9}{'SNAP':>9}{'SemLoc':>9}"
              f"{'det/img pre':>13}{'post':>8}{'chunks':>8}")
        for det in DETS:
            d = PATHS.outputs / "runs" / f"{ds}-test-{tag}__none__{det}__cls{'80' if ds=='coco' else '1203'}"
            if not d.is_dir():
                print(f"  MISSING {d.name}", file=sys.stderr); continue
            cfg, recs = load_run(d)
            preds = [r.to_predictions() for r in recs]
            # Uncapped count: the true number emitted, recorded at run time when
            # storage was bounded. Falls back to what is on disk otherwise.
            pre = float(np.mean([(r.parse_flags or {}).get("n_detections_uncapped",
                                                           len(r.scores)) for r in recs]))
            post = float(np.mean([len(p.scores) for p in cap_detections(preds, 100)]))
            caap = CAAP.summarise(cev.evaluate(preds, progress=False).per_threshold)
            snap = sev.evaluate(preds, control=False, progress=False).summary()
            sl = lev.evaluate(preds, progress=False)
            nchunk = int(cfg.notes.split("text_passes=")[1].split(";")[0]) \
                if "text_passes=" in cfg.notes else 1
            rows.append(dict(dataset=ds, detector=LABEL[det], vocabulary=vname,
                             vocab_size=80 if ds == "coco" else 1203,
                             images=len(recs), gt_objects=n_gt,
                             caap=round(caap["MACRO"], 4), snap=round(snap["MACRO"], 4),
                             semloc=round(sl.macro, 4),
                             det_per_img_pre=round(pre, 1), det_per_img_post=round(post, 1),
                             chunks=nchunk,
                             caap_lo=round(caap["LO"], 4), caap_mi=round(caap["MI"], 4),
                             caap_hi=round(caap["HI"], 4),
                             snap_lo=round(snap["LO"], 4), snap_mi=round(snap["MI"], 4),
                             snap_hi=round(snap["HI"], 4)))
            a = AUTO[ds][det]
            cmp_rows.append(dict(dataset=ds, detector=LABEL[det],
                                 auto_caap=a[0], fixed_caap=round(caap["MACRO"], 4),
                                 d_caap=round(caap["MACRO"] - a[0], 4),
                                 auto_snap=a[1], fixed_snap=round(snap["MACRO"], 4),
                                 d_snap=round(snap["MACRO"] - a[1], 4),
                                 auto_semloc=a[2], fixed_semloc=round(sl.macro, 4),
                                 d_semloc=round(sl.macro - a[2], 4)))
            r = rows[-1]
            print(f"{LABEL[det]:<20}{r['caap']:>9.4f}{r['snap']:>9.4f}{r['semloc']:>9.4f}"
                  f"{r['det_per_img_pre']:>13.1f}{r['det_per_img_post']:>8.1f}"
                  f"{nchunk:>8}", flush=True)
            # SemLoc marginals, cheap and useful for appendix consistency
            m = sl.matrix()
            rows[-1]["semloc_marg_iou"] = json.dumps(
                [round(float(x), 4) for x in m.mean(axis=1)])
            rows[-1]["semloc_marg_sem"] = json.dumps(
                [round(float(x), 4) for x in m.mean(axis=0)])
        meta[ds] = dict(split="test", images=len(gt), gt_objects=n_gt,
                        vocabulary=vname, vocab_size=80 if ds == "coco" else 1203,
                        vocab_source=f"{ds} annotation 'categories', sorted by id",
                        normalisation="laod.data.annotations.normalise_category "
                                      "(LVIS underscores -> spaces; COCO unchanged)")

    def w(rs, fn):
        with (out / fn).open("w", newline="") as fh:
            ww = csv.DictWriter(fh, fieldnames=list(rs[0])); ww.writeheader(); ww.writerows(rs)
        print(f"wrote {out/fn}")
    print()
    w([r for r in rows if r["dataset"] == "coco"], "coco_results.csv")
    w([r for r in rows if r["dataset"] == "lvis"], "lvis_results.csv")
    w(cmp_rows, "comparison_to_autonomous.csv")
    meta.update(max_dets=100, detector_thresholds={
        "yolo-world": {"conf": 0.001}, "owlv2-base": {"score_threshold": 0.01},
        "gdino-tiny": {"score_threshold": 0.03, "text_threshold": 0.3},
        "gdino-base": {"score_threshold": 0.03, "text_threshold": 0.3}},
        evaluators="CAAP / SNAP (SBERT all-MiniLM-L6-v2, bare labels) / "
                   "SemLoc-AP 10x10 Cartesian, all at maxDets=100",
        note="maxDets applied once after merging all Grounding DINO text chunks")
    json.dump(meta, (out / "meta.json").open("w"), indent=1)
    print(f"wrote {out/'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
