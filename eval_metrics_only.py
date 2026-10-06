#!/usr/bin/env python3
"""Score saved predictions with CAAP and SNAP -- no model inference required.

This is the reproducibility entry point: anyone holding detection outputs can
reproduce or contest the published numbers without re-running an LLM.

Examples:
    # reproduce the original paper's COCO-Val row from the archived dumps
    python eval_metrics_only.py \\
        --predictions_file reference/original_runs/coco_ours_results \\
        --legacy_npy --snap_grid legacy --caap_grid legacy

    # score your own predictions with the calibrated metric
    python eval_metrics_only.py \\
        --predictions_file outputs/preds.json \\
        --annotation_file datasets/annotations/instances_val2017.json \\
        --snap_grid v2 --center
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from laod.device import pin_from_argv

_PINNED = pin_from_argv()

from laod.config import PATHS
from laod.data.annotations import load_ground_truth
from laod.io.predictions import load_legacy_npy, load_predictions
from laod.metrics.caap import CAAPEvaluator
from laod.metrics.grids import CAAP_GRIDS, SNAP_GRIDS, SNAP_V2_TARGET_FMR

logger = logging.getLogger("laod.eval")

ANNOTATION_DEFAULTS = {
    "coco": lambda: PATHS.coco_ann,
    "lvis": lambda: PATHS.lvis_ann,
    "coco_ood": lambda: PATHS.coco_ood_ann,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group("inputs")
    src.add_argument("--predictions_file", required=True, type=Path,
                     help="predictions JSON, or a directory of legacy .npy dumps")
    src.add_argument("--legacy_npy", action="store_true",
                     help="read predictions_file as an all_gt/all_dt .npy pair")
    src.add_argument("--annotation_file", type=Path, default=None,
                     help="COCO-format annotations; defaults to the dataset's file")
    src.add_argument("--dataset", choices=sorted(ANNOTATION_DEFAULTS), default="coco")
    src.add_argument("--split_size", type=int, default=None,
                     help="evaluate only the first N images (pilot subsets)")

    m = p.add_argument_group("metrics")
    m.add_argument("--caap_grid", choices=sorted(CAAP_GRIDS), default="legacy")
    m.add_argument("--snap_grid", choices=sorted(SNAP_GRIDS), default="legacy")
    m.add_argument("--no_caap", action="store_true")
    m.add_argument("--no_snap", action="store_true")
    m.add_argument("--match_order", choices=["given", "score"], default="given",
                   help="'given' reproduces the original; 'score' follows the "
                        "written metric definition")

    c = p.add_argument_group("SNAP / CLIP")
    c.add_argument("--clip_model", default="ViT-B/32")
    c.add_argument("--clip_backend", choices=["clip", "hf"], default="clip")
    c.add_argument("--raw_text", action="store_true",
                   help="embed bare labels instead of the prompt template")
    c.add_argument("--prompt_template", default="a photo of {}")
    c.add_argument("--center", action="store_true",
                   help="mean-centre embeddings (removes the CLIP cone; "
                        "strongly recommended outside reproduction runs)")
    c.add_argument("--calibrate", action="store_true",
                   help="derive SNAP thresholds from the ground-truth null "
                        "distribution at 5%%/1%%/0.1%% false-match rate")
    c.add_argument("--no_control", action="store_true",
                   help="skip the label-shuffled chance baseline")
    c.add_argument("--device", default="cuda:1")

    o = p.add_argument_group("output")
    o.add_argument("--output", type=Path, default=None, help="write results as JSON")
    o.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if _PINNED:
        args.device = _PINNED
    if args.no_caap and args.no_snap:
        logger.error("nothing to do: both metrics disabled")
        return 2

    if args.legacy_npy:
        ann = args.annotation_file or ANNOTATION_DEFAULTS[args.dataset]()
        names = {c["id"]: c["name"]
                 for c in json.loads(Path(ann).read_text())["categories"]}
        gt, preds = load_legacy_npy(args.predictions_file, category_names=names)
    else:
        preds, meta = load_predictions(args.predictions_file)
        ann = args.annotation_file or ANNOTATION_DEFAULTS[args.dataset]()
        gt = load_ground_truth(ann, args.dataset)
        if meta:
            logger.info("prediction metadata: %s", meta)

    if args.split_size is not None:
        keep = {g.image_id for g in gt[: args.split_size]}
        gt = [g for g in gt if g.image_id in keep]
        preds = [p for p in preds if p.image_id in keep]
        logger.info("restricted to %d images", len(gt))

    results: dict[str, object] = {
        "predictions_file": str(args.predictions_file),
        "dataset": args.dataset,
        "n_images": len(gt),
    }

    if not args.no_caap:
        grid = CAAP_GRIDS[args.caap_grid]
        res = CAAPEvaluator(gt, grid.all_thresholds,
                            match_order=args.match_order).evaluate(preds)
        print("\n" + res.report())
        summary = grid.summarise(res.per_threshold)
        print(f"  [{grid.name}] LO {summary['LO']:.4f}  MI {summary['MI']:.4f}  "
              f"HI {summary['HI']:.4f}  MACRO {summary['MACRO']:.4f}")
        results["caap"] = {"grid": grid.name,
                           "per_threshold": {str(k): v for k, v in res.per_threshold.items()},
                           **summary}

    if not args.no_snap:
        from laod.metrics.snap import SNAPEvaluator, TextEmbedder, calibrate_thresholds

        embedder = TextEmbedder(
            args.clip_model, backend=args.clip_backend, device=args.device,
            template=None if args.raw_text else args.prompt_template,
        )
        grid = SNAP_GRIDS[args.snap_grid]
        thresholds = None
        if args.calibrate:
            vocab = sorted({l for g in gt for l in g.labels})
            derived = calibrate_thresholds(embedder, vocab, SNAP_V2_TARGET_FMR,
                                           center=args.center)
            thresholds = sorted(derived.values())
            print("\ncalibrated thresholds (false-match rate -> cosine):")
            for name, tau in derived.items():
                print(f"  {name} @ FMR {SNAP_V2_TARGET_FMR[name]:.3%} -> tau {tau:.4f}")
            results["snap_calibration"] = derived

        ev = SNAPEvaluator(gt, embedder, thresholds,
                           grid=None if thresholds else grid,
                           center=args.center, match_order=args.match_order)
        res = ev.evaluate(preds, control=not args.no_control)
        print("\n" + res.report())
        entry: dict[str, object] = {
            "centered": args.center,
            "clip_model": args.clip_model,
            "clip_backend": args.clip_backend,
            "template": "raw" if args.raw_text else args.prompt_template,
            "per_threshold": {str(k): v for k, v in res.per_threshold.items()},
            "chance": {str(k): v for k, v in res.chance.items()},
        }
        if thresholds is None:
            entry["grid"] = grid.name
            entry.update(res.summary())
        results["snap"] = entry

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
