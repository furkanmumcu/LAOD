#!/usr/bin/env python3
"""End-to-end LAOD v2 benchmark runner.

Runs in two stages, because the expensive half and the cheap half depend on
different things.

**Stage 1 - labels.** Each LLM looks at each image once and names what it sees.
The result depends only on ``(image, llm, prompt)``, never on the detector, so
it is cached to disk. This is ~95% of the compute.

**Stage 2 - detect.** Each detector localises the cached labels. Because labels
are fixed, a detector swap is *exactly* controlled -- every backend sees the
identical vocabulary rather than a fresh stochastic sample -- and the grid
becomes additive: 7 LLM passes feed 28 detector passes instead of 28 full runs.

Both stages resume: interrupting and re-running picks up at the first image that
has no record yet.

Examples:
    # one cell, 200-image smoke test
    python run_benchmark.py --dataset coco --llm gemma3-4b \\
        --detector yolo-world --split_size 200 --evaluate

    # stage 1 of the headline table: all seven LLMs over COCO-Val
    python run_benchmark.py --dataset coco --llm all --stage labels

    # stage 2: every detector over the cached labels
    python run_benchmark.py --dataset coco --llm all --detector all --stage detect
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from laod.device import pin_from_argv

# Must happen before anything imports torch: see laod/device.py.
_PINNED = pin_from_argv()

from tqdm.auto import tqdm

from laod.config import PATHS
from laod.data import coco_loader, coco_ood_loader, lvis_loader
from laod.data.splits import (ABLATION_LVIS_PATH, apply_split,
                              load_ablation_subset)
from laod.data.base import DetectionImageDataset, build_dataloader
from laod.io.label_cache import LabelCache
from laod.io.run_store import ImageRecord, RunConfig, RunStore, file_digest
from laod.models.label_parser import parse_labels
from laod.models.registry import (ACTIVE_LLMS, DETECTOR_PRESETS, DETECTORS,
                                   LLMS, PROMPTS, DEFAULT_PROMPT,
                                   detector_params)

logger = logging.getLogger("laod.run")

LOADERS = {"coco": coco_loader, "lvis": lvis_loader, "coco_ood": coco_ood_loader}
ANNOTATIONS = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}


def expand(values: list[str], universe: dict) -> list[str]:
    """Resolve ``all`` and validate keys against a registry."""
    if values == ["all"]:
        return list(universe)
    unknown = [v for v in values if v not in universe]
    if unknown:
        raise SystemExit(f"unknown key(s) {unknown}; available: {sorted(universe)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(LOADERS), default="coco")
    p.add_argument("--llm", nargs="+", default=["gemma3-4b"],
                   help="registry keys, or 'all'")
    p.add_argument("--detector", nargs="+", default=["yolo-world"],
                   help="registry keys, or 'all'")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS))
    p.add_argument("--split_size", type=int, default=None,
                   help="first N images; omit for the full split")
    p.add_argument("--subset", choices=["ablation"], default=None,
                   help="further restrict to a frozen subset of the chosen split; "
                        "label caches stay keyed by split, so labels already "
                        "generated for the full split are reused")
    p.add_argument("--split", choices=["test", "tune", "all"], default="test",
                   help="'test' excludes the frozen hyperparameter-selection "
                        "holdout and is what reported numbers must use; 'tune' "
                        "is the holdout itself; 'all' ignores the split and is "
                        "only for reproducing the original full-5k numbers")
    p.add_argument("--stage", choices=["labels", "detect", "all"], default="all")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--parse_mode", choices=["strict", "legacy"], default="strict")
    p.add_argument("--detector_preset", choices=DETECTOR_PRESETS, default="v2",
                   help="'legacy' reproduces the original's inherited thresholds; "
                        "'v2' uses the measured per-detector CAAP optimum")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--do_sample", action="store_true",
                   help="restore the original's sampling; makes runs unreproducible")
    p.add_argument("--device", default="cuda:1",
                   help="single device for LLM, detector and CLIP")
    p.add_argument("--output_root", type=Path, default=None)
    p.add_argument("--save_preds", type=Path, default=None,
                   help="also write metric-ready predictions JSON here")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--evaluate", action="store_true",
                   help="score each completed run with CAAP and SNAP")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def stage_labels(args, gt, dataset, llms: list[str]) -> dict[str, LabelCache]:
    """Generate and cache one label list per (image, llm, prompt)."""
    prompt = PROMPTS[args.prompt]
    caches: dict[str, LabelCache] = {}
    ids = [g.image_id for g in gt]

    for key in llms:
        cache = LabelCache(PATHS.label_cache, f"{args.dataset}_{args.split}", key, args.prompt)
        caches[key] = cache
        todo = cache.missing(ids) if not args.no_resume else ids
        if not todo:
            logger.info("[%s] labels already complete (%d images)", key, len(cache))
            continue

        logger.info("[%s] generating labels for %d/%d images", key, len(todo), len(ids))
        from laod.models.llm_agent import build_llm_agent
        agent = build_llm_agent(key, device_map=args.device,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.do_sample)
        pending = {i for i in todo}
        loader = build_dataloader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        bar = tqdm(total=len(todo), desc=f"labels:{key}", unit="img")
        for batch in loader:
            for item in batch:
                if item["image_id"] not in pending:
                    continue
                t0 = time.time()
                try:
                    raw = agent.generate(item["image"], prompt)
                except Exception:
                    logger.exception("generation failed for image %s", item["image_id"])
                    raw = ""
                parsed = parse_labels(raw, args.parse_mode)
                parsed.flags["llm_seconds"] = round(time.time() - t0, 3)
                cache.put(item["image_id"], parsed)
                bar.update(1)
        bar.close()
        del agent
        _free_gpu()
    return caches


def stage_detect(args, gt, dataset, llms: list[str], detectors: list[str]) -> list[Path]:
    """Localise cached labels. One image decode serves every LLM's vocabulary."""
    gt_by_id = {g.image_id: g for g in gt}
    ann_path = Path(getattr(PATHS, ANNOTATIONS[args.dataset]))
    ann_md5 = file_digest(ann_path)
    caches = {k: LabelCache(PATHS.label_cache, f"{args.dataset}_{args.split}", k, args.prompt)
              for k in llms}
    missing = [k for k, c in caches.items() if len(c) == 0]
    if missing:
        raise SystemExit(f"no cached labels for {missing}; run --stage labels first")

    out_root = args.output_root or (PATHS.outputs / "runs")
    written: list[Path] = []

    for det_key in detectors:
        from laod.models.detector import build_detector
        detector = build_detector(det_key, device=args.device,
                                  **detector_params(det_key, args.detector_preset))
        spec = DETECTORS[det_key]

        stores = {}
        for llm_key in llms:
            cfg = RunConfig(
                dataset=f"{args.dataset}-{args.split}" + (f"-{args.subset}" if args.subset else ""), llm=llm_key, detector=det_key, prompt=args.prompt,
                llm_model_id=LLMS[llm_key].model_id, detector_model_id=spec.model_id,
                parse_mode=args.parse_mode, split_size=args.split_size,
                detector_params=dict(detector.params),
                generation_params={"max_new_tokens": args.max_new_tokens,
                                   "do_sample": args.do_sample},
                annotation_file=str(ann_path), annotation_md5=ann_md5,
                notes=f"split={args.split}; detector_preset={args.detector_preset}",
            )
            stores[llm_key] = RunStore(out_root, cfg, resume=not args.no_resume)

        loader = build_dataloader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        total = sum(len(stores[k].missing([g.image_id for g in gt])) for k in llms)
        bar = tqdm(total=total, desc=f"detect:{det_key}", unit="cell")
        for batch in loader:
            for item in batch:
                image_id = item["image_id"]
                for llm_key in llms:
                    store = stores[llm_key]
                    if image_id in store:
                        continue
                    parsed = caches[llm_key].get(image_id)
                    if parsed is None:
                        continue
                    t0 = time.time()
                    detector.set_labels(parsed.labels)
                    boxes, scores, labels = detector.detect(item["image"])
                    store.append(ImageRecord(
                        image_id=image_id, file_name=item["file_name"],
                        width=item["width"], height=item["height"],
                        raw_response=parsed.raw, labels=parsed.labels,
                        parse_flags=parsed.flags,
                        boxes=boxes.tolist(), scores=scores.tolist(),
                        pred_labels=labels,
                        gt_labels=list(gt_by_id[image_id].labels),
                        timings={"detect_seconds": round(time.time() - t0, 4)},
                    ))
                    bar.update(1)
        bar.close()
        for llm_key, store in stores.items():
            logger.info("[%s x %s] %s", llm_key, det_key, store.summary())
            written.append(store.dir)
        del detector
        _free_gpu()
    return written


def _free_gpu() -> None:
    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate_run(run_dir: Path, gt, device: str = "cuda:1") -> None:
    """Score one stored run with CAAP and the calibrated SNAP."""
    from laod.io.run_store import run_to_predictions
    from laod.metrics.caap import CAAPEvaluator
    from laod.metrics.grids import CAAP_V2, SNAP_LEGACY

    preds = run_to_predictions(run_dir)
    print(f"\n=== {run_dir.name} ===")
    res = CAAPEvaluator(gt, CAAP_V2.all_thresholds).evaluate(preds)
    s = CAAP_V2.summarise(res.per_threshold)
    print(f"CAAP  LO {s['LO']:.4f}  MI {s['MI']:.4f}  HI {s['HI']:.4f}  "
          f"50:95 {s['MACRO']:.4f}")
    try:
        from laod.metrics.snap import SNAPEvaluator, TextEmbedder
        snap = SNAPEvaluator(gt, TextEmbedder(device=device),
                             grid=SNAP_LEGACY).evaluate(preds)
        print(snap.report())
    except Exception as exc:
        logger.warning("SNAP skipped: %s", exc)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if _PINNED:
        logger.info("pinned %s -> CUDA_VISIBLE_DEVICES=%s (in-process %s)",
                    args.device, os.environ.get("CUDA_VISIBLE_DEVICES"), _PINNED)
        args.device = _PINNED

    llms = expand(args.llm, ACTIVE_LLMS)
    detectors = expand(args.detector, DETECTORS)
    gt_all, dataset = LOADERS[args.dataset].load(split_size=args.split_size)
    gt = apply_split(gt_all, args.split)
    if args.split == "all":
        logger.warning("--split all includes the tuning holdout; valid only for "
                       "legacy reproduction, not for tuned (v2) numbers")
    if args.subset == "ablation":
        # Each dataset has its own frozen draw: the prompt ablation is repeated
        # on LVIS to test whether a prompt tuned for one annotation scheme
        # transfers, and the two subsets must be independent.
        sub = load_ablation_subset(
            ABLATION_LVIS_PATH if args.dataset == "lvis" else None)
        gt = [g for g in gt if g.image_id in sub]
        logger.info("restricted to the frozen %s ablation subset: %d images",
                    args.dataset, len(gt))
    keep = {g.image_id for g in gt}
    dataset = DetectionImageDataset(gt, dataset.image_dir, file_names=dataset.file_names)
    logger.info("dataset=%s split=%s images=%d/%d llms=%s detectors=%s prompt=%s preset=%s",
                args.dataset, args.split, len(gt), len(gt_all), llms, detectors,
                args.prompt, args.detector_preset)

    if args.stage in ("labels", "all"):
        stage_labels(args, gt, dataset, llms)
    written: list[Path] = []
    if args.stage in ("detect", "all"):
        written = stage_detect(args, gt, dataset, llms, detectors)

    if args.save_preds and written:
        from laod.io.predictions import save_predictions
        from laod.io.run_store import run_to_predictions
        for run_dir in written:
            out = args.save_preds / f"{run_dir.name}.json"
            save_predictions(run_to_predictions(run_dir), out,
                             meta={"run": run_dir.name})
    if args.evaluate:
        for run_dir in written:
            evaluate_run(run_dir, gt, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
