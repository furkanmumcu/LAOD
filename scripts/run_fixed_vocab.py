#!/usr/bin/env python3
"""Run a detector over a dataset with a **fixed** vocabulary -- no LLM.

This is the closed-vocabulary control for the LAOD pipeline. Everywhere else in
v2 the vocabulary is generated per image by a VLM; here every image is given the
same label list, by default the dataset's own category names read straight out
of the annotation file. That makes it the strongest possible naming baseline:
the detector is told exactly the words the ground truth is written in, so any
remaining gap to the LLM-driven cells is localisation and grounding, not
vocabulary.

Reading the names from ``categories`` rather than hard-coding them matters. The
ground-truth labels SNAP scores against come from the same dictionary through
the same normalisation, so predicted and ground-truth strings are byte-identical
and SNAP's cosine is measuring the detector's assignment, not a spelling
mismatch we introduced.

Output is an ordinary run directory -- same schema as every other run -- so
CAAP, SNAP, U-AP, the vocabulary analysis and the renderer all work on it
unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from laod.device import pin_from_argv

# Must happen before anything imports torch: see laod/device.py.
_PINNED = pin_from_argv()

import numpy as np
from tqdm.auto import tqdm

from laod.config import PATHS
from laod.data import coco_loader, coco_ood_loader, lvis_loader
from laod.data.annotations import normalise_category
from laod.data.base import DetectionImageDataset, build_dataloader
from laod.data.splits import apply_split
from laod.io.run_store import ImageRecord, RunConfig, RunStore, file_digest
from laod.models.registry import DETECTORS, detector_params

logger = logging.getLogger("laod.fixed_vocab")

LOADERS = {"coco": coco_loader, "lvis": lvis_loader, "coco_ood": coco_ood_loader}
ANNOTATIONS = {"coco": "coco_ann", "lvis": "lvis_ann", "coco_ood": "coco_ood_ann"}


def dataset_vocabulary(dataset: str) -> list[str]:
    """The dataset's category names, normalised exactly as ground truth is."""
    ann = json.loads(Path(getattr(PATHS, ANNOTATIONS[dataset])).read_text("utf-8"))
    cats = sorted(ann["categories"], key=lambda c: int(c["id"]))
    return [normalise_category(c["name"], dataset) for c in cats]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(LOADERS), default="coco")
    p.add_argument("--detector", nargs="+", default=["gdino-tiny"],
                   help="registry keys, or 'all'")
    p.add_argument("--split", choices=["test", "tune", "all"], default="test",
                   help="'test' excludes the frozen hyperparameter holdout and "
                        "is what every reported number uses")
    p.add_argument("--split_size", type=int, default=None)
    p.add_argument("--vocab_file", type=Path, default=None,
                   help="newline-separated labels; defaults to the dataset's "
                        "own category names")
    p.add_argument("--vocab_name", default=None,
                   help="tag for the run slug; defaults to 'cls<N>'")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a detector parameter on top of the preset, "
                        "e.g. --set text_threshold=0.03. Recorded in the run "
                        "config and in the slug, so an override can never be "
                        "mistaken for the preset it departs from.")
    p.add_argument("--store_top", type=int, default=None,
                   help="store only the top-N detections per image by score, "
                        "recording the true uncapped count. Lossless for every "
                        "metric here, which caps at maxDets=100 anyway, and "
                        "necessary for the 1,203-class LVIS ontology where "
                        "Grounding DINO emits ~12k boxes per image across its "
                        "16 text passes. Never set below a few hundred.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_root", type=Path, default=None)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if _PINNED:
        args.device = _PINNED
        logger.info("pinned -> in-process %s", _PINNED)

    detectors = (list(DETECTORS) if args.detector == ["all"] else args.detector)
    unknown = [d for d in detectors if d not in DETECTORS]
    if unknown:
        raise SystemExit(f"unknown detector(s) {unknown}; have {sorted(DETECTORS)}")

    if args.vocab_file:
        vocab = [l.strip() for l in args.vocab_file.read_text("utf-8").splitlines()
                 if l.strip()]
        source = str(args.vocab_file)
    else:
        vocab = dataset_vocabulary(args.dataset)
        source = f"{args.dataset} annotation categories"
    overrides = {}
    for item in args.set:
        k, _, v = item.partition("=")
        if not _:
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        overrides[k.strip()] = float(v)
    tag = args.vocab_name or f"cls{len(vocab)}"
    if overrides:
        tag += "-" + "-".join(f"{k}{v:g}" for k, v in sorted(overrides.items()))
    logger.info("fixed vocabulary: %d label(s) from %s", len(vocab), source)

    gt_all, ds = LOADERS[args.dataset].load(split_size=args.split_size)
    gt = apply_split(gt_all, args.split)
    dataset = DetectionImageDataset(gt, ds.image_dir, file_names=ds.file_names)
    gt_by_id = {g.image_id: g for g in gt}
    ann_path = Path(getattr(PATHS, ANNOTATIONS[args.dataset]))
    ann_md5 = file_digest(ann_path)
    out_root = args.output_root or (PATHS.outputs / "runs")

    # The raw_response slot holds the vocabulary verbatim, as it would hold an
    # LLM's reply: a run directory should still be self-describing without the
    # config, and the vocabulary is the whole experimental condition here.
    raw = ", ".join(vocab)

    for det_key in detectors:
        from laod.models.detector import build_detector
        params = detector_params(det_key)
        params.update(overrides)
        detector = build_detector(det_key, device=args.device, **params)
        detector.set_labels(vocab)
        n_chunks = len(getattr(detector, "_chunks", [vocab]))
        logger.info("[%s] %s; vocabulary fits in %d text pass(es)",
                    det_key, params, n_chunks)

        cfg = RunConfig(
            dataset=f"{args.dataset}-{args.split}-fixed{len(vocab)}",
            llm="none", detector=det_key, prompt=tag,
            llm_model_id="", detector_model_id=DETECTORS[det_key].model_id,
            parse_mode="fixed", split_size=args.split_size,
            detector_params=dict(detector.params),
            generation_params={},
            annotation_file=str(ann_path), annotation_md5=ann_md5,
            notes=(f"closed-vocabulary control: no LLM. {len(vocab)} fixed "
                   f"labels from {source}; split={args.split}; "
                   f"text_passes={n_chunks}; "
                   f"store_top={args.store_top}"),
        )
        store = RunStore(out_root, cfg, resume=not args.no_resume)
        todo = set(store.missing([g.image_id for g in gt]))
        logger.info("[%s] -> %s (%d image(s) to do)", det_key, store.dir, len(todo))

        loader = build_dataloader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        bar = tqdm(total=len(todo), desc=f"fixed:{det_key}", unit="img")
        for batch in loader:
            for item in batch:
                image_id = item["image_id"]
                if image_id not in todo:
                    continue
                t0 = time.time()
                boxes, scores, labels = detector.detect(item["image"])
                n_raw = len(scores)
                if args.store_top and n_raw > args.store_top:
                    keep = np.argsort(-np.asarray(scores, float),
                                      kind="stable")[:args.store_top]
                    keep.sort()          # preserve the detector's own ordering
                    boxes, scores = boxes[keep], scores[keep]
                    labels = [labels[i] for i in keep]
                store.append(ImageRecord(
                    image_id=image_id, file_name=item["file_name"],
                    width=item["width"], height=item["height"],
                    raw_response=raw, labels=list(vocab),
                    parse_flags={"parse_ok": 1, "n_labels": len(vocab),
                                 "n_detections_uncapped": n_raw},
                    boxes=boxes.tolist(), scores=scores.tolist(),
                    pred_labels=labels,
                    gt_labels=list(gt_by_id[image_id].labels),
                    timings={"detect_seconds": round(time.time() - t0, 4)},
                ))
                bar.update(1)
        bar.close()
        logger.info("[%s] %s", det_key, store.summary())
        del detector
        try:
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
