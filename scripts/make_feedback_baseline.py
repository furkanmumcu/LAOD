#!/usr/bin/env python3
"""Round-1 baseline for the feedback experiment, at the current v2 preset.

The stored ``coco-tune__*`` runs were produced *during* the threshold sweep and
therefore sit at pre-fit operating points -- OWLv2 at 0.15 against the fitted
0.01, Grounding DINO at 0.05/0.075 against 0.03. They are correct for what they
were for, and are left untouched.

This experiment needs a Round 1 at exactly the settings Round 2 will use,
because both the baseline comparison and the per-concept feedback are derived
from it. Writing a dedicated baseline makes that identity structural rather
than something to be checked after the fact.

Labels are read from the existing cache, so this is a detector pass only -- no
LLM, and C0 is bit-identical to the stored run's.
"""

from __future__ import annotations

import argparse, logging, sys, time
sys.path.insert(0, ".")
from laod.device import pin_from_argv

_PINNED = pin_from_argv()

from tqdm.auto import tqdm

from laod.config import PATHS
from laod.data import coco_loader
from laod.data.annotations import load_ground_truth
from laod.data.base import DetectionImageDataset
from laod.data.splits import apply_split
from laod.io.label_cache import LabelCache
from laod.io.run_store import ImageRecord, RunConfig, RunStore, file_digest
from laod.models.registry import DETECTORS, LLMS, detector_params

logger = logging.getLogger("laod.fb_baseline")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["tune", "test"], default="tune")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dev = _PINNED or args.device

    gt = apply_split(load_ground_truth(PATHS.coco_ann, "coco"), args.split)
    if args.limit:
        gt = gt[:args.limit]
    _, ds = coco_loader.load()
    dataset = DetectionImageDataset(gt, ds.image_dir, file_names=ds.file_names)
    images = {it["image_id"]: it for i in range(len(dataset)) for it in [dataset[i]]}

    cache = LabelCache(PATHS.label_cache, f"coco_{args.split}", args.llm, "default")
    if not len(cache):
        raise SystemExit(f"no cached labels for {args.llm} on coco_{args.split}")

    from laod.models.detector import build_detector
    params = detector_params(args.detector)
    detector = build_detector(args.detector, device=dev, **params)
    cfg = RunConfig(
        dataset=f"coco-{args.split}-fb-baseline", llm=args.llm,
        detector=args.detector, prompt="default",
        llm_model_id=LLMS[args.llm].model_id,
        detector_model_id=DETECTORS[args.detector].model_id,
        parse_mode="strict", detector_params=dict(detector.params),
        annotation_file=str(PATHS.coco_ann),
        annotation_md5=file_digest(PATHS.coco_ann),
        notes=f"round-1 baseline for the vocabulary-feedback experiment; "
              f"split={args.split}; detector params={params}")
    store = RunStore(PATHS.outputs / "runs", cfg, resume=False)
    logger.info("%s + %s at %s -> %s", args.llm, args.detector, params, store.dir.name)

    t0 = time.time()
    for g in tqdm(gt, desc=f"baseline:{args.detector}", unit="img"):
        parsed = cache.get(g.image_id)
        if parsed is None:
            continue
        item = images[g.image_id]
        detector.set_labels(parsed.labels)
        boxes, scores, labels = detector.detect(item["image"])
        store.append(ImageRecord(
            image_id=g.image_id, file_name=item["file_name"],
            width=item["width"], height=item["height"],
            raw_response=parsed.raw, labels=list(parsed.labels),
            parse_flags=parsed.flags, boxes=boxes.tolist(),
            scores=scores.tolist(), pred_labels=labels,
            gt_labels=list(g.labels)))
    logger.info("%s  (%.1f min)", store.summary(), (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
