#!/usr/bin/env python3
"""Vocabulary-level MLLM-OVOD feedback: one revision round, three conditions.

    C0 -> D0 -> (MLLM revises) -> C1 -> D1

Round 1 is not recomputed. It is read from the existing stored runs, so the
baseline condition is the current pipeline by construction rather than by
coincidence.

Detector settings are untouched between rounds -- same model, same thresholds,
same post-processing, same corrected label association, same image. Nothing is
tuned for the loop, and D0 and D1 are never merged: the output of an iterative
condition is D1 alone.

Round-2 decoding is greedy and every reply is cached on disk, so a re-run
reproduces the same vocabulary rather than resampling it.
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

_PINNED = pin_from_argv()

from tqdm.auto import tqdm

from laod.config import PATHS
from laod.data import coco_loader
from laod.data.annotations import load_ground_truth
from laod.data.base import DetectionImageDataset
from laod.data.splits import apply_split
from laod.feedback.prompts import NEEDS_FEEDBACK, build_prompt
from laod.feedback.schema import parse_revision
from laod.feedback.summary import concept_support, feedback_block
from laod.io.run_store import (ImageRecord, RunConfig, RunStore, file_digest,
                               load_run)
from laod.models.registry import DETECTORS, LLMS, detector_params

logger = logging.getLogger("laod.feedback")

CONDITIONS = ("selfreflect", "feedback_refine", "feedback_full")


class ReplyCache:
    """Round-2 replies keyed by image, so a re-run never re-queries the model.

    Self-reflection sees only the image and C0, so its key omits the detector:
    one pass serves every detector paired with that LLM, and the comparison
    across detectors is then exact rather than merely similar.
    """

    def __init__(self, root: Path, split: str, llm: str, condition: str,
                 detector: str | None) -> None:
        parts = [split, llm, condition] + ([detector] if detector else [])
        self.path = Path(root) / "feedback_cache" / f"{'__'.join(parts)}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[int, str] = {}
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        self._mem[int(d["image_id"])] = d["raw"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def get(self, image_id: int) -> str | None:
        return self._mem.get(int(image_id))

    def put(self, image_id: int, raw: str) -> None:
        self._mem[int(image_id)] = raw
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"image_id": int(image_id), "raw": raw}) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["tune", "test"], default="tune",
                   help="'tune' is the 500-image development holdout, which is "
                        "disjoint from the 4,500-image test split")
    p.add_argument("--llm", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   choices=list(CONDITIONS))
    p.add_argument("--max_vocab", type=int, default=40,
                   help="safeguard against an unbounded ADD; recorded per image "
                        "when it binds")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--detector_device", default=None)
    p.add_argument("--limit", type=int, default=None, help="first N images; smoke tests")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    dev = _PINNED or args.device
    det_dev = args.detector_device or dev

    gt_all = load_ground_truth(PATHS.coco_ann, "coco")
    gt = apply_split(gt_all, args.split)
    if args.limit:
        gt = gt[:args.limit]
    keep = {g.image_id for g in gt}
    _, ds = coco_loader.load()
    dataset = DetectionImageDataset([g for g in gt], ds.image_dir,
                                    file_names=ds.file_names)
    images = {it["image_id"]: it for i in range(len(dataset))
              for it in [dataset[i]]}

    # The dedicated baseline, built at exactly the preset Round 2 will use.
    # The plain coco-{split}__* runs are NOT interchangeable with it: they were
    # produced during the threshold sweep and sit at pre-fit operating points
    # (OWLv2 0.15 against the fitted 0.01), which would put Round 1 and Round 2
    # at different thresholds and confound the comparison with the cap.
    base_dir = (PATHS.outputs / "runs" /
                f"coco-{args.split}-fb-baseline__{args.llm}__{args.detector}__default")
    if not base_dir.is_dir():
        raise SystemExit(f"round-1 baseline not found: {base_dir}\n"
                         f"build it first with scripts/make_feedback_baseline.py")
    cfg0, recs0 = load_run(base_dir)
    base = {r.image_id: r for r in recs0 if r.image_id in keep}
    missing = keep - set(base)
    if missing:
        raise SystemExit(f"round-1 run is missing {len(missing)} image(s)")
    logger.info("round 1: %s (%d images)", base_dir.name, len(base))

    ann_md5 = file_digest(PATHS.coco_ann)
    from laod.models.detector import build_detector
    from laod.models.llm_agent import build_llm_agent

    agent = None
    for condition in args.conditions:
        cache = ReplyCache(PATHS.outputs, f"coco_{args.split}", args.llm, condition,
                           None if condition == "selfreflect" else args.detector)
        todo = [g for g in gt if cache.get(g.image_id) is None]
        logger.info("[%s] %d cached reply/replies, %d to generate",
                    condition, len(cache), len(todo))

        if todo:
            if agent is None:
                # Greedy decoding: a resampled vocabulary would make the whole
                # comparison irreproducible.
                agent = build_llm_agent(args.llm, device_map=dev,
                                        max_new_tokens=args.max_new_tokens,
                                        do_sample=False)
            bar = tqdm(total=len(todo), desc=f"round2:{condition}", unit="img")
            for g in todo:
                rec = base[g.image_id]
                c0 = list(rec.labels)
                fb = ""
                if condition in NEEDS_FEEDBACK:
                    # Support is counted over every detection above the
                    # detector's own confidence threshold -- the maxDets cap is
                    # an evaluation rule and would misreport a concept as
                    # unsupported merely because the image was crowded.
                    fb = feedback_block(concept_support(c0, rec.pred_labels,
                                                        rec.scores))
                prompt = build_prompt(condition, c0, fb)
                try:
                    raw = agent.generate(images[g.image_id]["image"], prompt)
                except Exception:
                    logger.exception("round-2 generation failed on %s", g.image_id)
                    raw = ""
                cache.put(g.image_id, raw)
                bar.update(1)
            bar.close()

    if agent is not None:
        del agent
        try:
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- ground each revised vocabulary ----------------------------------
    detector = build_detector(args.detector, device=det_dev,
                              **detector_params(args.detector))
    for condition in args.conditions:
        cache = ReplyCache(PATHS.outputs, f"coco_{args.split}", args.llm, condition,
                           None if condition == "selfreflect" else args.detector)
        cfg = RunConfig(
            dataset=f"coco-{args.split}-fb-{condition}", llm=args.llm,
            detector=args.detector, prompt="default",
            llm_model_id=LLMS[args.llm].model_id,
            detector_model_id=DETECTORS[args.detector].model_id,
            parse_mode="strict", detector_params=dict(detector.params),
            generation_params={"max_new_tokens": args.max_new_tokens,
                               "do_sample": False},
            annotation_file=str(PATHS.coco_ann), annotation_md5=ann_md5,
            notes=(f"vocabulary feedback, condition={condition}, "
                   f"round 2 of 2, max_vocab={args.max_vocab}; "
                   f"round 1 = {base_dir.name}"))
        store = RunStore(PATHS.outputs / "runs", cfg, resume=False)
        trace = store.dir / "feedback.jsonl"
        trace.unlink(missing_ok=True)

        bar = tqdm(total=len(gt), desc=f"detect:{condition}", unit="img")
        t0 = time.time()
        for g in gt:
            rec = base[g.image_id]
            c0 = list(rec.labels)
            raw = cache.get(g.image_id) or ""
            rev = parse_revision(raw, c0, condition, max_vocab=args.max_vocab)
            fb = ""
            if condition in NEEDS_FEEDBACK:
                fb = feedback_block(concept_support(c0, rec.pred_labels, rec.scores))
            item = images[g.image_id]
            detector.set_labels(rev.c1)
            boxes, scores, labels = detector.detect(item["image"])
            store.append(ImageRecord(
                image_id=g.image_id, file_name=item["file_name"],
                width=item["width"], height=item["height"],
                raw_response=raw, labels=list(rev.c1), parse_flags=rev.flags(),
                boxes=boxes.tolist(), scores=scores.tolist(),
                pred_labels=labels, gt_labels=list(g.labels)))
            with trace.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"image_id": g.image_id,
                                     "condition": condition,
                                     "feedback": fb, **rev.to_json()}) + "\n")
            bar.update(1)
        bar.close()
        logger.info("[%s] %s  (%.1f min) -> %s", condition, store.summary(),
                    (time.time() - t0) / 60, store.dir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
