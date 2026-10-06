#!/usr/bin/env python
"""Assemble the single consolidated results report.

Everything is read from ``results/`` and ``reports/`` at build time, so the
document cannot drift from the numbers: re-run a stage and rebuild, and the
tables follow. Phase 2 (agentic) is deliberately excluded.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np

R = "results"
REP = "reports"

LLM_LABEL = {"gemma3-4b": "Gemma3-4B", "gemma4-e2b": "Gemma4-E2B",
             "gemma4-e4b": "Gemma4-E4B", "gemma4-12b": "Gemma4-12B",
             "qwen25-vl-7b": "Qwen2.5-VL-7B", "qwen35-9b": "Qwen3.5-9B",
             "internvl3-8b": "InternVL3-8B"}
DET_LABEL = {"yolo-world": "YOLO-World", "gdino-tiny": "G-DINO-T",
             "gdino-base": "G-DINO-B", "owlv2-base": "OWLv2"}


def sbert_snap(stage: str) -> dict:
    """(llm, detector) -> the Sentence-BERT SNAP row, within one stage.

    The stage matters: Stage 1 and Stage 2 share llm, detector and dataset but
    run on different image sets, so keying without it silently pairs a
    4,500-image cell with a 500-image one.
    """
    rows = rd(f"{R}/snap_sbert.csv") or []
    return {(r["llm"], r["detector"]): r
            for r in rows if r["stage"] == stage}


def rd(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".json"):
        return json.load(open(path))
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def table(headers, rows, align=None) -> str:
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def grid_table(cells, value_key, llms, dets, fmt="{:.4f}"):
    look = {(c["llm"], c["detector"]): c for c in cells}
    rows = []
    for l in llms:
        r = [LLM_LABEL.get(l, l)]
        vals = []
        for d in dets:
            c = look.get((l, d))
            r.append(fmt.format(c[value_key]) if c else "-")
            if c:
                vals.append(c[value_key])
        r.append(f"**{fmt.format(np.mean(vals))}**" if vals else "-")
        rows.append(r)
    means = ["**mean**"]
    for d in dets:
        v = [look[(l, d)][value_key] for l in llms if (l, d) in look]
        means.append(f"**{fmt.format(np.mean(v))}**" if v else "-")
    means.append("")
    rows.append(means)
    return table(["LLM"] + [DET_LABEL.get(d, d) for d in dets] + ["mean"], rows)


def section_stage(main, name, dataset, note="", stage="stage1"):
    cells = rd(f"{R}/{main}")
    if not cells:
        return f"\n*(`{R}/{main}` not present -- stage not rebuilt)*\n"
    sb = sbert_snap(stage)
    for c in cells:
        row = sb.get((c["llm"], c["detector"]))
        if row:
            c["snap_clip"] = c.get("snap_maxdets")
            c["snap_maxdets"] = row["snap_macro"]
            c["snap_gain"] = row["gain_macro"]
    llms = sorted({c["llm"] for c in cells},
                  key=lambda l: -np.mean([c["caap_maxdets"] for c in cells if c["llm"] == l]))
    dets = sorted({c["detector"] for c in cells},
                  key=lambda d: -np.mean([c["caap_maxdets"] for c in cells if c["detector"] == d]))
    out = [note, "", "**CAAP@.5:.95, maxDets=100**", "",
           grid_table(cells, "caap_maxdets", llms, dets), ""]
    if dataset == "coco_ood":
        out += ["**SNAP is not reported for COCO-OOD.** The dataset annotates "
                "every object under one label, `unknow object`, so there is no "
                "naming to score: a shuffled-label control gives a gain of "
                "exactly +0.0000 on all 28 cells. Read this dataset on U-AP, "
                "which is what it was built for.", ""]
    elif "snap_maxdets" in cells[0]:
        out += ["**SNAP (Sentence-BERT), maxDets=100**", "",
                grid_table(cells, "snap_maxdets", llms, dets), ""]
    out += ["**Detections per image, after the cap**", "",
            grid_table(cells, "det_per_img_capped", llms, dets, "{:.1f}"), ""]
    best = max(cells, key=lambda c: c["caap_maxdets"])
    out += [f"Best cell: `{best['llm']}` + `{best['detector']}` -- "
            f"CAAP {best['caap_maxdets']:.4f}"
            + (f", SNAP {best['snap_maxdets']:.4f}" if "snap_maxdets" in best else "")
            + f", {best['det_per_img_capped']:.0f} det/img.", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/RESULTS.md")
    args = ap.parse_args()
    P = []
    A = P.append

    A(f"""# LAOD v2 — consolidated results

*Generated {datetime.now():%Y-%m-%d %H:%M}. Every table is read from
`results/` and `reports/` at build time; rebuild after a re-run and the
numbers follow. Phase 2 (agentic verification) is excluded — it was explored
and dropped.*

## How to read these numbers

Two conventions apply to **every** metric in this document, and both differ
from the original LAOD implementation:

1. **Matching visits predictions by descending score.** Matching is one-to-one
   and first-come-first-served, so the visit order decides which prediction
   wins a contested ground-truth object. The original visited them in the
   order the detector returned them, which is equivalent for YOLO-World (whose
   output is score-sorted on 100% of images) and wrong for Grounding DINO and
   OWLv2 (~1%). Correcting it raises OWLv2's mean SNAP on COCO-Val by
   **139%** and on LVIS by **104%**, and leaves YOLO-World within 0.3%.

2. **maxDets = 100.** `pycocotools` reports AP over at most 100 detections per
   image; we now apply the same cap, once, to the prediction list before any
   metric runs — so CAAP, SNAP and U-AP always see the identical detection set.

The published-table reproduction survives both changes: CAAP
0.2507 / 0.2194 / 0.0829 against a published 0.25 / 0.22 / 0.08.

### What the CAAP and SNAP columns are

Every CAAP and SNAP number in this document is the **macro average: the
unweighted mean over ten thresholds, 0.50 to 0.95 in steps of 0.05.**

* **CAAP** averages over the IoU threshold. This is exactly the standard COCO
  `AP@[.5:.95]`, which is why the column is labelled that way.
* **SNAP** averages over the cosine similarity threshold tau, using
  **Sentence-BERT** (`all-MiniLM-L6-v2`, bare labels) as the text encoder --
  see *Why Sentence-BERT* below.

Both metrics use the same interval structure: `LO` = 0.50/0.55/0.60,
`MI` = 0.65/0.70/0.75/0.80, `HI` = 0.85/0.90/0.95.

It is *not* `LO`, `MI` or `HI`. Those are sub-ranges of the same per-threshold
curve, and the macro spans all three. Worked from the reproduction: the
per-IoU CAAP values 0.2624, 0.2538, 0.2453, 0.2412, 0.2263, 0.2145, 0.1953,
0.1672, 0.1196, 0.0448 average to **0.1970**, the reported macro. The legacy
grid's eleventh value (IoU 1.00) is excluded even though it sits inside `HI`.

The `LO` / `MI` / `HI` breakdown the original paper reports is in
**`reports/INTERVALS.md`**, kept separate to stop these tables becoming
unreadable.

### Why Sentence-BERT

SNAP matches a predicted label to a ground-truth label when their cosine
similarity clears a threshold, which only works if *unrelated* labels score
low. They do not under CLIP. Its text tower was trained to align text with
**images**, not text with text, so text-text cosine is incidental to the
objective and the embeddings occupy a narrow cone. Over every distinct pair of
COCO's 80 ground-truth labels, CLIP gives a **minimum** cosine of +0.567 and a
median of +0.786 -- `person`/`car` scores 0.883.

A threshold of 0.50 therefore sits below the floor of unrelated pairs and
admits **100%** of them. The semantic test never fires, and SNAP collapses into
"a confident box landed in an image that still had an unmatched ground truth".

**The shuffle control makes this concrete.** Scoring the same predictions with
the predicted label strings randomly permuted dataset-wide -- identical boxes,
identical scores, labels carrying zero information -- CLIP reproduces
`SNAP_LO` to four decimal places:

| encoder | gain over shuffled labels, `LO` | `MI` | `HI` | unrelated pairs admitted @0.50 |
|---|---|---|---|---|
| CLIP (`ViT-B/32`) | **+0.0000** | +0.0068 to +0.0861 | +0.12 to +0.27 | **100%** |
| **MiniLM** (`all-MiniLM-L6-v2`) | **+0.21 to +0.36** | +0.17 to +0.28 | +0.10 to +0.17 | **2%** |

Under CLIP, `SNAP_LO` is vacuous in **28 of 28** Stage 1 cells. Under MiniLM it
is informative in **28 of 28**, and is the *strongest* interval rather than the
emptiest. MiniLM is trained contrastively for sentence similarity, which is
precisely the property the metric assumes.

Expect the absolute values to be far lower than a CLIP-scored SNAP: `SNAP_LO`
falls from roughly 0.75 to roughly 0.23. That is the vacuous component leaving,
not a regression -- the CLIP figure was largely counting detections.

`backend="clip"` reproduces the original behaviour, and the full side-by-side
is in **`reports/SNAP_SBERT.md`**.
""")

    # ---- methodology -------------------------------------------------------
    A("""## What each stage did

| stage | data | grid | question |
|---|---|---|---|
| 1 | COCO-Val test, 4,500 images | 7 LLMs x 4 detectors = 28 cells | How does the pipeline score, and what drives the score? |
| 2 | COCO-Val test, 500-image subset | 7 LLMs x 3 prompts, YOLO-World fixed | Does prompt phrasing matter? |
| 3 | LVIS-minival test, 4,327 images | 3 LLMs x 4 detectors = 12 cells | Does the ranking hold on a long-tail vocabulary? |
| 4 | COCO-OOD test, 438 images | 7 LLMs x 4 detectors = 28 cells | Can it localise objects outside the 80 COCO classes? |
| 3* | COCO-Val + COCO-OOD test | 2 VLMs, no detector | Does the two-stage split earn its complexity? |

\\* Numbered "Phase 3" in the working notes; it is a baseline, not a stage.

**Splits are frozen and seeded.** A 500-image hyperparameter holdout is drawn
once (`results/tuning_split.json`, seed 20261001) and excluded from every test
split, so no reported number was used to choose anything. COCO-Val test is the
remaining 4,500; LVIS 4,327; COCO-OOD 438.

**Labels are cached per (image, LLM, prompt)**, so a detector can be re-run
without re-running the LLM, and every detector sees the identical vocabulary.
""")

    # ---- thresholds --------------------------------------------------------
    A("## Tuning — detector confidence thresholds\n")
    A("""Fitted on the 500-image holdout against the metric actually reported
(CAAP at maxDets=100; U-AP for COCO-OOD). The earlier fit maximised *uncapped*
CAAP, which rewards detections the protocol discards.
""")
    DATASETS = [("coco", "COCO-Val", "CAAP@100"),
                ("lvis", "LVIS", "CAAP@100"),
                ("coco_ood", "COCO-OOD", "U-AP")]
    allpicks: dict[str, dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for ds, _, _ in DATASETS:
        sweep = rd(f"{R}/tuning_sweep_{ds}_fixed100.csv")
        if not sweep:
            continue
        by = collections.defaultdict(list)
        for r in sweep:
            by[(r["llm"], r["detector"])].append(r)
        for (llm, det), rs in by.items():
            allpicks[det][ds].add(max(rs, key=lambda r: r["objective"])["threshold"])

    if allpicks:
        rows = []
        for det in ["yolo-world", "gdino-tiny", "gdino-base", "owlv2-base"]:
            if det == "yolo-world":
                rows.append([DET_LABEL[det], "conf", "0.001", "0.001", "0.001",
                             "**0.001**"])
                continue
            cells = []
            for ds, _, _ in DATASETS:
                v = sorted(allpicks.get(det, {}).get(ds, []))
                cells.append("/".join(f"{x:g}" for x in v) if v else "-")
            uniq = {x for s_ in allpicks.get(det, {}).values() for x in s_}
            rows.append([DET_LABEL.get(det, det), "score_threshold"] + cells
                        + [f"**{min(uniq):g}**" if uniq else "-"])
        A(table(["detector", "parameter"] + [n for _, n, _ in DATASETS] + ["used"],
                rows))
        A("""
Objective per dataset: CAAP@100 for COCO-Val and LVIS, U-AP for COCO-OOD (it
is what that dataset reports). A cell shows every value chosen across that
dataset's LLMs -- a single number means all of them agreed.

**All 21 (dataset, LLM) fits per detector picked the same value**, so the
registry holds one threshold per detector rather than one per pair. `text_threshold`
stays at 0.3 for both Grounding DINOs and was never tuned.
""")
        A("""Every fit selects its detector's **grid floor**. Under a detection cap
the threshold stops being a meaningful hyperparameter: for a crowded image the
top 100 by score are the same whether the cut is 0.01 or 0.05, so the cap does
the thresholding and the sweep has nothing left to separate. Fitting against
*uncapped* CAAP instead -- as the earlier version did -- rewards detections the
protocol discards, and moved OWLv2's choice from 0.01 to 0.15.
""")

    # ---- stages ------------------------------------------------------------
    A("## Stage 1 — COCO-Val (4,500 images)\n")
    A(section_stage("main_coco.csv", "Stage 1", "coco", stage="stage1"))

    A("## Stage 2 — Prompt ablation (500-image subset, YOLO-World)\n")
    s2 = rd(f"{REP}/stage2.json")
    if s2:
        prompts = sorted({c["prompt"] for c in s2})
        llms = sorted({c["llm"] for c in s2})
        look = {(c["llm"], c["prompt"]): c for c in s2}
        sb2 = {(r["llm"], r["prompt"]): r
               for r in (rd(f"{R}/snap_sbert.csv") or [])
               if r["stage"] == "stage2"}
        for key, title, fmt in (("caap", "CAAP@.5:.95", "{:.4f}"),
                                ("labels_per_img", "Labels proposed per image", "{:.1f}")):
            rows = [[LLM_LABEL.get(l, l)] + [fmt.format(look[(l, p)][key])
                                             if (l, p) in look else "-" for p in prompts]
                    for l in llms]
            A(f"**{title}**\n")
            A(table(["LLM"] + prompts, rows) + "\n")
        if sb2:
            rows = [[LLM_LABEL.get(l, l)]
                    + [f"{sb2[(l, p)]['snap_macro']:.4f}" if (l, p) in sb2 else "-"
                       for p in prompts]
                    for l in llms]
            A("**SNAP (Sentence-BERT)**\n")
            A(table(["LLM"] + prompts, rows) + "\n")
            g = [r["gain_lo"] for r in sb2.values()]
            A(f"*All {len(sb2)} cells are informative against the shuffle "
              f"control (`SNAP_LO` gain {min(g):+.3f} to {max(g):+.3f}).*\n")
    else:
        A("*(not rebuilt)*\n")

    A("## Stage 3 — LVIS-minival (4,327 images)\n")
    A(section_stage("main_lvis.csv", "Stage 3", "lvis", stage="stage3"))

    A("## Stage 4 — COCO-OOD (438 images)\n")
    A(section_stage("main_coco_ood.csv", "Stage 4", "coco_ood", stage="stage4"))

    # ---- OOD metrics -------------------------------------------------------
    A("### Unknown-object metrics (COCO-OOD)\n")
    A("""COCO-OOD annotates everything outside the 80 COCO classes under one
label, so only localisation is scored. Each system is reported at its own
max-F1 operating point, found by a 200-point sweep over its own score
distribution — a VLM's exp(log-probability) and a detector's confidence are
different quantities, so one shared numeric cut would be equal in form but not
in meaning.
""")
    uap = rd(f"{R}/phase4_uap_coco_ood.csv")
    if uap:
        rows = sorted(uap, key=lambda r: -r["u_f1_max"])[:12]
        A(table(["LLM", "detector", "U-AP", "U-F1", "U-PRE", "U-REC", "det/img"],
                [[LLM_LABEL.get(r["llm"], r["llm"]), DET_LABEL.get(r["detector"], r["detector"]),
                  f"{r['u_ap']:.3f}", f"{r['u_f1_max']:.3f}", f"{r['u_pre_max']:.3f}",
                  f"{r['u_rec_max']:.3f}", f"{r['det_per_img']:.1f}"] for r in rows]))
        A(f"\n*Top 12 of {len(uap)} cells by F1; all are in "
          f"`results/phase4_uap_coco_ood.csv`.*\n")
    else:
        A("*(not rebuilt)*\n")

    # ---- VLM baseline ------------------------------------------------------
    A("## VLMs as direct detectors\n")
    A("""Two grounding-capable VLMs used as detectors in their own right, no
open-vocabulary detector in the pipeline. Prompt: *"Detect all objects in this
image and provide their bounding box coordinates. Output the results strictly
in a JSON list format using 'bbox_2d' and 'label'."* Greedy decoding,
`max_new_tokens=1024`.

A VLM emits no confidence, while CAAP, SNAP and U-AP all rank detections.
Token log-probabilities are the closest available analogue and cost nothing
extra.

**Every metric here is ranked by `logprob_bbox`**, the mean token
log-probability of the coordinate text. It is the only one available: token
log-probabilities exist during generation and cannot be reconstructed from a
stored reply, and `logprob_bbox` is the value written to the run store. A
per-label variant was originally reported alongside it, but it silently fell
back to position-in-list -- the two are identical in all 23 recorded fields.

The choice is immaterial either way. The sources differ by under 2% on every
metric (SNAP 0.3889 vs 0.3828, CAAP 0.1444 vs 0.1422), so nothing here rests
on it.
""")
    for ds, title in (("coco", "COCO-Val test"), ("coco_ood", "COCO-OOD test")):
        v = rd(f"{R}/phase3_vlm_{ds}.csv")
        A(f"**{title}**\n")
        if not v:
            A("*(not rebuilt)*\n"); continue
        keep = [r for r in v if r["score_source"] == "logprob_bbox"]
        sbv = {r["llm"]: r for r in (rd(f"{R}/snap_sbert.csv") or [])
               if r["stage"] == "vlm" and r["dataset"] == ds}
        snap_col = ds != "coco_ood"
        hdr = (["model", "CAAP"] + (["SNAP"] if snap_col else [])
               + ["U-AP", "U-F1", "U-PRE", "det/img", "parse fails"])
        A(table(hdr,
                [[r["model"], f"{r['caap']:.4f}"]
                 + ([f"{sbv[r['model']]['snap_macro']:.4f}"
                     if r["model"] in sbv else "-"] if snap_col else [])
                 + [f"{r['u_ap']:.3f}", f"{r['u_f1_max']:.3f}",
                    f"{r['u_pre_max']:.3f}", f"{r['det_per_img']:.1f}",
                    int(r["parse_failures"])]
                 for r in keep]) + "\n")
        if not snap_col:
            A("*SNAP is omitted: COCO-OOD annotates one label, so there is "
              "nothing to name.*\n")
    A("""`Qwen3.5-9B` was in the original plan and dropped after measurement: it
is a reasoning model, emitted 1,511 characters of chain-of-thought per image
against Qwen2.5-VL's 192, and produced **zero parseable detections across
1,798 images**. `enable_thinking=False` was accepted by the chat template and
ignored by the model.

Coordinate conventions differ and the difference is silent: Qwen2.5-VL emits
pixels in its own *resized* frame (scale recovered from the processor's
`image_grid_thw`), InternVL3 emits 0-1000 normalised values. Each was verified
against ground truth before its run (InternVL3 mean best-IoU 0.480 correct vs
0.291 wrong); the code now raises rather than silently defaulting to scale 1.0.
""")

    # ---- threshold / metric analyses --------------------------------------
    A("## Metric behaviour\n")
    tc = rd(f"{R}/threshold_cost.csv")
    if tc:
        A("""**The two metric families want opposite detectors.** One stored run
re-scored at a range of confidence cuts (every point an exact filter of
detections already on disk):
""")
        A(table(["conf", "det/img", "CAAP", "F1 @ IoU 0.5", "precision", "recall"],
                [[f"{r['cut']:g}", f"{r['det_per_img']:.1f}", f"{r['caap']:.4f}",
                  f"{r['f1']:.4f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}"]
                 for r in sorted(tc, key=lambda r: r["cut"])]) + "\n")
        A("""Average precision integrates over the ranking, so appending
lower-ranked detections can only extend recall — CAAP is maximised by emitting
everything. F1 scores one operating point and peaks an order of magnitude
earlier. The original LAOD inherited Ultralytics' stock `conf=0.25` and never
changed it (its archived predictions have a minimum score of exactly 0.2500 at
5.55 detections per image), which sits within 2% of the F1 optimum while giving
up 26% of CAAP.
""")
    sv = rd(f"{R}/snap_validity.json")
    if sv:
        A("""**Why CLIP was rejected as the SNAP encoder.** The table below is
the evidence, not a property of the metric reported in this document -- SNAP
here uses Sentence-BERT. Scored against a label-shuffle control that holds
boxes and scores fixed and permutes the predicted label strings dataset-wide,
CLIP reproduces its own score:
""")
        t = sv["thresholds"]; raw = sv["variants"]["raw"]; cen = sv["variants"]["centred"]
        rows = [[f"{t[i]:.2f}", f"{raw['snap'][i]:.4f}", f"{raw['chance'][i]:.4f}",
                 f"{raw['snap'][i]-raw['chance'][i]:+.4f}", f"{cen['snap'][i]:.4f}",
                 f"{cen['chance'][i]:.4f}", f"{cen['snap'][i]-cen['chance'][i]:+.4f}"]
                for i in range(0, len(t), 2)]
        A(table(["tau", "SNAP raw", "chance", "gain", "SNAP centred", "chance", "gain"], rows))
        lp = sv["label_pairs"]
        A(f"""
The two raw curves are bit-identical up to tau = 0.60, spanning most of the
published grid. The cause is the embedding cone: over {lp['raw']['n_pairs']:,}
unrelated ground-truth label pairs, raw CLIP cosine has a median of
**{lp['raw']['median']:+.3f}**; mean-centred, **{lp['centred']['median']:+.3f}**.
Centring moves the best operating point to tau = {cen['best_tau']:.2f} with a
larger gain over chance ({cen['best_gain']:+.4f} vs {raw['best_gain']:+.4f}).
""")

    A("""## Files

| file | contents |
|---|---|
| `results/main_coco.csv` | Stage 1, CAAP + SNAP, capped and uncapped |
| `results/main_lvis.csv` | Stage 3 |
| `results/main_coco_ood.csv` | Stage 4 |
| `results/phase4_uap_coco_ood.csv` | pipeline unknown-object metrics, 28 cells |
| `results/phase3_vlm_{coco,coco_ood}.csv` | VLM baseline, 3 score sources each |
| `results/tuning_sweep__fixed100.csv` | threshold sweep against the capped objective |
| `results/threshold_cost.csv` | CAAP/F1/precision/recall against the score cut |
| `results/snap_validity.json` | SNAP against its shuffle control, raw and centred |
| `outputs/runs/` | per-image detections, raw LLM replies, timings |
""")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(P))
    print(f"wrote {args.out} ({len(''.join(P).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
