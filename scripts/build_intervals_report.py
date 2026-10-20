#!/usr/bin/env python
"""Assemble the LO / MI / HI interval breakdown.

The original paper reports CAAP and SNAP as three interval means rather than a
single macro, so comparing against it needs the breakdown. It lives apart from
`RESULTS.md` because three extra columns per metric per cell makes those tables
unreadable.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np

from laod.data.annotations import has_semantic_labels

R, REP = "results", "reports"
LLM_LABEL = {"gemma3-4b": "Gemma3-4B", "gemma4-e2b": "Gemma4-E2B",
             "gemma4-e4b": "Gemma4-E4B", "gemma4-12b": "Gemma4-12B",
             "qwen25-vl-7b": "Qwen2.5-VL-7B", "qwen35-9b": "Qwen3.5-9B",
             "internvl3-8b": "InternVL3-8B"}
DET_LABEL = {"yolo-world": "YOLO-World", "gdino-tiny": "G-DINO-T",
             "gdino-base": "G-DINO-B", "owlv2-base": "OWLv2"}


def rd(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".json"):
        return json.load(open(path))
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def semloc_marginals(stage: str):
    """Mean SemLoc-AP along each threshold axis, averaged over a stage's cells.

    SemLoc-AP has no LO/MI/HI structure -- it is a 10x10 grid, not a sweep of
    one threshold -- so the analogue of an interval breakdown is the pair of
    marginals: hold IoU and average over cosine, then the reverse. That is what
    shows whether a cell is losing points to localisation or to naming.
    """
    rows = rd(f"{R}/semloc.csv") or []
    mats = rd(f"{R}/semloc_matrices.json") or {}
    runs = [r["run"] for r in rows if r["stage"] == stage]
    grids = [np.array(mats[k]["matrix"]) for k in runs if k in mats]
    if not grids:
        return None
    entry = mats[next(k for k in runs if k in mats)]
    m = np.mean(grids, axis=0)
    return (entry["iou_thresholds"], entry["sem_thresholds"],
            m.mean(axis=1), m.mean(axis=0), float(m.mean()), len(grids))


def merge_sbert(cells, dataset, stage):
    """Replace the CLIP SNAP intervals with the Sentence-BERT ones.

    Keyed within a stage: Stage 1 and Stage 2 share llm, detector and dataset
    but score different image sets, so a key without the stage pairs a
    4,500-image cell with a 500-image one.
    """
    rows = rd(f"{R}/snap_sbert.csv") or []
    sb = {(r["llm"], r["detector"]): r
          for r in rows if r["stage"] == stage}
    for c in cells:
        r = sb.get((c["llm"], c["detector"]))
        if not r:
            continue
        for k in ("lo", "mi", "hi"):
            c[f"snap_{k}"] = r[f"snap_{k}"]
            c[f"snapgain_{k}"] = r[f"gain_{k}"]
        c["snap_maxdets"] = r["snap_macro"]
    return cells


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def cells_table(cells, prefix: str) -> str:
    """One row per cell: LO, MI, HI and the macro, sorted by macro."""
    key = f"{prefix}_maxdets"
    rows = []
    for c in sorted(cells, key=lambda c: -c.get(key, 0)):
        if f"{prefix}_lo" not in c:
            continue
        rows.append([LLM_LABEL.get(c["llm"], c["llm"]),
                     DET_LABEL.get(c["detector"], c["detector"]),
                     f"{c[prefix + '_lo']:.4f}", f"{c[prefix + '_mi']:.4f}",
                     f"{c[prefix + '_hi']:.4f}", f"**{c[key]:.4f}**",
                     f"{c['det_per_img_capped']:.1f}"])
    if not rows:
        return "*(interval columns not present -- re-run the scoring step)*"
    return table(["LLM", "detector", "LO", "MI", "HI", "macro", "det/img"], rows)


def interval_means(cells, prefix: str) -> str:
    by = collections.defaultdict(list)
    for c in cells:
        if f"{prefix}_lo" in c:
            by[c["detector"]].append(c)
    rows = []
    for d, cs in sorted(by.items(),
                        key=lambda kv: -np.mean([c[f"{prefix}_maxdets"] for c in kv[1]])):
        rows.append([DET_LABEL.get(d, d)]
                    + [f"{np.mean([c[f'{prefix}_{k}'] for c in cs]):.4f}"
                       for k in ("lo", "mi", "hi")]
                    + [f"**{np.mean([c[f'{prefix}_maxdets'] for c in cs]):.4f}**"])
    return table(["detector", "LO", "MI", "HI", "macro"], rows) if rows else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/INTERVALS.md")
    args = ap.parse_args()
    P = []
    A = P.append

    A(f"""# LAOD v2 — LO / MI / HI interval breakdown

*Generated {datetime.now():%Y-%m-%d %H:%M}. Companion to `reports/RESULTS.md`,
which reports the macro only. Same runs, same corrected matcher, same
maxDets = 100 — this document just splits each per-threshold curve into the
three intervals the original paper reports.*

## The intervals

CAAP and SNAP share one set of intervals:

| | LO | MI | HI |
|---|---|---|---|
| CAAP (IoU) and SNAP (cosine) | 0.50, 0.55, 0.60 | 0.65, 0.70, 0.75, 0.80 | 0.85, 0.90, 0.95 |

They are disjoint and every threshold is attainable. Neither property is
automatic: an earlier implementation built these with `np.arange`, where
`(0.65 - 0.50) / 0.05` evaluates to `3.0000000000000004` and the extra element
put 0.65 in both `LO` and `MI` while `HI` ran to 1.00 — a threshold no
predicted box attains, contributing ~0.0001 and dragging the interval down.

**The macro is not the mean of the three intervals.** It is the unweighted
mean over the ten thresholds 0.50 … 0.95, which weights `MI`'s four values
differently from a mean-of-means. Averaging the three interval figures instead
gives a visibly different number.

**CAAP and SNAP use the same intervals** — disjoint and attainable — with
**Sentence-BERT** (`all-MiniLM-L6-v2`) as SNAP's text encoder. Dropping
tau = 1.00 matters most for SNAP: it demands near-identical embeddings, scores
~0.03 against 0.19–0.24 for the rest of `HI`, and was deflating every
`SNAP_HI` by about 22%.

> **All three SNAP intervals are informative.** Under CLIP they were not: a
> label-shuffle control — identical boxes and scores, with the predicted label
> strings permuted dataset-wide — reproduced `SNAP_LO` *bit-identically*,
> because CLIP's unrelated-label cosine floor of +0.567 put every threshold
> below the point where the semantic test binds. Switching to Sentence-BERT
> moves `SNAP_LO`'s gain over chance from +0.0000 to +0.21–0.36: vacuous in 28
> of 28 Stage 1 cells, informative in 28 of 28. `reports/SNAP_SBERT.md` has the
> per-encoder evidence.
""")

    for name, path, note in (
            ("Stage 1 — COCO-Val (4,500 images)", "main_coco.csv", "stage1"),
            ("Stage 3 — LVIS-minival (4,327 images)", "main_lvis.csv", "stage3"),
            ("Stage 4 — COCO-OOD (438 images)", "main_coco_ood.csv", "stage4")):
        cells = rd(f"{R}/{path}")
        A(f"## {name}\n")
        if not cells:
            A(f"*(`{R}/{path}` not present)*\n")
            continue
        ds = cells[0]["dataset"]
        cells = merge_sbert(cells, ds, note)
        A("**CAAP by interval, mean over LLMs**\n")
        A(interval_means(cells, "caap") + "\n")
        if not has_semantic_labels(ds):
            A("""**SNAP and SemLoc-AP are not defined for COCO-OOD.** Every object
is annotated under the single label `unknow object`, so there is nothing to
name: the shuffled-label control gives a gain of exactly +0.0000 on all 28
cells, under any encoder. The scorers decline to emit the columns rather than
filling them with zeros. This dataset is read on U-AP.\n""")
        else:
            A("**SNAP by interval, mean over LLMs** (Sentence-BERT)\n")
            A(interval_means(cells, "snap") + "\n")
        sm = semloc_marginals(note)
        if sm and has_semantic_labels(ds):
            ious, taus, by_iou, by_tau, macro, n = sm
            A(f"**SemLoc-AP marginals, mean over {n} cells** "
              f"(macro {macro:.4f})\n")
            A(table(["axis"] + [f"{t:.2f}" for t in ious],
                    [["hold IoU, mean over τ"] + [f"{v:.4f}" for v in by_iou]]) + "\n")
            A(table(["axis"] + [f"{t:.2f}" for t in taus],
                    [["hold τ, mean over IoU"] + [f"{v:.4f}" for v in by_tau]]) + "\n")
            A("*SemLoc-AP has no LO/MI/HI structure -- it is a 10x10 grid rather "
              "than a sweep of one threshold -- so these marginals are the "
              "analogue: the first row shows what localisation strictness costs, "
              "the second what naming strictness costs. Full per-cell matrices "
              "are in `reports/SEMLOC.md` and "
              "`results/semloc_matrices.json`.*\n")
        A("<details><summary>Per-cell CAAP</summary>\n")
        A(cells_table(cells, "caap") + "\n")
        A("</details>\n")
        if has_semantic_labels(ds):
            A("<details><summary>Per-cell SNAP</summary>\n")
            A(cells_table(cells, "snap") + "\n")
            A("</details>\n")

    # ---- Stage 2 -----------------------------------------------------------
    s2 = rd(f"{REP}/stage2.json")
    A("## Stage 2 — Prompt ablation (500-image subset, YOLO-World)\n")
    if s2:
        sb2 = {(r["llm"], r["prompt"]): r
               for r in (rd(f"{R}/snap_sbert.csv") or []) if r["stage"] == "stage2"}
        rows = []
        for c in sorted(s2, key=lambda c: -c["caap"]):
            b = sb2.get((c["llm"], c["prompt"]))
            rows.append([LLM_LABEL.get(c["llm"], c["llm"]), c["prompt"],
                         f"{c['caap_lo']:.4f}", f"{c['caap_mi']:.4f}",
                         f"{c['caap_hi']:.4f}", f"**{c['caap']:.4f}**"]
                        + ([f"{b['snap_lo']:.4f}", f"{b['snap_mi']:.4f}",
                            f"{b['snap_hi']:.4f}", f"**{b['snap_macro']:.4f}**",
                            f"{b['gain_lo']:+.4f}"] if b else ["-"] * 5))
        A(table(["LLM", "prompt", "CAAP LO", "CAAP MI", "CAAP HI", "CAAP macro",
                 "SNAP LO", "SNAP MI", "SNAP HI", "SNAP macro", "SNAP gain LO"],
                rows) + "\n")
    else:
        A("*(not present)*\n")

    # ---- VLM ---------------------------------------------------------------
    A("## VLMs as direct detectors\n")
    for ds, title in (("coco", "COCO-Val test"), ("coco_ood", "COCO-OOD test")):
        v = rd(f"{R}/phase3_vlm_{ds}.csv")
        A(f"**{title}** — CAAP ranked by `logprob_bbox`, SNAP by "
          f"`logprob_label`; SNAP uses Sentence-BERT\n")
        if not v:
            A("*(not present)*\n")
            continue
        sbv = {r["llm"]: r for r in (rd(f"{R}/snap_sbert.csv") or [])
               if r["stage"] == "vlm" and r["dataset"] == ds}
        snap_col = has_semantic_labels(ds)
        rows = []
        for m in ("qwen25-vl-7b", "internvl3-8b"):
            c = next((r for r in v if r["model"] == m
                      and r["score_source"] == "logprob_bbox"), None)
            if not c:
                continue
            b = sbv.get(m)
            rows.append([m, f"{c['caap_lo']:.4f}", f"{c['caap_mi']:.4f}",
                         f"{c['caap_hi']:.4f}", f"**{c['caap']:.4f}**"]
                        + ([f"{b['snap_lo']:.4f}", f"{b['snap_mi']:.4f}",
                            f"{b['snap_hi']:.4f}", f"**{b['snap_macro']:.4f}**"]
                           if (snap_col and b) else []))
        hdr = ["model", "CAAP LO", "CAAP MI", "CAAP HI", "CAAP macro"] + (
            ["SNAP LO", "SNAP MI", "SNAP HI", "SNAP macro"] if snap_col else [])
        A(table(hdr, rows) + "\n")
        if not snap_col:
            A("*SNAP omitted: COCO-OOD annotates one label, so there is "
              "nothing to name.*\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(P))
    print(f"wrote {args.out} ({len(chr(10).join(P).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
