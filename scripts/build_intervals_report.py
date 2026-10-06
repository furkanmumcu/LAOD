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

The original implementation builds its thresholds with `np.arange`, and the
result is not what its prose describes. Both grids are reproduced exactly as
that code computed them, because the published table only regenerates this way:

| | LO | MI | HI |
|---|---|---|---|
| **reported** (CAAP and SNAP) | 0.50, 0.55, 0.60 | 0.65, 0.70, 0.75, 0.80 | 0.85, 0.90, 0.95 |
| *original implementation* | 0.50, 0.55, 0.60, 0.65 | 0.65, 0.70, 0.75, 0.80 | 0.85, 0.90, 0.95, 1.00 |

The original grid is built with `np.arange` and carries three defects, all
fixed in the reported one:

1. **`LO` holds four values, not the three the paper describes.**
   `(0.65 - 0.50) / 0.05` evaluates to `3.0000000000000004` in floating point,
   so `arange` emits an extra element.
2. **`LO` and `MI` overlap at 0.65**, which is therefore counted twice.
3. **`HI` includes IoU = 1.00**, which no predicted box attains — it
   contributes ~0.0001 and drags the interval down.

**The macro is not the mean of the three intervals.** It is the unweighted mean
over the ten thresholds 0.50 … 0.95, excluding 1.00 even though `HI` contains
it. On the archived original predictions: LO 0.2507, MI 0.2193, HI 0.0829, and
macro **0.1970** — against 0.1843 if you averaged the three intervals instead.

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
        if ds == "coco_ood":
            A("""**SNAP is not reported for COCO-OOD.** Every object is annotated
under the single label `unknow object`, so there is nothing to name: the
shuffled-label control gives a gain of exactly +0.0000 on all 28 cells, under
any encoder. This dataset is read on U-AP.\n""")
        else:
            A("**SNAP by interval, mean over LLMs** (Sentence-BERT)\n")
            A(interval_means(cells, "snap") + "\n")
        A("<details><summary>Per-cell CAAP</summary>\n")
        A(cells_table(cells, "caap") + "\n")
        A("</details>\n")
        if ds != "coco_ood":
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
        snap_col = ds != "coco_ood"
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

    # ---- the published anchor ---------------------------------------------
    A("""## Against the published table

The original paper's COCO-Val row regenerates exactly from its own archived
predictions, which is what licenses every comparison above:

| interval | computed | published | |
|---|---|---|---|
| `CAAP_LO` | 0.2507 | **0.25** | match |
| `CAAP_MI` | 0.2193 | **0.22** | match |
| `CAAP_HI` | 0.0829 | **0.08** | match |
| `SNAP_LO` | 0.5485 | 0.54 | +0.008 |
| `SNAP_MI` | 0.5282 | 0.52 | +0.008 |
| `SNAP_HI` | 0.1881 | **0.19** | match |

`SNAP_LO` and `SNAP_MI` land one rounding step high and the cause is not
explained. Ruled out: the CLIP checkpoint, and fp16-vs-fp32 arithmetic (fp16
moves `HI` in the wrong direction). Recorded rather than rationalised.

Those figures use the **legacy** grid, uncapped, with the original match order
— the configuration the published numbers were produced under. Applying
maxDets = 100 moves `CAAP_MI` by 1e-4 (two images exceed 100 detections) and
nothing else.
""")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(P))
    print(f"wrote {args.out} ({len(chr(10).join(P).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
