#!/usr/bin/env python
"""Assemble the Sentence-BERT SNAP report.

Separate from RESULTS.md and INTERVALS.md, which keep CLIP: this is a proposed
replacement metric, not a correction to the reported one, and the two should be
readable side by side before anything is switched.
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

R = "results"
LLM_LABEL = {"gemma3-4b": "Gemma3-4B", "gemma4-e2b": "Gemma4-E2B",
             "gemma4-e4b": "Gemma4-E4B", "gemma4-12b": "Gemma4-12B",
             "qwen25-vl-7b": "Qwen2.5-VL-7B", "qwen35-9b": "Qwen3.5-9B",
             "internvl3-8b": "InternVL3-8B"}
DET_LABEL = {"yolo-world": "YOLO-World", "gdino-tiny": "G-DINO-T",
             "gdino-base": "G-DINO-B", "owlv2-base": "OWLv2", "none": "—"}


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


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def verdict(g: float) -> str:
    return "vacuous" if g < 0.01 else ("weak" if g < 0.05 else "informative")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/SNAP_SBERT.md")
    args = ap.parse_args()
    sb = rd(f"{R}/snap_sbert.csv")
    if not sb:
        raise SystemExit("results/snap_sbert.csv not found")
    diag = rd(f"{R}/encoder_diagnostic.json")
    cmp_ = rd(f"{R}/snap_encoder_compare.json")
    clip = {}
    for f in ("main_coco.csv", "main_lvis.csv", "main_coco_ood.csv"):
        for r in (rd(f"{R}/{f}") or []):
            clip[(r["llm"], r["detector"], r["dataset"])] = r

    P = []
    A = P.append
    enc = sb[0]["encoder"]
    A(f"""# SNAP with Sentence-BERT

*Generated {datetime.now():%Y-%m-%d %H:%M}. Encoder `{enc}`, bare labels (no
prompt template). Same runs, same corrected matcher, same maxDets = 100 as
`RESULTS.md` — the only thing that changes is the text encoder.*

**This is a proposal, not a correction.** `RESULTS.md` and `INTERVALS.md` still
report CLIP, which is what the original paper used and what keeps the published
SNAP row comparable. Nothing there has been changed.

## The problem this addresses

SNAP matches a predicted label to a ground-truth label when their cosine
similarity clears a threshold. That only works if *unrelated* labels score low.
CLIP's text embeddings occupy a narrow cone — they were trained to align text
with images, not text with text, so text-text cosine is incidental to the
objective — and unrelated labels therefore start high.

Measured over every distinct pair of COCO's 80 ground-truth labels:
""")
    if diag:
        e = diag["encoders"]
        taus = ["0.50", "0.60", "0.85"]
        rows = []
        for name in ("clip-raw", "clip-centred", "minilm", "minilm-photo",
                     "mpnet", "mpnet-photo"):
            if name not in e:
                continue
            s = e[name]
            rows.append([f"`{name}`", f"{s['min']:+.3f}", f"{s['median']:+.3f}",
                         f"{s['max']:+.3f}"]
                        + [f"{s['admits'][t]*100:.0f}%" for t in taus])
        A(table(["encoder", "min", "median", "max"]
                + [f"admits @{t}" for t in taus], rows))
        A(f"""
"admits" is the fraction of **unrelated** pairs a threshold would accept. At
tau = 0.50 — inside both `SNAP_LO` and `SNAP_MI` — CLIP accepts **100%**: the
semantic test never fires, and SNAP degenerates into "a confident box landed in
an image that still had an unmatched ground truth". MiniLM accepts **2%**.

Note that `"a photo of {{}}"` *hurts* MiniLM, raising its median from +0.27 to
+0.39. The template is shared text in every string, so it injects the very
common component the cone problem is made of. It helps CLIP only because it
matches CLIP's training distribution. Bare labels are used throughout here.
""")

    # ---- the control ------------------------------------------------------
    A("""## Does it actually fix the low thresholds?

A SNAP value cannot answer this on its own, and values are not comparable
across encoders — the same tau is a different constraint in a different
geometry. What is comparable is the **gain over a label-shuffle control**:
identical boxes and scores, predicted label strings permuted dataset-wide. A
threshold carrying naming information scores above its shuffled twin; one that
does not, does not.
""")
    if cmp_:
        rows = []
        for cell, encs in cmp_["cells"].items():
            for name in ("clip-raw", "minilm"):
                if name not in encs:
                    continue
                d = encs[name]
                rows.append([cell if name == "clip-raw" else "", f"`{name}`"]
                            + [f"{d['snap'][k] - d['chance'][k]:+.4f}"
                               for k in ("LO", "MI", "HI")]
                            + [verdict(d["snap"]["LO"] - d["chance"]["LO"])])
        A("Gain over chance, three representative cells:\n")
        A(table(["cell", "encoder", "LO", "MI", "HI", "LO verdict"], rows) + "\n")

    # ---- full results -----------------------------------------------------
    by_stage = collections.defaultdict(list)
    for r in sb:
        by_stage[r["stage"]].append(r)

    TITLES = {"stage1": "Stage 1 — COCO-Val (4,500 images)",
              "stage2": "Stage 2 — Prompt ablation (500-image subset)",
              "stage3": "Stage 3 — LVIS-minival (4,327 images)",
              "stage4": "Stage 4 — COCO-OOD (438 images)",
              "vlm": "VLMs as direct detectors"}

    A("## Results\n")
    A("""Every cell below is scored with the shuffle control. `gain` columns are
SNAP minus chance; a threshold is only measuring naming to the extent its gain
is above zero.
""")
    for stage in ("stage1", "stage2", "stage3", "stage4", "vlm"):
        rows_s = by_stage.get(stage)
        if not rows_s:
            continue
        A(f"### {TITLES[stage]}\n")
        if stage in ("stage1", "stage3", "stage4"):
            by_det = collections.defaultdict(list)
            for r in rows_s:
                by_det[r["detector"]].append(r)
            rows = []
            for d, cs in sorted(by_det.items(),
                                key=lambda kv: -np.mean([c["snap_macro"] for c in kv[1]])):
                m = lambda k: np.mean([c[k] for c in cs])
                ds = rows_s[0]["dataset"]
                old = [clip.get((c["llm"], d, ds), {}).get("snap_maxdets")
                       for c in cs]
                old = [o for o in old if o is not None]
                rows.append([DET_LABEL.get(d, d),
                             f"{m('snap_lo'):.4f}", f"{m('snap_mi'):.4f}",
                             f"{m('snap_hi'):.4f}", f"**{m('snap_macro'):.4f}**",
                             f"{m('gain_lo'):+.4f}", f"{m('gain_mi'):+.4f}",
                             f"{m('gain_hi'):+.4f}",
                             f"{np.mean(old):.4f}" if old else "-"])
            A("**Detector means**\n")
            A(table(["detector", "LO", "MI", "HI", "macro",
                     "gain LO", "gain MI", "gain HI", "CLIP macro"], rows) + "\n")
        A("<details><summary>Per-cell</summary>\n")
        key = "prompt" if stage == "stage2" else "detector"
        rows = []
        for r in sorted(rows_s, key=lambda r: -r["snap_macro"]):
            label = (r["prompt"] if stage == "stage2"
                     else DET_LABEL.get(r["detector"], r["detector"]))
            rows.append([LLM_LABEL.get(r["llm"], r["llm"]), label,
                         f"{r['snap_lo']:.4f}", f"{r['snap_mi']:.4f}",
                         f"{r['snap_hi']:.4f}", f"**{r['snap_macro']:.4f}**",
                         f"{r['gain_lo']:+.4f}", f"{r['gain_mi']:+.4f}",
                         f"{r['gain_hi']:+.4f}",
                         f"{r['dataset']}" if stage == "vlm" else ""])
        hdr = ["LLM", key, "LO", "MI", "HI", "macro",
               "gain LO", "gain MI", "gain HI"] + (["dataset"] if stage == "vlm" else [""])
        A(table(hdr, rows) + "\n")
        A("</details>\n")

    # ---- summary ----------------------------------------------------------
    s1 = by_stage.get("stage1", [])
    if s1:
        vac = sum(1 for r in s1 if r["gain_lo"] < 0.01)
        A(f"""## What this changes

Across the {len(s1)} Stage 1 cells, `SNAP_LO` is informative in
**{len(s1) - vac} of {len(s1)}** under MiniLM. Under CLIP it is vacuous in all
of them — a shuffled copy of the labels reproduces the score.

Two consequences worth weighing before switching:

1. **The absolute numbers drop sharply.** `SNAP_LO` falls from roughly 0.75
   to roughly 0.23. That is the vacuous inflation coming out, not a regression:
   the CLIP figure was mostly counting detections, not naming.
2. **Comparability with the published SNAP row is lost.** The original paper's
   0.54 / 0.52 / 0.19 is a CLIP number on a CLIP geometry. If MiniLM becomes
   the reported metric, CLIP has to stay alongside it as the legacy encoder —
   the same arrangement already used for the threshold grids.

## Files

| file | contents |
|---|---|
| `results/snap_sbert.csv` | every cell: LO/MI/HI/macro, chance, and gain |
| `results/encoder_diagnostic.json` | unrelated-pair cosine distribution per encoder |
| `results/snap_encoder_compare.json` | four encoders on three cells, per-tau |
""")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(P))
    print(f"wrote {args.out} ({len(chr(10).join(P).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
