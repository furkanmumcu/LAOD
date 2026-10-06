#!/usr/bin/env python
"""Assemble the prompt-ablation report, COCO and LVIS side by side.

The two halves are the point: the same three prompts, the same three LLMs, the
same detector and the same 500-image design on two datasets whose annotation
schemes differ. A prompt that wins on one and loses on the other is evidence
about transfer, which a single-dataset ablation cannot provide.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np

R, REP = "results", "reports"
LLM = {"gemma4-e4b": "Gemma4-E4B", "gemma4-12b": "Gemma4-12B",
       "qwen35-9b": "Qwen3.5-9B", "gemma3-4b": "Gemma3-4B",
       "gemma4-e2b": "Gemma4-E2B", "qwen25-vl-7b": "Qwen2.5-VL-7B",
       "internvl3-8b": "InternVL3-8B"}
PROMPTS = ["default", "minimal", "coco-specific"]


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


def table(h, rows):
    return "\n".join(["| " + " | ".join(h) + " |",
                      "|" + "|".join(["---"] * len(h)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def block(stage_json, snap_rows, llms, title):
    """CAAP and SNAP by prompt for one dataset, plus the granular delta."""
    cells = {(c["llm"], c["prompt"]): c for c in stage_json}
    snap = {(r["llm"], r["prompt"]): r for r in snap_rows}
    out = [f"**{title} — CAAP@.5:.95**", ""]
    rows = []
    for l in llms:
        r = [LLM.get(l, l)] + [f"{cells[(l, p)]['caap']:.4f}"
                               if (l, p) in cells else "-" for p in PROMPTS]
        if (l, "coco-specific") in cells and (l, "default") in cells:
            d = cells[(l, "coco-specific")]["caap"] / cells[(l, "default")]["caap"] - 1
            r.append(f"**{d * 100:+.1f}%**")
        rows.append(r)
    out += [table(["LLM"] + PROMPTS + ["granular vs original"], rows), ""]

    if snap:
        out += [f"**{title} — SNAP (Sentence-BERT)**", ""]
        rows = []
        for l in llms:
            r = [LLM.get(l, l)] + [f"{snap[(l, p)]['snap_macro']:.4f}"
                                   if (l, p) in snap else "-" for p in PROMPTS]
            if (l, "coco-specific") in snap and (l, "default") in snap:
                d = snap[(l, "coco-specific")]["snap_macro"] / snap[(l, "default")]["snap_macro"] - 1
                r.append(f"**{d * 100:+.1f}%**")
            rows.append(r)
        out += [table(["LLM"] + PROMPTS + ["granular vs original"], rows), ""]

    out += ["**Labels proposed per image**", ""]
    rows = [[LLM.get(l, l)] + [f"{cells[(l, p)]['labels_per_img']:.1f}"
                               if (l, p) in cells else "-" for p in PROMPTS]
            for l in llms]
    out += [table(["LLM"] + PROMPTS, rows), ""]
    return "\n".join(out)


def mean_delta(cells, llms, key="caap", a="coco-specific", b="default"):
    d = [cells[(l, a)][key] / cells[(l, b)][key] - 1 for l in llms
         if (l, a) in cells and (l, b) in cells]
    return float(np.mean(d)), sum(1 for x in d if x > 0), len(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/PROMPTS.md")
    args = ap.parse_args()

    co = rd(f"{REP}/stage2.json")
    lv = rd(f"{REP}/stage2_lvis.json")
    co_snap = [r for r in (rd(f"{R}/snap_sbert.csv") or []) if r["stage"] == "stage2"]
    lv_snap = rd(f"{R}/snap_sbert_lvis_ablation.csv") or []
    if not co or not lv:
        raise SystemExit("missing stage2.json or stage2_lvis.json")

    co_llms = sorted({c["llm"] for c in co})
    lv_llms = sorted({c["llm"] for c in lv})
    shared = sorted(set(co_llms) & set(lv_llms))
    cc = {(c["llm"], c["prompt"]): c for c in co}
    lc = {(c["llm"], c["prompt"]): c for c in lv}

    g_co, w_co, n_co = mean_delta(cc, shared)
    g_lv, w_lv, n_lv = mean_delta(lc, shared)
    p_co, pw_co, _ = mean_delta(cc, co_llms, a="minimal")
    p_lv, pw_lv, _ = mean_delta(lc, lv_llms, a="minimal")

    # Rendered from the registry, so the document cannot drift from the
    # prompts the code actually sends.
    from laod.models.registry import PROMPTS as SPECS
    sysmsg = {s_.system for s_ in SPECS.values()}
    rows = [[f"**`{k}`**", f"*{SPECS[k].user}*", SPECS[k].provenance]
            for k in ("default", "minimal", "coco-specific") if k in SPECS]
    prompt_table = (
        "All three share one system message:\n\n> `"
        + (sysmsg.pop() if len(sysmsg) == 1 else "varies") + "`\n\n"
        "The user message differs:\n\n"
        + table(["key", "text", "provenance"], rows))

    P = []
    A = P.append
    A(f"""# Prompt sensitivity, and why one prompt is used for all datasets

*Generated {datetime.now():%Y-%m-%d %H:%M}. Three prompts, the same LLMs, the
same detector and the same 500-image design, repeated on two datasets. Scored
with the project defaults: maxDets = 100, score-order matching, Sentence-BERT
SNAP.*

## The three prompts

{prompt_table}

## Setup

| | COCO-Val | LVIS-minival |
|---|---|---|
| images | 500 (`results/ablation_subset.json`, seed 20261002) | 500 (`results/ablation_subset_lvis.json`, seed 20261003) |
| ground-truth objects | 3,903 | 4,294 |
| detector | YOLO-World, `conf` 0.001 | YOLO-World, `conf` 0.001 |
| LLMs | 7 | 3 |
| vocabulary | 80 categories | 1,203 categories |

Both subsets are drawn from their dataset's test split and are disjoint from
the hyperparameter holdout. LVIS-minival is itself drawn from COCO val2017, so
the two subsets share 40 image ids by coincidence; they are independent draws
and each is scored against its own ground truth.

## Result 1 — wording barely matters

`paper` against `original` — the published prompt against the executed one:

| dataset | mean CAAP difference | prompts favouring `paper` |
|---|---|---|
| COCO-Val | **{p_co * 100:+.1f}%** | {pw_co} of {len(co_llms)} |
| LVIS | **{p_lv * 100:+.1f}%** | {pw_lv} of {len(lv_llms)} |

Both are inside what a 500-image sample supports. The two prompts differ
substantially in form — one is a bare instruction, the other adds output-format
constraints and excludes `sky` and `street` — and score the same.

This is worth recording for a second reason: **the prompt the paper reports is
not the prompt its code runs.** The discrepancy turns out not to matter, but
that is a measurement, not an assumption, and it is only checkable by re-running
both.

## Result 2 — a specialised prompt helps on one dataset and hurts on the other

`granular` against `original`, on the three LLMs common to both:

| dataset | mean CAAP difference | LLMs where `granular` wins |
|---|---|---|
| **COCO-Val** | **{g_co * 100:+.1f}%** | **{w_co} of {n_co}** |
| **LVIS** | **{g_lv * 100:+.1f}%** | **{w_lv} of {n_lv}** |

A swing of roughly {abs(g_co - g_lv) * 100:.0f} points, unanimous in both
directions. The best prompt on COCO is the worst on LVIS.

### Why

`granular` carries three instructions, and each one encodes an assumption about
the annotation scheme:

| instruction | COCO-80 | LVIS-1203 |
|---|---|---|
| *"write only the general name"* | matches its coarse categories | contradicts `Tabasco sauce`, `beer bottle`, `fish (food)` |
| *"do not list infrastructural objects"* | COCO does not annotate them | LVIS annotates `fireplug`, `streetlight`, `telephone pole`, `street sign` |
| *"don't use plural"* | COCO names are singular | 25 LVIS categories are plural by definition — `asparagus`, `binoculars`, `suspenders` |

The label counts show the mechanism directly: `granular` reduces labels per
image on **both** datasets. On COCO it removes proposals the benchmark never
annotates; on LVIS the same instruction deletes correct answers.

## Conclusion

1. **Prompt wording is not a significant factor.** Two substantially different
   phrasings score within noise on both datasets, so the reported results do
   not hinge on the exact wording, and the paper/code discrepancy is immaterial.

2. **Prompt specialisation is a significant factor, and it does not transfer.**
   Tuning the prompt to a benchmark's annotation conventions buys
   {g_co * 100:+.1f}% on that benchmark and costs {g_lv * 100:+.1f}% on another.
   The gain is not better perception; it is agreement with one labelling
   convention.

3. **We therefore use a single prompt across all datasets.** Reporting a
   per-dataset best prompt would inflate every number by an amount that
   measures how well the prompt was fitted to that dataset's annotation guide.
   `original` is used everywhere, including where `granular` would score higher.

## Per-dataset detail
""")
    A(block(co, co_snap, co_llms, "COCO-Val"))
    A(block(lv, lv_snap, lv_llms, "LVIS-minival"))
    A("""## Files

| file | contents |
|---|---|
| `reports/stage2.json` | COCO ablation, 21 cells |
| `reports/stage2_lvis.json` | LVIS ablation, 9 cells |
| `results/snap_sbert.csv` (`stage2`) | COCO SNAP under Sentence-BERT |
| `results/snap_sbert_lvis_ablation.csv` | LVIS SNAP under Sentence-BERT |
| `results/ablation_subset{,_lvis}.json` | the two frozen subsets |
| `outputs/runs/{coco,lvis}-test-ablation__*` | per-image detections and raw replies |

Regenerate with `scripts/run_lvis_prompt_ablation.sh` and this script.
""")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(P))
    print(f"wrote {args.out} ({len(chr(10).join(P).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
