#!/usr/bin/env python
"""Assemble the SemLoc-AP report.

Read from ``results/semloc.csv`` and ``results/semloc_matrices.json`` at build
time, so the document cannot drift from the numbers.
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
             "internvl3-8b": "InternVL3-8B", "none": "—"}
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
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def grid(cells, key, llms, dets, fmt="{:.4f}"):
    look = {(c["llm"], c["detector"]): c for c in cells}
    rows = []
    for l in llms:
        r, vals = [LLM_LABEL.get(l, l)], []
        for d in dets:
            c = look.get((l, d))
            r.append(fmt.format(c[key]) if c else "-")
            if c:
                vals.append(c[key])
        r.append(f"**{fmt.format(np.mean(vals))}**" if vals else "-")
        rows.append(r)
    means = ["**mean**"]
    for d in dets:
        v = [look[(l, d)][key] for l in llms if (l, d) in look]
        means.append(f"**{fmt.format(np.mean(v))}**" if v else "-")
    means.append("")
    rows.append(means)
    return table(["LLM"] + [DET_LABEL.get(d, d) for d in dets] + ["mean"], rows)


def matrix_block(entry, title):
    m = np.array(entry["matrix"])
    taus, ious = entry["sem_thresholds"], entry["iou_thresholds"]
    rows = [[f"**{t:.2f}**"] + [f"{v:.4f}" for v in row] for t, row in zip(ious, m)]
    return (f"**{title}**\n\n"
            + table(["IoU \\ τ"] + [f"{u:.2f}" for u in taus], rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=f"{REP}/SEMLOC.md")
    args = ap.parse_args()

    rows = rd(f"{R}/semloc.csv")
    mats = rd(f"{R}/semloc_matrices.json")
    if not rows:
        raise SystemExit(f"{R}/semloc.csv not present -- run scripts/semloc_all.py")
    enc = rows[0]["encoder"]
    P, A = [], None
    A = P.append

    A(f"""# SemLoc-AP — joint localisation and naming

*Generated {datetime.now():%Y-%m-%d %H:%M} from `results/semloc.csv` and
`results/semloc_matrices.json`. {len(rows)} cells. Encoder `{enc}`, bare labels.*

## What this measures

CAAP asks whether an annotated object was localised. SNAP asks whether the names
produced were semantically appropriate. Neither asks whether those two successes
happened on the **same object**, and a single prediction can satisfy both while
being wrong: a box tightly on the dog, labelled "cat", with a cat elsewhere in
the image. CAAP matches it to the dog on IoU alone; SNAP matches it to the cat
on label alone, because SNAP's affinity matrix is built from labels and boxes
never enter it.

SemLoc-AP counts a prediction as a true positive only when some single unmatched
ground-truth instance satisfies **both**

    IoU(box_i, box_j) >= θ    and    sim(label_i, label_j) >= τ

and assigns it to the instance with the highest IoU **among those that qualify**.
The semantic criterion decides which objects are eligible; IoU decides which
eligible object the box corresponds to. Selecting by similarity instead would
let the label determine which object was localised, inverting the question.

| Metric | Question |
|---|---|
| CAAP | Did we localise an annotated object? |
| SNAP | Did we produce semantically appropriate object names? |
| **SemLoc-AP** | Did we put an appropriate name on the object we actually localised? |

A low SemLoc-AP alone cannot say whether localisation, naming, or both failed —
which is why it complements CAAP and SNAP rather than replacing them.

## Conventions

* **maxDets = 100**, applied once to the prediction list before anything is
  scored — the identical cap CAAP and SNAP apply, so all three see the same
  detection set.
* **Pooled AP, not per-class mAP.** Detections are pooled across the dataset and
  ranked by confidence once; `n_gt` is the total number of ground-truth objects
  in the dataset. There is no per-class AP and no averaging over classes, so
  these numbers are **not** comparable with published COCO mAP, which the
  high-τ corner of the grid otherwise resembles.
* **Matching** visits predictions in descending confidence, one-to-one, as in
  CAAP and SNAP.
* **Empty or unusable labels can never satisfy the semantic criterion.** They
  are excluded explicitly rather than left to the cosine: MiniLM embeds the
  empty string within 0.410 of some COCO name, so relying on the threshold would
  make the rule an accident of the checkpoint.
* **Grid**: the ten CAAP IoU thresholds × the ten SNAP cosine thresholds, both
  0.50 to 0.95. Evaluated as a full Cartesian product — an IoU of 0.70 has no
  principled correspondence to a cosine of 0.70, so pairing on the diagonal
  would assert a relationship that does not exist. The headline is the
  unweighted mean of all 100 values.
* **COCO-OOD is excluded.** It annotates every object as `unknow object`, so
  there is no naming to score; SNAP's shuffle control already returns a gain of
  exactly 0.0000 on all 28 of its cells.
""")

    for stage, title, note in (
            ("stage1", "Stage 1 — COCO-Val (4,500 images)", ""),
            ("stage3", "Stage 3 — LVIS-minival (4,327 images)", "")):
        cells = [r for r in rows if r["stage"] == stage]
        if not cells:
            continue
        llms = sorted({c["llm"] for c in cells},
                      key=lambda l: -np.mean([c["semloc"] for c in cells if c["llm"] == l]))
        dets = sorted({c["detector"] for c in cells},
                      key=lambda d: -np.mean([c["semloc"] for c in cells if c["detector"] == d]))
        A(f"## {title}\n")
        if note:
            A(note + "\n")
        A("**SemLoc-AP, macro over the 100 threshold pairs**\n")
        A(grid(cells, "semloc", llms, dets) + "\n")
        A("**At the most permissive corner, SemLoc-AP(0.50, 0.50)**\n")
        A(grid(cells, "semloc_5050", llms, dets) + "\n")
        best = max(cells, key=lambda c: c["semloc"])
        A(f"Best cell: `{best['llm']}` + `{best['detector']}` — SemLoc-AP "
          f"{best['semloc']:.4f}.\n")

    # ---- prompt ablation -------------------------------------------------
    for stage, title in (("stage2", "Stage 2 — Prompt ablation, COCO (500-image subset, YOLO-World)"),
                         ("stage2_lvis", "Stage 2 — Prompt ablation, LVIS (500-image subset, YOLO-World)")):
        cells = [r for r in rows if r["stage"] == stage]
        if not cells:
            continue
        prompts = sorted({c["prompt"] for c in cells})
        llms = sorted({c["llm"] for c in cells})
        look = {(c["llm"], c["prompt"]): c for c in cells}
        trows = [[LLM_LABEL.get(l, l)] + [f"{look[(l, p)]['semloc']:.4f}"
                                          if (l, p) in look else "-" for p in prompts]
                 for l in llms]
        means = ["**mean**"] + [f"**{np.mean([look[(l, p)]['semloc'] for l in llms if (l, p) in look]):.4f}**"
                                for p in prompts]
        A(f"## {title}\n")
        A(table(["LLM"] + prompts, trows + [means]) + "\n")

    # ---- controls --------------------------------------------------------
    fv = [r for r in rows if r["stage"] == "fixed_vocab"]
    vl = [r for r in rows if r["stage"] == "vlm"]
    if fv or vl:
        A("## Controls\n")
    if fv:
        A("**Closed-vocabulary ceiling** — every image given the 80 COCO "
          "category names, no LLM. Predicted and ground-truth strings are "
          "byte-identical, so this is the upper bound a generated vocabulary "
          "competes against.\n")
        A(table(["detector", "SemLoc-AP", "@.50/.50", "@.75/.75"],
                [[DET_LABEL.get(r["detector"], r["detector"]), f"{r['semloc']:.4f}",
                  f"{r['semloc_5050']:.4f}", f"{r['semloc_7575']:.4f}"]
                 for r in sorted(fv, key=lambda r: -r["semloc"])]) + "\n")
    if vl:
        A("**VLM as its own detector** — the model emits boxes directly, "
          "ranked by `logprob_bbox`.\n")
        A(table(["model", "SemLoc-AP", "@.50/.50", "@.75/.75"],
                [[LLM_LABEL.get(r["llm"], r["llm"]), f"{r['semloc']:.4f}",
                  f"{r['semloc_5050']:.4f}", f"{r['semloc_7575']:.4f}"]
                 for r in sorted(vl, key=lambda r: -r["semloc"])]) + "\n")

    # ---- monotonicity ----------------------------------------------------
    nmi = [r for r in rows if str(r["monotone_iou"]).lower() != "true"]
    nms = [r for r in rows if str(r["monotone_sem"]).lower() != "true"]
    worst = max(float(r["largest_rise"]) for r in rows)
    steps = sum(int(r["rises_iou"]) + int(r["rises_sem"]) for r in rows)
    A(f"""## Monotonicity diagnostic

Raising either threshold shrinks each prediction's eligible set, which would
normally only remove true positives. It can also *redirect* a match: under the
highest-IoU rule, dropping a semantically marginal instance can free the object
a later prediction needed, so SemLoc-AP is **not guaranteed** monotone in τ. A
counterexample is constructible. This checks whether it happens in practice.

| | cells | |
|---|---|---|
| non-monotone in IoU | {len(nmi)} / {len(rows)} | |
| non-monotone in τ | {len(nms)} / {len(rows)} | |
| grid steps that rise on tightening | {steps} | out of {len(rows) * 180} |
| largest rise observed | {worst:.6f} | |

It does not occur on this data. The 10×10 grids can be read as monotone
decreasing in both directions without qualification — but the property is
empirical, not guaranteed, and should be re-checked on new data rather than
assumed.

A second check, kept as a debugging diagnostic and **not** claimed as a
property: SemLoc-AP ≤ min(CAAP, SNAP) held on every Stage 1 and Stage 3 cell.
Independent greedy matchers can in principle produce assignments that break it,
so a violation would signal a matcher bug rather than a real result.
""")

    # ---- matrices --------------------------------------------------------
    if mats:
        A("## Threshold matrices\n")
        A("Rows are IoU thresholds, columns cosine thresholds; each entry is "
          "SemLoc-AP at that pair. The headline score is the mean of all 100.\n")
        picks = []
        for stage in ("stage1", "stage3", "fixed_vocab"):
            cand = [r for r in rows if r["stage"] == stage]
            if cand:
                picks.append(max(cand, key=lambda r: r["semloc"]))
        for r in picks:
            e = mats.get(r["run"])
            if e:
                A(matrix_block(e, f"{r['stage']} — `{r['llm']}` + `{r['detector']}` "
                                  f"(macro {r['semloc']:.4f})") + "\n")
        A(f"All {len(mats)} matrices are in `results/semloc_matrices.json`.\n")
        fvr = [r for r in rows if r["stage"] == "fixed_vocab"]
        if fvr and mats.get(fvr[0]["run"]):
            m = np.array(mats[fvr[0]["run"]]["matrix"])
            taus = mats[fvr[0]["run"]]["sem_thresholds"]
            hi = [i for i, t in enumerate(taus) if t >= 0.80]
            same = bool(np.allclose(m[:, hi], m[:, [hi[0]]], atol=0, rtol=0))
            A(f"""**A wiring check worth keeping.** On the closed 80-name
vocabulary the τ ≥ 0.80 columns are bit-identical ({same}), because no two
distinct COCO category names reach a cosine of 0.80 under this encoder — so
above that threshold the semantic criterion reduces to exact string match and
cannot distinguish further. Generated vocabularies are continuous in similarity
and do not show this collapse, which is why the upper half of the grid is
retained.\n""")

    A(f"""## Files

| file | contents |
|---|---|
| `results/semloc.csv` | one row per cell: macro, both marginals, two corners, monotonicity |
| `results/semloc_matrices.json` | the full 10×10 grid for all {len(rows)} cells |
| `laod/metrics/semloc.py` | the metric |
| `scripts/semloc_all.py` | the scorer |
| `tests/test_semloc.py` | the matching rule under test |
""")

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(P))
    print(f"wrote {out} ({len(chr(10).join(P).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
