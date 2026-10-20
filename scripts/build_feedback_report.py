#!/usr/bin/env python
"""Assemble the vocabulary-feedback report from results/feedback_dev/."""

from __future__ import annotations

import argparse, csv, os, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

CONDS = ("baseline", "selfreflect", "feedback_refine", "feedback_full")
LABEL = {"baseline": "A · baseline", "selfreflect": "B · self-reflection",
         "feedback_refine": "C · feedback-refine", "feedback_full": "D · feedback-full"}
CFGS = ("qwen35-9b+yolo-world", "gemma4-e4b+yolo-world", "qwen35-9b+owlv2-base")


def rd(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def tbl(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "|".join(["---"] * len(head)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/feedback_dev")
    ap.add_argument("--out", default="reports/FEEDBACK.md")
    a = ap.parse_args()
    D = Path(a.indir)
    met = {(r["config"], r["condition"]): r for r in rd(D / "metrics.csv")}
    voc = {(r["config"], r["condition"]): r for r in rd(D / "vocabulary.csv")}
    cnf = {(r["config"], r["condition"]): r for r in rd(D / "confidence_usage.csv")}
    rec = {(r["config"], r["condition"]): r for r in rd(D / "recovery.csv")
           if float(r["tau"]) == 0.60}
    if not met:
        raise SystemExit(f"no metrics in {D}")

    P = []; A = P.append
    A(f"""# Vocabulary-level MLLM–OVOD feedback — development results

*Generated {datetime.now():%Y-%m-%d %H:%M}. 500-image COCO **tuning holdout**
(`results/tuning_split.json`), which is disjoint from the 4,500-image test
split. Read from `{D}/` at build time.*

> **Outcome: the loop does not work.** Of nine iterative cells, CAAP falls in
> all nine; SNAP and SemLoc-AP rise in two. The best result is +2.5% SNAP with
> CAAP flat. The 4,500-image evaluation was **not** run — the development phase
> answered the question.

## What was tested

> Can feedback from the OVOD help the MLLM revise its generated vocabulary,
> reducing unsupported concepts and recovering concepts missed in the first
> round?

One revision round, `C0 → D0 → C1 → D1`, with the components keeping their
roles: the MLLM generates and revises **concepts**, the detector does all
**localisation**. The MLLM never sees or emits a coordinate. D1 is the sole
output of each iterative condition — D0 and D1 are never merged.

| condition | sees the image | sees C0 | sees detector feedback | may add concepts |
|---|---|---|---|---|
| **A** baseline | — (one pass) | — | — | — |
| **B** self-reflection | yes | yes | no | yes |
| **C** feedback-refine | yes | yes | yes | **no** |
| **D** feedback-full | yes | yes | yes | yes |

Feedback is per-concept detection count and maximum confidence, built from every
detection above the detector's own threshold (before the maxDets cap, which is
an evaluation rule and would misreport a concept as unsupported merely because
the image was crowded). Detector settings are identical across rounds.

## Headline metrics
""")
    for cfg in CFGS:
        b = met.get((cfg, "baseline"))
        if not b:
            continue
        rows = []
        for c in CONDS:
            r = met.get((cfg, c))
            if not r:
                continue
            d = lambda k: ("—" if c == "baseline" else
                           f"{(float(r[k]) / float(b[k]) - 1) * 100:+.1f}%")
            rows.append([LABEL[c], f"{float(r['caap']):.4f}", d("caap"),
                         f"{float(r['snap']):.4f}", d("snap"),
                         f"{float(r['semloc']):.4f}", d("semloc"),
                         f"{float(r['det_per_img']):.1f}",
                         f"{float(r['vocab_per_img']):.1f}"])
        A(f"**{cfg}**\n")
        A(tbl(["condition", "CAAP", "ΔCAAP", "SNAP", "ΔSNAP", "SemLoc-AP",
               "ΔSemLoc", "det/img", "\\|C\\|"], rows) + "\n")

    A("""## What the revision did
""")
    A(tbl(["config", "condition", "C0→C1", "remove", "add", "rename",
           "zero-det/img", "% zero-det removed", "% added detected", "no-op"],
          [[r["config"], LABEL[r["condition"]].split(" · ")[1],
            f"{r['c0_per_img']}→{r['c1_per_img']}", r["removed_per_img"],
            r["added_per_img"], r["renames_per_img"], r["zero_det_per_img"],
            f"{r['pct_zero_det_removed']}%", f"{r['pct_added_detected']}%",
            f"{r['noop_pct']}%"]
           for r in rd(D / "vocabulary.csv")]) + "\n")

    A("""## Semantic coverage of ground-truth concepts (τ = 0.60)

A ground-truth category is *covered* by a vocabulary when some concept in it is
within τ cosine of the category name, under the same Sentence-BERT encoder SNAP
uses. **Recovered** = covered by C1 but not C0; **lost** = the reverse. These
are diagnostics, not benchmark metrics; the full τ sweep is in
`recovery.csv`.
""")
    A(tbl(["config", "condition", "recovered", "lost", "net"],
          [[k[0], LABEL[k[1]].split(" · ")[1], r["recovered"], r["lost"],
            r["net"]] for k, r in rec.items()]) + "\n")

    A("""## Did the MLLM reason about the image, or about the score?

Mean maximum detector confidence of the C0 concepts it **removed** against those
it **kept**. Self-reflection is the control: it is shown no confidence at all.
""")
    A(tbl(["config", "condition", "removed", "kept", "ratio"],
          [[k[0], LABEL[k[1]].split(" · ")[1], r["mean_maxconf_removed"],
            r["mean_maxconf_kept"],
            f"{float(r['mean_maxconf_kept']) / max(float(r['mean_maxconf_removed']), 1e-9):.1f}×"]
           for k, r in cnf.items()]) + "\n")

    A("""## Files

| file | contents |
|---|---|
| `results/feedback_dev/metrics.csv` | CAAP, SNAP, SemLoc-AP, det/img for all 12 cells |
| `results/feedback_dev/vocabulary.csv` | per-condition revision operations |
| `results/feedback_dev/recovery.csv` | coverage recovered/lost across the full τ grid |
| `results/feedback_dev/confidence_usage.csv` | removed-vs-kept confidence separation |
| `outputs/runs/coco-tune-fb-*` | D1 per condition, plus a `feedback.jsonl` trace carrying C0, the feedback text, the raw reply, parsed operations and C1 for every image |
| `outputs/runs/coco-tune-fb-baseline__*` | Round 1 at the v2 preset |
| `outputs/feedback_cache/` | every Round-2 reply, greedy-decoded and cached |
| `laod/feedback/` | summary, schema, prompts |
| `scripts/run_vocab_feedback.py` | runner |
| `scripts/eval_vocab_feedback.py` | scorer and diagnostics |
| `scripts/make_feedback_baseline.py` | Round-1 baseline at the current preset |
| `tests/test_feedback_schema.py` | 15 tests on parsing and condition enforcement |
""")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(P))
    print(f"wrote {out} ({len(chr(10).join(P).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
