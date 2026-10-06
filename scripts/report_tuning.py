#!/usr/bin/env python3
"""Write a markdown report for one dataset's threshold-tuning sweep."""
from __future__ import annotations
import argparse, csv, json, statistics as st, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, ".")
from laod.config import PATHS

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--note", default="")
a = ap.parse_args()

R = PATHS.root / "results"
rows = list(csv.DictReader(open(R / f"tuning_sweep_{a.tag}.csv")))
for r in rows:
    for k in ("threshold", "detections_per_image", "caap_50_95", "caap_50"):
        if k in r: r[k] = float(r[k])
    r["objective"] = float(r.get("objective", r["caap_50_95"]))
llms = sorted({r["llm"] for r in rows}); dets = sorted({r["detector"] for r in rows})
best = {(l, d): max((r for r in rows if r["llm"] == l and r["detector"] == d),
                    key=lambda r: r["objective"]) for l in llms for d in dets}
metric = rows[0].get("metric", "caap")

def tbl(rs, hs):
    return "\n".join(["| " + " | ".join(hs) + " |", "|" + "|".join(["---"]*len(hs)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rs])

L = [f"# {a.title}", "",
     f"*Generated {datetime.now():%Y-%m-%d %H:%M} · {len(rows)} measurements · "
     f"{len(llms)} LLMs × {len(dets)} detectors · objective `{metric}`*", ""]
if a.note: L += [a.note, ""]

L += ["## Selected threshold per pair", "",
      tbl([[l] + [best[(l, d)]["threshold"] for d in dets] for l in llms], ["LLM"] + dets), "",
      f"## Objective (`{metric}`) at the selected threshold", "",
      tbl([[l] + [f"{best[(l,d)]['objective']:.4f}" for d in dets]
           + [f"**{st.mean([best[(l,d)]['objective'] for d in dets]):.4f}**"] for l in llms]
          + [["**mean**"] + [f"**{st.mean([best[(l,d)]['objective'] for l in llms]):.4f}**" for d in dets] + [""]],
          ["LLM"] + dets + ["mean"]), ""]

L += ["## Per-pair tuning vs one shared threshold per detector", ""]
cmp_rows = []
for d in dets:
    pp = st.mean([best[(l, d)]["objective"] for l in llms])
    grid = sorted({r["threshold"] for r in rows if r["detector"] == d})
    shared = {t: st.mean([next(r["objective"] for r in rows if r["detector"] == d
                               and r["llm"] == l and r["threshold"] == t) for l in llms])
              for t in grid}
    ts = max(shared, key=shared.get)
    n = len({best[(l, d)]["threshold"] for l in llms})
    cmp_rows.append([d, f"{pp:.4f}", f"{shared[ts]:.4f} (@{ts})",
                     f"{(pp-shared[ts])/shared[ts]*100:+.2f}%", f"{n} of {len(llms)}"])
L += [tbl(cmp_rows, ["detector", "per-pair", "best shared", "gain", "distinct thresholds"]), ""]

L += ["## Threshold sensitivity", "",
      tbl([[d, f"{st.median([(max(v)-min(v))/min(v)*100 for v in [[r['objective'] for r in rows if r['llm']==l and r['detector']==d] for l in llms]]):.0f}%"]
           for d in dets], ["detector", "median swing across the grid"]), "",
      "Same detector, same labels, same images — only the threshold changes.", "",
      "## Full sweep", "",
      f"Every measurement is in `results/tuning_sweep_{a.tag}.csv`; the selected "
      f"values in `results/tuned_thresholds_{a.tag}.json`.", ""]

out = PATHS.root / "reports" / f"tuning_{a.tag}.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
