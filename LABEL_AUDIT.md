# Label-extraction audit — all four OVOD backends

*Audit only. No detector was modified.*

Two phases: an offline pass over every stored run (146 run directories,
57.9M detections), and a live trace into each backend's internals on 60
COCO-test images using gemma3-4b's generated vocabulary under the `default`
prompt.

## Audit table

| | YOLO-World | OWLv2-base | G-DINO-T | G-DINO-B |
|---|---|---|---|---|
| **association mechanism** | integer class index → `self._labels[i]` | integer query index → `self._labels[i]` | decoded text phrase, re-matched to vocabulary by string | same |
| **saved label always reflects it** | **yes** | **yes** | **no** | **no** |
| **empty / unparseable** | 0.000% | 0.000% | **96.74%** | **96.51%** |
| **not in supplied vocabulary** | 0.000% | 0.000% | 0.022% | 0.020% |
| **multi-token labels correct** | 100% | 100% (index-based) | **3.94%** | **5.03%** |
| **text threshold can suppress the label but keep the box** | no — no text threshold exists | no — no text threshold exists | **yes** (`text_threshold` 0.30 > `score_threshold` 0.03) | **yes** |
| **does changing it change the association?** | n/a | n/a | **no — rendered string only** | **no** |
| **saved label vs true argmax entry** | 100% by construction | 100% by construction | 97.6% of non-empty → **3.0% of all boxes** | 97.8% of non-empty → **4.0% of all boxes** |

Percentages in rows 3–5 are over the 17 reported cells per detector
(COCO-test, LVIS-test, COCO-OOD-test): 3.7M / 15.7M / 16.6M / 16.1M
detections respectively. Rows 7–8 are from the live trace.

## Finding: the defect is specific to Grounding DINO

YOLO-World and OWLv2 both associate a box to a vocabulary entry by **integer
index**. An index cannot be partially decoded or fall below a threshold, so the
association survives into the saved label by construction. Verified rather than
assumed:

- YOLO-World: `set_classes()` altered the supplied vocabulary on **0 of 50**
  images — no dedup, no reorder, no index drift. Class index in range on
  4,085/4,085 boxes. Multi-word labels preserved 26/26.
- OWLv2: query index in range on 17,002/17,002 boxes.

Grounding DINO associates by **decoded text phrase**, and that path loses the
association twice over.

### (a) Label suppression — ~97% of boxes

The model emits a `(900 queries × 256 text tokens)` score matrix. A box is kept
when its max token score clears `score_threshold` (0.03), but its phrase is
assembled only from tokens clearing `text_threshold` (0.30). With the box bar an
order of magnitude below the text bar, a box routinely survives with no token
above the text bar and comes back with an empty phrase.

Trace, COCO val `000000000285.jpg`, gdino-tiny, 80 COCO names — four boxes clear 0.03:

| rendered at `text_threshold` 0.30 | per-box argmax token |
|---|---|
| `'bear'` (0.8350) | `bear` 0.8350 |
| `''` (0.0655) | `cat` 0.0655 |
| `''` (0.0446) | `person` 0.0446 |
| `''` (0.0304) | `bear` 0.0304 |

Every box **is** associated with a vocabulary term. The association lives in the
logits; the post-processing discards it.

`text_threshold` changes **only the rendered string**, never the box set or the
association: box count was invariant between 0.30 and 0.05 on 20/20 traced
chunks, and the argmax entry is a property of the logits, which no
post-processing threshold touches.

### (b) Multi-token shredding — 96% of multi-word entries

When a phrase *is* emitted for a multi-word entry, only the above-threshold
tokens are decoded, which yields a sub-word fragment. `_snap_to_vocabulary`'s
containment fallback then resolves it wrongly or not at all:

| saved | true argmax entry |
|---|---|
| `'##et'` | `'tennis racket'` |
| `'##board'` | `'paddleboard'` |
| `'##hiskers'` | `'whiskers'` |
| `'ski'` | `'skis'` |
| `'signs'` | `'exit sign'` |
| `'stop sign'` | `'signpost'` |

Multi-word entries are reconstructed correctly on only **3.94%** (tiny) /
**5.03%** (base) of the boxes where the true winner is multi-word. This is
visible in the offline statistics as well: on the *same* vocabularies, 14–15% of
YOLO-World and OWLv2 labels are multi-word against 0.4% of Grounding DINO's.

Lowering `text_threshold` does not fix this — it concatenates every
above-threshold token instead (`'person cat dog bear giraffe'`), which
`_snap_to_vocabulary` then collapses to the first vocabulary entry it contains
by list order, not the highest-scoring one.

## Secondary findings (not label loss)

- **OWLv2 query truncation.** The CLIP tokenizer caps at 16 tokens and queries
  are built as `"a photo of a {label}"`. 84 of 491,395 cached labels (0.017%)
  exceed it — longest 47 tokens, all from `qwen25-vl-7b` and `gemma4-12b`
  run-on strings. The index association is unaffected, so the *saved* label is
  still right; the model simply scored a truncated query. Worth a footnote, not
  a fix.
- **Grounding DINO dot collision.** The prompt is built by joining entries with
  `". "`, so a label containing a literal period is split into several phrases:
  `'j.p. morgan sign'`, `'neveracap.com'`, `'usopen.org sign'`. 9 of 31,562
  chunks (0.03%) across COCO-test. This affects the current pipeline, not only
  the proposed recovery — the model already sees more phrases than were supplied.
- **Agentic-loop runs** report 100% "off-vocabulary" because the loop rewrites
  labels by design. Excluded from all totals above.

## Verified recovery path

Each vocabulary entry occupies a contiguous token span in the dot-joined prompt.
Scoring every entry by the max over its own span and taking the per-box argmax
recovers the association. The span map was validated on 31,562 chunks across all
seven LLMs: it reassembles every entry exactly, including multi-token ones
(`'hair drier'` → `['hair','dr','##ier']`), failing only on the 9 dot-collision
cases above.

Measured on 200 tuning images with the closed 80-name vocabulary
(gdino-tiny): unnamed 95.0% → 0.0%, SNAP 0.7343 → 0.7956, CAAP unchanged. Where
the current pipeline does emit a label, the recovery agrees with it on 2,067 of
~2,075 boxes (99.6%).

That estimate is a **lower bound for the real cells**: COCO-80 has few
multi-word names, so it barely exercises defect (b), which is where the LLM
vocabularies lose the most.

## Files

- `results/label_audit_per_run.csv` — per-run statistics, all 146 runs
