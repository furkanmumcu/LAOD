"""Parse and validate the MLLM's Round-2 revision.

Three rules shape this module, each from a failure we have already paid for.

**No prose fallback.** When the JSON is malformed we try to recover the
``revised_vocabulary`` array specifically, and if that fails we record a parse
failure and treat the round as a no-op (``C1 = C0``). Running a generic label
parser over the whole reply would happily turn JSON punctuation, key names and
instruction echoes into "concepts" and hand them to the detector -- a silent
fallback masquerading as a measurement, which is exactly how the VLM phase
mis-reported its parse rate.

**Refinement-only is enforced against the vocabulary, not the ``add`` field.**
A model that wants to add a concept can simply place it in
``revised_vocabulary`` and leave ``add`` empty. Checking the field alone would
let the condition leak the very mechanism it exists to exclude. A term absent
from C0 is admissible only as the target of a valid rename.

**Operations are an audit trail; ``revised_vocabulary`` is authoritative.**
Inconsistencies between the two are recorded rather than repaired, because the
disagreement is itself evidence about how well the model followed the format.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

logger = logging.getLogger(__name__)

Condition = Literal["selfreflect", "feedback_refine", "feedback_full"]
#: Conditions in which a concept absent from C0 may legitimately appear.
ADDITIONS_ALLOWED: frozenset[str] = frozenset({"selfreflect", "feedback_full"})

_WS = re.compile(r"\s+")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
#: Targeted recovery: the revised_vocabulary array alone, when the surrounding
#: object will not parse. Deliberately narrow -- it matches a JSON string array
#: and nothing else, so it cannot scrape prose.
_VOCAB_ARRAY = re.compile(
    r'"revised[_\s]*vocabulary"\s*:\s*\[(.*?)\]', re.S | re.I)
_JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')


def normalise(concept: str) -> str:
    """Concept string in the same form Round 1 produced.

    Mirrors ``parse_labels(..., "strict")`` per token: whitespace collapsed,
    surrounding punctuation stripped, lower-cased. Keeping the two identical is
    what lets C0 and C1 be compared as sets at all.
    """
    return _WS.sub(" ", str(concept).strip().strip(".;:").strip()).lower()


def normalise_all(concepts: Iterable) -> list[str]:
    """Normalise, drop empties, de-duplicate, preserve order."""
    seen: dict[str, None] = {}
    for c in concepts:
        if not isinstance(c, (str, int, float)):
            continue
        n = normalise(c)
        if n:
            seen.setdefault(n, None)
    return list(seen)


@dataclass(slots=True)
class Revision:
    """One image's Round-2 outcome, with everything needed to audit it."""

    c0: list[str]
    c1: list[str]
    keep: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    rename: dict[str, str] = field(default_factory=dict)
    add: list[str] = field(default_factory=list)
    #: 0 = JSON parsed; 1 = recovered revised_vocabulary only; 2 = no-op (C1=C0)
    fallback_level: int = 0
    parse_ok: bool = True
    violations: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_noop(self) -> bool:
        return self.c1 == self.c0

    def flags(self) -> dict[str, int]:
        """Integer-only summary, for the run store's parse_flags field."""
        return {"parse_ok": int(self.parse_ok),
                "fallback_level": self.fallback_level,
                "n_c0": len(self.c0), "n_c1": len(self.c1),
                "n_keep": len(self.keep), "n_remove": len(self.remove),
                "n_rename": len(self.rename), "n_add": len(self.add),
                "n_violations": len(self.violations),
                "n_rejected": len(self.rejected),
                "noop": int(self.is_noop)}

    def to_json(self) -> dict:
        return {"c0": self.c0, "c1": self.c1, "keep": self.keep,
                "remove": self.remove, "rename": self.rename, "add": self.add,
                "fallback_level": self.fallback_level,
                "parse_ok": self.parse_ok, "violations": self.violations,
                "rejected": self.rejected, "raw": self.raw}


def _load_json_object(raw: str) -> dict | None:
    """The reply as a JSON object, tolerating code fences and surrounding prose."""
    for candidate in _candidates(raw):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _candidates(raw: str) -> list[str]:
    text = raw.strip()
    out = [text]
    out += [m.group(1).strip() for m in _FENCE.finditer(text)]
    # The outermost brace pair, for a reply that wraps JSON in commentary.
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        out.append(text[i:j + 1])
    return out


def recover_vocabulary(raw: str) -> list[str] | None:
    """The ``revised_vocabulary`` array when the enclosing object will not parse.

    Matches a JSON string array under that key and extracts its string
    elements. Returns ``None`` rather than guessing if the key is absent.
    """
    m = _VOCAB_ARRAY.search(raw)
    if not m:
        return None
    items = [json.loads(f'"{s}"') if "\\" in s else s
             for s in _JSON_STRING.findall(m.group(1))]
    vocab = normalise_all(items)
    return vocab or None


def parse_revision(raw: str, c0: Sequence[str], condition: Condition,
                   *, max_vocab: int | None = None) -> Revision:
    """Turn a Round-2 reply into a validated revision.

    Always returns a usable ``c1``: on any unrecoverable failure it is ``c0``,
    flagged as a no-op so the condition is never silently fed garbage.
    """
    base = normalise_all(c0)
    rev = Revision(c0=base, c1=list(base), raw=raw)

    obj = _load_json_object(raw)
    if obj is None:
        vocab = recover_vocabulary(raw)
        if vocab is None:
            rev.parse_ok = False
            rev.fallback_level = 2
            rev.violations.append("unparseable reply; round treated as a no-op")
            logger.warning("Round-2 reply unparseable; C1 = C0")
            return rev
        rev.fallback_level = 1
        rev.violations.append("object malformed; recovered revised_vocabulary only")
        obj = {"revised_vocabulary": vocab}

    vocab = obj.get("revised_vocabulary")
    if not isinstance(vocab, list) or not normalise_all(vocab):
        recovered = recover_vocabulary(raw)
        if recovered is None:
            rev.parse_ok = False
            rev.fallback_level = 2
            rev.violations.append("no usable revised_vocabulary; no-op")
            return rev
        rev.fallback_level = max(rev.fallback_level, 1)
        vocab = recovered

    c1 = normalise_all(vocab)
    rev.keep = normalise_all(obj.get("keep") or [])
    rev.remove = normalise_all(obj.get("remove") or [])
    rev.add = normalise_all(obj.get("add") or [])
    raw_rename = obj.get("rename") or {}
    if isinstance(raw_rename, dict):
        rev.rename = {normalise(k): normalise(v) for k, v in raw_rename.items()
                      if normalise(k) and normalise(v)}
    else:
        rev.violations.append("rename was not an object; ignored")

    # --- condition enforcement, against the vocabulary itself ---------------
    if condition not in ADDITIONS_ALLOWED:
        base_set = set(base)
        # A rename target is the only legitimate new term here, and only when
        # its source was actually in C0.
        allowed_new = {t for s, t in rev.rename.items() if s in base_set}
        illegal = [c for c in c1 if c not in base_set and c not in allowed_new]
        if illegal:
            rev.violations.append(
                f"{len(illegal)} concept(s) absent from C0 and not a valid "
                f"rename target, under a no-additions condition")
            rev.rejected = illegal
            c1 = [c for c in c1 if c not in set(illegal)]
            logger.warning("refinement-only: rejected %d illegal addition(s): %s",
                           len(illegal), illegal[:5])
        if rev.add:
            rev.violations.append("add field non-empty under a no-additions "
                                  "condition; cleared")
            rev.add = []

    if max_vocab is not None and len(c1) > max_vocab:
        rev.violations.append(f"vocabulary truncated from {len(c1)} to {max_vocab}")
        rev.rejected.extend(c1[max_vocab:])
        c1 = c1[:max_vocab]

    if not c1:
        rev.violations.append("revised vocabulary empty after validation; no-op")
        rev.parse_ok = False
        rev.fallback_level = 2
        rev.c1 = list(base)
        return rev

    rev.c1 = c1
    # Operations are an audit trail: record where they disagree with the
    # vocabulary rather than rewriting either to match.
    stated = set(rev.keep) | set(rev.add) | set(rev.rename.values())
    if stated and set(c1) - stated:
        rev.violations.append(
            f"{len(set(c1) - stated)} concept(s) in revised_vocabulary are not "
            "accounted for by keep/add/rename")
    return rev
