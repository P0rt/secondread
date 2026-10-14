"""Shared helpers for synthetic error generation (scripts/synth_rules.py, scripts/synth_llm.py).

Split hygiene is enforced here, not in the generators: a synthetic example for split S
is built only from units of data/splits/<lp>.<S>.jsonl, every parent's document is
checked against the committed doc->split map, and no parent may share its source
sentence with another split. Train synth therefore never sees test documents and the
held-out synth test set is built from test documents only.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path

SPLITS_DIR = Path("data/splits")
SYNTH_DIR = Path("data/synth")
SPLITS = ("train", "dev", "test")
MAJOR_SEVERITIES = ("major", "critical")

# Span category per synthetic error type. WMT MQM (Unbabel style) files wrong numbers,
# dates and flipped negation under "mistranslation", so we do the same.
SPAN_CATEGORY = {"omission": "omission", "addition": "addition"}


# --------------------------------------------------------------------------- I/O

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(lp: str, split: str) -> list[dict]:
    """Rows of one split, after checking every row's document really belongs to it."""
    doc_split = json.loads((SPLITS_DIR / f"{lp}.docs.json").read_text(encoding="utf-8"))
    rows = read_jsonl(SPLITS_DIR / f"{lp}.{split}.jsonl")
    bad = [r["id"] for r in rows if doc_split.get(r["doc"]) != split]
    if bad:
        raise SystemExit(f"split hygiene: {len(bad)} rows of {lp}.{split} belong to other splits, e.g. {bad[:3]}")
    return rows


def norm_src(text: str) -> str:
    return " ".join(text.lower().split())


def sources_outside(lp: str, split: str) -> set[str]:
    """Normalised source sentences of every *other* split (for the text-level leak guard)."""
    out: set[str] = set()
    for s in SPLITS:
        if s != split and (SPLITS_DIR / f"{lp}.{s}.jsonl").exists():
            out.update(norm_src(r["src"]) for r in read_jsonl(SPLITS_DIR / f"{lp}.{s}.jsonl"))
    return out


def has_major_span(row: dict) -> bool:
    return any(sp["severity"] in MAJOR_SEVERITIES for sp in row["spans"])


def source_pool(lp: str, split: str) -> tuple[list[dict], dict]:
    """Clean parents for synthesis: label OK, no major/critical span of any family.

    Human references (system "refA") are ordinary rows in the split files, so they are
    included automatically whenever their unit is clean. Duplicate (src, mt) pairs
    (two systems producing the same output) are kept once. Returns (parents, stats).
    """
    rows = load_split(lp, split)
    outside = sources_outside(lp, split)
    stats = collections.Counter(rows=len(rows))
    seen, pool = set(), []
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["label"] != "OK" or has_major_span(r):
            stats["skip_not_clean"] += 1
            continue
        if "CANARY" in r["src"]:
            stats["skip_canary"] += 1
            continue
        if norm_src(r["src"]) in outside:
            stats["skip_src_in_other_split"] += 1
            continue
        key = (r["src"], r["mt"])
        if key in seen:
            stats["skip_duplicate"] += 1
            continue
        seen.add(key)
        pool.append(r)
    stats["parents"] = len(pool)
    return pool, dict(stats)


def natural_mts(lp: str, split: str) -> dict[str, set[str]]:
    """src -> every human-rated MT of it in this split (any label), to avoid label clashes."""
    out: dict[str, set[str]] = collections.defaultdict(set)
    for r in load_split(lp, split):
        out[r["src"]].add(r["mt"])
    return out


def segment_key(row: dict) -> tuple[str, str]:
    """One source segment; each appears once per MT system (up to 16 times)."""
    return row["doc"], row["src"]


# --------------------------------------------------------------------------- determinism

def rng_for(seed: int, *parts: str) -> random.Random:
    """Independent, order-free RNG per (seed, item): adding items never reshuffles others."""
    h = hashlib.sha256("|".join([str(seed), *parts]).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# --------------------------------------------------------------------------- records

def shift_spans(spans: list[dict], start: int, end: int, new_len: int) -> list[dict]:
    """Carry the parent's (minor/neutral) MT spans over an edit of mt[start:end] -> new_len chars.

    Spans before the edit stay, spans after it move, spans touching it are dropped.
    """
    delta = new_len - (end - start)
    out = []
    for sp in spans:
        if sp["side"] != "mt" or sp["end"] <= start:
            out.append(dict(sp))
        elif sp["start"] >= end:
            out.append({**sp, "start": sp["start"] + delta, "end": sp["end"] + delta})
    return out


def error_span(mt: str, start: int, end: int, synth_type: str) -> dict:
    category = SPAN_CATEGORY.get(synth_type, "mistranslation")
    return {"start": start, "end": end, "text": mt[start:end], "side": "mt",
            "category": category, "family": "meaning", "severity": "major"}


def make_record(parent: dict, *, mt: str, label: str, synth_type: str, generator: str,
                edit: dict, spans: list[dict]) -> dict:
    """A synthetic unit in the split-file schema plus synthetic/synth_* provenance fields.

    edit = {"start", "end", "text", "orig"}: the changed region in the new MT and the
    parent text it replaced (empty start==end region for a pure deletion).
    """
    assert label in ("Major", "OK"), label
    return {
        "id": f"{parent['id']}|synth:{generator}:{synth_type}",
        "lp": parent["lp"], "year": parent["year"], "system": parent["system"],
        "doc": parent["doc"], "domain": parent["domain"], "rater": "synthetic",
        "src": parent["src"], "mt": mt, "spans": spans,
        "label_any_major": label == "Major", "label": label,
        "synthetic": True, "synth_type": synth_type, "synth_generator": generator,
        "synth_parent": parent["id"], "synth_edit": edit,
    }


def assert_hygiene(records: list[dict], lp: str, split: str) -> None:
    """Final guard on generated output: every record's doc and source belong to `split`."""
    doc_split = json.loads((SPLITS_DIR / f"{lp}.docs.json").read_text(encoding="utf-8"))
    outside = sources_outside(lp, split)
    for r in records:
        if r["lp"] != lp:
            continue
        assert doc_split[r["doc"]] == split, (r["id"], doc_split[r["doc"]], split)
        assert norm_src(r["src"]) not in outside, r["id"]


# --------------------------------------------------------------------------- lexical checks

# Grammatical negation markers. Lexical negation fused into a word ("невозможно",
# "imposible") is deliberately not counted: the checks look for an added/removed marker.
NEGATION_RE = {
    "en": re.compile(r"\b(?:not|no|never|nothing|nobody|none|neither|nor|nowhere|cannot|without)\b|n't",
                     re.IGNORECASE),
    "ru": re.compile(r"(?<!\w)(?:не|ни|нет|нельзя|никогда|нигде|никуда|ниоткуда|нисколько|ничуть|"
                     r"ничто|ничего|ничему|ничем|ничём|никто|никого|никому|никем|ником|никак\w*|ничь\w*|ничей)(?!\w)",
                     re.IGNORECASE),
    "es": re.compile(r"(?<!\w)(?:no|nunca|jamás|nada|nadie|ningún|ninguno|ninguna|ningunos|ningunas|"
                     r"ni|tampoco|sin)(?!\w)", re.IGNORECASE),
}


def target_lang(lp: str) -> str:
    return lp.split("-")[1]


def negation_count(text: str, lang: str) -> int:
    return len(NEGATION_RE[lang].findall(text))


def digit_string(text: str) -> str:
    """All digits in reading order: equal strings mean no number was added, dropped or changed."""
    return "".join(re.findall(r"\d", text))
