"""Align official EN/ES versions of public documents into segment pairs.

usage: uv run --frozen python -I scripts/realdocs_align.py [DOC_ID ...]

Reads data/realdocs/<doc_id>/raw/{en,es}.html (fetched as recorded in each
SOURCE.md) and writes data/realdocs/<doc_id>/aligned.jsonl with rows
{id, src, ref_es, section}. Only pairs we can align confidently are kept:

1. extract leaf text blocks (headings, paragraphs, list items) from the
   document body, dropping navigation, footnote markers and link-only items;
2. align blocks EN<->ES with a monotone DP over length ratio, numbers and
   block type (gaps allowed, so out-of-date or extra blocks are skipped);
3. split aligned blocks into sentences and pair them 1-1 / 1-2 / 2-1 by the
   same costs, falling back to the whole block when sentences do not pair;
4. keep a pair only if both sides carry the same numbers and the length
   ratio is plausible.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from lxml import html as LH

ROOT = Path(__file__).resolve().parents[1] / "data" / "realdocs"

BLOCK = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "dt", "dd", "blockquote", "figcaption", "caption"}
HEAD = {"h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass
class Block:
    tag: str
    text: str
    link_only: bool = False
    heading: bool = False
    sub: bool = False  # short label (accordion title, table caption): becomes part of `section`


def norm(s: str) -> str:
    s = s.replace("\xa0", " ").replace("\u200b", "").replace("\u00ad", "")
    s = re.sub(r"(?<=[^\W\d_])\(\d\)", "", s)  # OJ footnote calls glued to a word: "carriers(5);"
    return re.sub(r"\s+", " ", s).strip()


def _has_block_desc(el) -> bool:
    return any(isinstance(d.tag, str) and d.tag in BLOCK for d in el.iterdescendants())


def _own_text(el) -> str:
    parts = [el.text or ""]
    for c in el:
        if isinstance(c.tag, str) and not (c.tag in BLOCK or _has_block_desc(c)):
            parts.append(c.text_content())
        parts.append(c.tail or "")
    return norm("".join(parts))


def _link_only(el, text: str) -> bool:
    links = norm(" ".join(a.text_content() for a in el.iter("a")))
    return bool(links) and len(links) >= 0.9 * len(text)


def extract(path: Path, root_xpath: str) -> list[Block]:
    raw = path.read_bytes().decode("utf-8", "replace")
    doc = LH.document_fromstring(raw)
    drop = "//script|//style|//noscript|//nav|//sup|//*[contains(concat(' ',normalize-space(@class),' '),' sr-only ')]"
    for bad in doc.xpath(drop):
        bad.drop_tree()
    root = doc.xpath(root_xpath)[0]
    out: list[Block] = []
    for el in root.iter():
        if not isinstance(el.tag, str) or el.tag not in BLOCK:
            continue
        nested = _has_block_desc(el)
        text = _own_text(el) if nested else norm(el.text_content())
        if not text:
            continue
        label = (el.tag == "p" or nested) and len(text.split()) <= 8 and not re.search(r"[.:;,?!]$", text) \
            and text[:1].isupper()
        out.append(Block(el.tag, text, link_only=(not nested and _link_only(el, text)), heading=el.tag in HEAD,
                         sub=label and el.tag not in HEAD))
    return out


# ---------------------------------------------------------------- numbers

EN_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ES_WORDS = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
            "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "diecisiete": 17,
            "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintidós": 22, "veintitrés": 23,
            "veinticuatro": 24, "veinticinco": 25, "veintiséis": 26, "veintisiete": 27, "veintiocho": 28,
            "veintinueve": 29, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
            "ochenta": 80, "noventa": 90}


def numbers(text: str, lang: str) -> set[float]:
    t = text.lower()
    vals: set[float] = set()
    if lang == "en":
        for m in re.finditer(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", t):
            vals.add(float(m.group().replace(",", "")))
        for m in re.finditer(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(two|three|four|five|six|seven|eight|nine|one)\b", t):
            vals.add(EN_WORDS[m.group(1)] + (EN_WORDS.get(m.group(2)) or 1))
            t = t.replace(m.group(), " ")
        for w, v in EN_WORDS.items():
            if re.search(rf"\b{w}\b", t):
                vals.add(v)
    else:
        for m in re.finditer(r"\d{1,3}(?:[. ]\d{3})+(?:,\d+)?|\d+(?:[,.]\d+)?", t):
            g = m.group()
            if re.fullmatch(r"\d{1,3}(?:[. ]\d{3})+(?:,\d+)?", g):
                g = g.replace(".", "").replace(" ", "").replace(",", ".")
            else:
                g = g.replace(",", ".")
            vals.add(float(g))
        for m in re.finditer(r"\b(treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa) y (uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)\b", t):
            vals.add(ES_WORDS[m.group(1)] + (ES_WORDS.get(m.group(2)) or 1))
            t = t.replace(m.group(), " ")
        for w, v in ES_WORDS.items():
            if re.search(rf"(?<![\wáéíóúñ]){w}(?![\wáéíóúñ])", t):
                vals.add(v)
    vals.discard(1.0)
    return vals


# ---------------------------------------------------------------- costs

RATIO = 1.15  # ES is ~15% longer than EN in these sources


def unit_cost(en: str, es: str) -> float:
    r = (len(es) + 8) / (len(en) + 8)
    c = abs(math.log(r / RATIO))
    ne, ns = numbers(en, "en"), numbers(es, "es")
    if ne or ns:
        c += 1.5 * (1 - len(ne & ns) / len(ne | ns))
    return c


def block_cost(a: Block, b: Block) -> float:
    c = unit_cost(a.text, b.text)
    if a.heading != b.heading:
        c += 0.6
    if a.link_only != b.link_only:
        c += 0.3
    return c


def dp_align(xs, ys, cost, skip=1.0, merges=False):
    """Monotone alignment. Returns list of ((i0,i1),(j0,j1),cost) beads; gaps omitted."""
    n, m = len(xs), len(ys)
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    moves = [(1, 1), (1, 0), (0, 1)] + ([(1, 2), (2, 1)] if merges else [])
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] == INF:
                continue
            for di, dj in moves:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                if di and dj:
                    c = cost(xs[i:ni], ys[j:nj]) + (0.25 if (di, dj) != (1, 1) else 0.0)
                else:
                    c = skip
                if D[i][j] + c < D[ni][nj]:
                    D[ni][nj] = D[i][j] + c
                    B[ni][nj] = (i, j, c)
    beads = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, c = B[i][j]
        if pi != i and pj != j:
            beads.append(((pi, i), (pj, j), c))
        i, j = pi, pj
    return beads[::-1]


# ---------------------------------------------------------------- sentences

ABBR = r"(?:U\.S|e\.g|i\.e|No|Nos|Dr|Mr|Mrs|Ms|St|vs|etc|Inc|Rev|Proc|p|pp|Art|art|núm|Sr|Sra|EE\.UU|a\.m|p\.m|Fig|aprox|approx|[A-Z])"


def sentences(text: str) -> list[str]:
    parts, start = [], 0
    for m in re.finditer(r"(?<=[.!?])[\"”’)]?\s+(?=[\"“¿¡(\[]?[A-ZÁÉÍÓÚÑ])", text):
        before = text[start:m.start()]
        if re.search(rf"\b{ABBR}\.$", before.rstrip("\"”’)")) or re.fullmatch(r"\s*\d+\.", before):
            continue
        parts.append(text[start:m.end()].strip())
        start = m.end()
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def ok_pair(en: str, es: str) -> bool:
    if numbers(en, "en") != numbers(es, "es"):
        return False
    r = len(es) / max(1, len(en))
    lo, hi = (0.75, 1.9) if len(en) >= 40 else (0.5, 3.0)
    return lo <= r <= hi


def align_block(en: str, es: str, max_block: int) -> list[tuple[str, str]]:
    se, ss = sentences(en), sentences(es)
    if len(se) > 1 or len(ss) > 1:
        beads = dp_align(se, ss, lambda a, b: unit_cost(" ".join(a), " ".join(b)), skip=2.0, merges=True)
        covered_e = sum(e1 - e0 for (e0, e1), _, _ in beads)
        covered_s = sum(s1 - s0 for _, (s0, s1), _ in beads)
        pairs = [(" ".join(se[e0:e1]), " ".join(ss[s0:s1])) for (e0, e1), (s0, s1), _ in beads]
        if covered_e == len(se) and covered_s == len(ss) and all(c < 0.6 for *_, c in beads) \
                and all(ok_pair(a, b) for a, b in pairs):
            return pairs
    if len(en) <= max_block and ok_pair(en, es):
        return [(en, es)]
    return []


# ---------------------------------------------------------------- docs

def _span(blocks: list[Block], start: str | None, end: str | None, heading_only: bool = False) -> list[Block]:
    """Slice from the first block matching `start` up to (excluding) the next matching `end`."""
    def hit(b: Block, pat: str | None) -> bool:
        return bool(pat) and (b.heading or not heading_only) and re.search(pat, b.text) is not None
    i0 = next((i for i, b in enumerate(blocks) if hit(b, start)), 0)
    i1 = next((i for i, b in enumerate(blocks) if i > i0 and hit(b, end)), len(blocks))
    return blocks[i0:i1]


def eurlex_headings(blocks: list[Block], art_re: str) -> list[Block]:
    """'Article 7' and the title line that follows it are headings in the old OJ HTML."""
    for i, b in enumerate(blocks):
        if re.fullmatch(art_re, b.text):
            b.heading = True
            if i + 1 < len(blocks):
                blocks[i + 1].heading = True
    return blocks


DOCS: dict[str, dict] = {
    "eu-air-passenger-rights": dict(
        not_equivalent={
            r"delayed for at least two hours, ask at the check-in": "EN 'at least two hours'; official ES 'superior a dos horas' (more than two hours).",
        },
        en="raw/eng.html", es="raw/spa.html", root='//div[@id="TexteOnly"]',
        span_en=(r"^Article 1$", r"^Done at Strasbourg"), span_es=(r"^Artículo 1$", r"^Hecho en Estrasburgo"),
        art=(r"Article \d+", r"Artículo \d+"), max_block=700,
        drop_en=[r"Gibraltar"],
    ),
    "ready-hurricanes": dict(
        intro_section="Hurricanes (introduction)",
        not_equivalent={
            r"^If you have questions about your disaster assistance application": "ES page lists the FEMA app instead of the AskIA email address.",
        },
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^If you are a disaster survivor", r"^Additional Resources$"),
        span_es=(r"^Si usted es un sobreviviente", r"^Recursos adicionales$"), max_block=600,
    ),
    "niddk-hypoglycemia": dict(
        not_equivalent={
            r"^are age 65 or older": "EN '65 or older'; official ES 'tiene más de 65 años' (over 65).",
            r"^But severely low blood glucose can cause": "official ES drops 'severely' (any low glucose can cause coma or death).",
            r"^If you have hypoglycemia unawareness": "official ES 'desconoce que tiene hipoglucemia' (does not know they have it) for the clinical term hypoglycemia unawareness.",
        },
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^What is low blood glucose\?$", r"^Clinical Trials for Low Blood Glucose$"),
        span_es=(r"^¿Qué es una concentración baja de glucosa en la sangre\?$", r"^Ensayos clínicos para la concentración baja"),
        span_heading_only=True, max_block=600,
    ),
    "irs-failure-to-file-penalty": dict(
        intro_section="Failure to file penalty (introduction)",
        not_equivalent={
            r"^If failure to file and failure to pay penalties both apply": "official ES garbles which penalty is reduced by which.",
            r"^The failure to file penalty is charged for each month \(or partial month\) the failure continues": "official ES garbled ('de la multa y continúa').",
            r"^Penalty relief for reasonable cause for filing a late": "official ES drops 'will be presumed' (no main verb).",
            r"husband and wife .*considered one partner": "official ES 'un solo cónyuge' (one spouse) instead of one partner (socio).",
        },
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^The failure to file penalty applies", r"^Related$|^Page Last Reviewed"),
        span_es=(r"^La multa por incumplimiento de presentar", r"^Relacionado$|^Fecha de la última revisión"),
        max_block=600, drop_tags={"td", "th"},
    ),
    "fda-medicine-disposal": dict(
        intro_section="Where and How to Dispose of Unused Medicines (introduction)",
        not_equivalent={
            r"flush these potentially dangerous medicines": "official ES omits 'these potentially dangerous medicines'.",
            r"^Some pharmacies offer on-site medicine drop-off boxes": "official ES omits the on-site drop-off boxes.",
            r"^Disposing medicines in household trash": "official ES omits the FDA flush list reference and the dosage-form list.",
            r"^The FDA and the U.S. Environmental Protection Agency take seriously": "official ES omits the middle sentence.",
        },
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^Is your medicine cabinet full", r"^Content current as of"),
        span_es=(r"^¿Está su botiquín lleno", r"^Content current as of"), max_block=600,
    ),
}


def load(doc_id: str, side: str, cfg: dict) -> list[Block]:
    blocks = extract(ROOT / doc_id / cfg[side], cfg["root"])
    start, end = cfg[f"span_{side}"]
    blocks = _span(blocks, start, end, cfg.get("span_heading_only", False))
    if "art" in cfg:
        blocks = eurlex_headings(blocks, cfg["art"][0 if side == "en" else 1])
    drop_tags = cfg.get("drop_tags", set())
    blocks = [b for b in blocks if b.tag not in drop_tags]
    return blocks


def build(doc_id: str) -> tuple[list[dict], list[dict]]:
    """Return (aligned rows, pairs dropped because the official Spanish is not equivalent)."""
    cfg = DOCS[doc_id]
    en, es = load(doc_id, "en", cfg), load(doc_id, "es", cfg)
    beads = dp_align(en, es, lambda a, b: block_cost(a[0], b[0]), skip=1.0)
    matched = {e0: (s0, c) for (e0, _), (s0, _), c in beads}
    rows, deviations, seen = [], [], set()
    section, sub, art_title = cfg.get("intro_section", ""), "", None
    for i, b in enumerate(en):
        if b.heading:
            if "art" in cfg and re.fullmatch(cfg["art"][0], b.text):
                art_title = b.text
                continue
            section, sub = (f"{art_title} ({b.text})" if art_title else b.text), ""
            art_title = None
            continue
        if b.sub:
            sub = b.text
            continue
        sec = f"{section} > {sub}" if section and sub else (section or sub)
        if i not in matched or b.link_only:
            continue
        s0, c = matched[i]
        tb = es[s0]
        if tb.heading or tb.link_only or c > 0.9:
            continue
        if any(re.search(p, b.text) for p in cfg.get("drop_en", [])):
            continue
        es_text = re.sub(r"\s*\((?:en inglés|sp)\)(?=[\s,.;:]|$)", "", tb.text)  # link-language markers
        for src, ref in align_block(b.text, es_text, cfg.get("max_block", 600)):
            if len(src.split()) < 4 or src in seen:
                continue
            seen.add(src)
            note = next((n for pat, n in cfg.get("not_equivalent", {}).items() if re.search(pat, src)), None)
            if note:
                deviations.append({"src": src, "ref_es": ref, "section": sec, "note": note})
                continue
            rows.append({"id": f"{doc_id}-{len(rows) + 1:03d}", "src": src, "ref_es": ref, "section": sec})
    return rows, deviations


def main(argv: list[str]) -> None:
    ids = argv or list(DOCS)
    for doc_id in ids:
        rows, deviations = build(doc_id)
        for name, data in (("aligned.jsonl", rows), ("ref_deviations.jsonl", deviations)):
            with (ROOT / doc_id / name).open("w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{doc_id}: {len(rows)} segments, {len(deviations)} set aside (reference not equivalent)")


if __name__ == "__main__":
    main(sys.argv[1:])
