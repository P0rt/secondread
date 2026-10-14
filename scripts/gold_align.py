"""Align the extra gold-corpus documents EN<->official ES with the realdocs aligner, unchanged.

usage: uv run --frozen python -I scripts/gold_align.py [DOC_ID ...] [--print]

The five original documents live in data/realdocs/<doc_id>/ (built by scripts/realdocs_align.py).
The documents added for the gold annotation corpus live in data/realdocs/gold/docs/<doc_id>/ and
are built here by the SAME code: this script imports scripts/realdocs_align.py, points its ROOT
at data/realdocs/gold/docs and registers the new documents' configs in its DOCS table, then calls
its build(). Nothing in the alignment, filtering or set-aside logic differs from the five.

Per document it reads raw/{en,es}.html (fetched 2026-10-14, recorded in SOURCE.md) and writes:
  aligned.jsonl         {id, src, ref_es, section}
  ref_deviations.jsonl  pairs set aside because the official Spanish is not equivalent (with reason)
  SOURCE.md             URLs, retrieval date, sha256 and size of each raw file, licence, scope

Two additions, both declared per document:
- REF_CLEAN: per-site link-language markers that the publisher appends to Spanish link text
  (SSA marks English-only links with "*"), removed from ref_es after alignment, the same way
  realdocs_align.py removes "(en inglés)" and "(sp)".
- reorder_es (doc config): when the Spanish page lists whole sections in a different order, the
  aligner (monotone by design) would pair the wrong items. The listed Spanish runs of blocks are
  moved back into the English order before alignment; nothing is edited or dropped.
--print dumps every kept and set-aside pair for the hand check.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realdocs_align as ra  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GOLD_DOCS = REPO / "data" / "realdocs" / "gold" / "docs"
RETRIEVED = "2026-10-14"

# Same keys and meaning as realdocs_align.DOCS.
NEW_DOCS: dict[str, dict] = {
    "uscis-common-scams": dict(
        intro_section="Common Scams (introduction)",
        not_equivalent={
            r"^If you suspect you are a victim of immigration fraud": "official ES names the Federal Communications Commission ('Comisión Federal de Comunicaciones') instead of the Federal Trade Commission.",
            r"USCIS may email you from a uscis\.gov email address": "official ES gives the sender domain as 'uscis.dhs.gov' where EN says 'uscis.gov'.",
            r"^Beware of companies offering a job in the United States": "official ES omits the last sentence (job offers on social media).",
            r"^Do not pay for or submit any form until USCIS publishes a Federal Register notice": "official ES drops the Federal Register notice ('hasta que USCIS actualice la información oficial').",
            r"^We will never ask you to transfer money to an individual\. We do not accept": "official ES rewrites the block (bank-account transfers, adds Zelle and Cash App, drops 'as payment for immigration fees').",
        },
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^Use this information to avoid common immigration scams", None),
        span_es=(r"^Use esta información para evitar estafas comunes", None), max_block=600,
    ),
    "ssa-disability-qualify": dict(
        not_equivalent={
            r"^If you can.t do other work, we.ll decide you are eligible": "official ES 'decidiremos si es elegible' (we will decide whether you are eligible) for 'we'll decide you are eligible'.",
        },
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^How Does Someone Become Eligible\?$", r"^Related Information$"),
        span_es=(r"^¿Cómo alguien puede ser elegible\?$", r"^Información relacionada$"),
        span_heading_only=True, max_block=600,
    ),
    "cdc-carbon-monoxide": dict(
        not_equivalent={},
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^At a glance$", r"^On This Page$"),
        span_es=(r"^Vistazo general$", r"^En esta página$"),
        span_heading_only=True, max_block=600, drop_tags={"figcaption"},
    ),
    "ready-home-fires": dict(
        intro_section="Home Fires (introduction)",
        not_equivalent={
            r"^Say where you are and signal for help": "official ES 'Quédese donde está' (stay where you are) for 'Say where you are'.",
            r"^Immediately shut off, then professionally replace": "official ES drops 'professionally' (replace the switches yourself).",
        },
        # ES lists 'Fireplaces and Woodstoves' after 'Keeping Children Safe' (EN: before 'Portable Space
        # Heaters'), and swaps the last two 'More Fire Prevention Tips' items.
        reorder_es=[
            (r"^Chimeneas y hornos de leña$", r"^Más consejos sobre la prevención", r"^Calentadores ambientales portátiles$"),
            (r"^Mantenga los líquidos combustibles e inflamables", None, r"^NUNCA use generadores portátiles"),
        ],
        en="raw/en.html", es="raw/es.html", root="//main",
        span_en=(r"^A fire can become life-threatening", r"^Associated Content$"),
        span_es=(r"^Un incendio puede volverse mortal", r"^Contenido relacionado$"), max_block=600,
    ),
    "medlineplus-pt-inr": dict(
        not_equivalent={
            r"^Other names: prothrombin time": "official ES gives a different list of alternative names (no 'PT, pro time').",
            r"you may need this test if you have symptoms of a bleeding or clotting disorder": "official ES drops 'or clotting disorder' ('trastorno hemorrágico (problema de la coagulación)').",
        },
        en="raw/en.html", es="raw/es.html", root="//article",
        span_en=(r"^What is a prothrombin time test", r"^References$"),
        span_es=(r"^¿Qué es una prueba de tiempo de protrombina", r"^Referencias$"),
        span_heading_only=True, max_block=600,
    ),
}

REF_CLEAN: dict[str, list[tuple[str, str]]] = {
    # "padecimientos médicos*": English-only link marker; then the blank it leaves before punctuation
    "ssa-disability-qualify": [(r"(?<=[^\W\d_])\*", ""), (r"\s+(?=[.,;:])", "")],
}

META: dict[str, dict] = {  # SOURCE.md content, one entry per doc
    "uscis-common-scams": dict(
        title="Common Scams (USCIS)",
        publisher="U.S. Citizenship and Immigration Services (USCIS), U.S. Department of Homeland Security",
        en_url="https://www.uscis.gov/scams-fraud-and-misconduct/avoid-scams/common-scams",
        es_url="https://www.uscis.gov/es/fraude-estafas-y-fallas-en-el-desempeno/evite-las-estafas-de-inmigracion/estafas-comunes",
        version="Both pages 'Last Reviewed/Updated: 08/04/2026' (article:modified_time 2026-08-04); EN and ES "
                "list each other as hreflang alternates.",
        licence_status="U.S. federal government work; USCIS states that the information on its website is public "
                       "information that may be copied (only some licensed images are excepted; no image is reused).",
        licence_url="https://www.uscis.gov/website-policies", licence_file="uscis_website_policies.html",
        licence_quote="\"considered public information and may be distributed or copied\" (USCIS Website "
                      "Policies; a byline credit is requested).",
        attribution="Source: U.S. Citizenship and Immigration Services (USCIS), \"Common Scams\" / \"Estafas "
                    "Comunes\", USCIS.gov (retrieved 2026-10-14). U.S. government work. Segmented and aligned by "
                    "SecondRead; no DHS/USCIS seal or logo is reused and no endorsement is implied.",
        scope="Main page body from the opening sentence to the end of 'Form I-9 and Email Scams'. The h4 headings "
              "go into `section`; the 'Tips' label is a sub-section.",
        content="Scam warnings with negations and conditions: 'will never ask you to transfer money to an "
                "individual', 'We do not charge a fee', a notario público 'is not authorized to provide you with "
                "any legal services', hotline 888-373-7888, Form I-134A / I-9 / I-20 names, '.gov' addresses.",
    ),
    "ssa-disability-qualify": dict(
        title="Disability Benefits: How Does Someone Become Eligible? (SSA)",
        publisher="Social Security Administration (SSA)",
        en_url="https://www.ssa.gov/benefits/disability/qualify.html",
        es_url="https://www.ssa.gov/espanol/beneficios/incapacidad/elegible.html",
        version="Neither page shows a date; both carry the same 2026 amounts. EN links to ES ('En español') and ES "
                "to EN ('In English', href /benefits/disability/qualify.html).",
        licence_status="U.S. federal government work; SSA states that most material on its website is public "
                       "domain and that it posts copyrighted material only with permission, saying so (no such "
                       "notice on this page).",
        licence_url="https://www.ssa.gov/agency/websitepolicies.html", licence_file="ssa_websitepolicies.html",
        licence_quote="\"as such, it is considered public domain\" (SSA Website Policies, 'Copyright Policy').",
        attribution="Source: Social Security Administration, \"Disability Benefits | How Does Someone Become "
                    "Eligible?\" / \"Beneficios por incapacidad | Cómo alguien puede ser elegible\", SSA.gov "
                    "(retrieved 2026-10-14). U.S. government work. Segmented and aligned by SecondRead; no SSA "
                    "logo is reused and no endorsement is implied.",
        scope="From 'How Does Someone Become Eligible?' to the end of 'What happens if the DAC gets married?'; "
              "'Related Information' and 'Publications' link lists are left out. SSA marks English-only links in "
              "the Spanish text with '*'; the markers are removed from ref_es. Sentences whose Spanish writes "
              "dollar amounts US-style ('$1,690') fail the aligner's number check and are dropped, exactly as "
              "for the five original documents.",
        content="Eligibility rules: a year (12 consecutive months), five-month waiting period, up to 12 months "
                "before the application, up to four credits a year, 40 credits with 20 in the last 10 years "
                "(20/40 Rule), 20/200 vision and 20 degrees, ages 50 to 60, disability before age 22; the "
                "five-step decision with negations ('No benefits are payable for partial disability').",
    ),
    "cdc-carbon-monoxide": dict(
        title="Carbon Monoxide Poisoning Basics (CDC)",
        publisher="Centers for Disease Control and Prevention (CDC), U.S. Department of Health and Human Services",
        en_url="https://www.cdc.gov/carbon-monoxide/about/index.html",
        es_url="https://www.cdc.gov/carbon-monoxide/es/about/informacion-basica-sobre-la-intoxicacion-por-monoxido-de-carbono.html",
        version="EN cdc:last_updated 'January 12, 2026'; ES cdc:last_updated 'el 12 de enero del 2026' (ES page "
                "modified 2026-01-22). Same structure, block for block.",
        licence_status="U.S. federal government work; CDC states that most information on its websites is in the "
                       "public domain (exceptions are marked; none on this page's text).",
        licence_url="https://www.cdc.gov/other/agencymaterials.html", licence_file="cdc_agencymaterials.html",
        licence_quote="\"is not subject to copyright, is in the public domain\" (CDC, Use of Agency Materials).",
        conditions="CDC asks for attribution ('Source: CDC'), a no-endorsement disclaimer, no change to the "
                   "substantive content, and a statement that the material is available on cdc.gov at no charge. "
                   "Machine translations and annotations are SecondRead's and are labelled as machine "
                   "translation, never as CDC text.",
        attribution="Source: CDC, \"Carbon Monoxide Poisoning Basics\" / \"Información básica sobre la "
                    "intoxicación por monóxido de carbono\" (retrieved 2026-10-14), available at no charge on "
                    "cdc.gov. Segmented and aligned by SecondRead; this use does not imply endorsement by CDC, "
                    "HHS or the U.S. Government. No CDC logo is reused.",
        scope="From 'At a glance' to the end of 'Automobile'; 'On This Page' links and image captions "
              "(figcaption, which repeat paragraph text) are left out.",
        content="Prevention instructions with negations and quantities: 'Never burn charcoal indoors', 'Never use "
                "a generator inside your home or garage, even if doors and windows are open', 'more than 20 feet', "
                "'every 5 years', more than 400 deaths, 100,000 emergency visits and 14,000 hospitalizations.",
    ),
    "ready-home-fires": dict(
        title="Home Fires (Ready.gov)",
        publisher="FEMA (U.S. Department of Homeland Security), Ready.gov",
        en_url="https://www.ready.gov/home-fires",
        es_url="https://www.ready.gov/es/incendios-en-el-hogar",
        version="Both pages 'Last Updated: 02/26/2026'; EN lists ES as hreflang alternate.",
        licence_status="U.S. federal government work (17 U.S.C. 105), public domain; FEMA states its web material "
                       "is free of copyright and lists Ready.gov among the .gov domains it owns.",
        licence_url="https://www.fema.gov/about/website-information", licence_file="fema_website_information.html",
        licence_quote="\"Most material on FEMA.gov is free of copyright and may be copied\" (FEMA Website "
                      "Information, 'Reuse and Copyright').",
        attribution="Source: FEMA, Ready.gov, \"Home Fires\" / \"Incendios en el hogar\" (retrieved 2026-10-14). "
                    "U.S. government work. Segmented and aligned by SecondRead. No FEMA/DHS seal or logo is reused.",
        scope="From 'A fire can become life-threatening...' to the end of 'More Fire Prevention Tips'; navigation, "
              "language menu, in-page links and 'Associated Content' are left out. The ES page lists 'Fireplaces "
              "and Woodstoves' after 'Keeping Children Safe' and swaps the last two tips; those Spanish blocks are "
              "moved back into the English order before alignment (`reorder_es` in scripts/gold_align.py).",
        content="Safety instructions: 'Never disable a smoke alarm while cooking', 'DO NOT attempt to reconnect "
                "utilities yourself', 'stop, drop and roll', cool water for three to five minutes, 10 feet, 12 "
                "inches, three feet, every 10 years, call 9-1-1.",
    ),
    "medlineplus-pt-inr": dict(
        title="Prothrombin Time Test and INR (PT/INR) (MedlinePlus)",
        publisher="MedlinePlus, National Library of Medicine (NLM), National Institutes of Health",
        en_url="https://medlineplus.gov/lab-tests/prothrombin-time-test-and-inr-ptinr/",
        es_url="https://medlineplus.gov/spanish/pruebas-de-laboratorio/prueba-de-tiempo-de-protrombina-e-inr-tpinr/",
        version="EN 'Last updated May 15, 2024'; ES 'Última actualización 15 mayo 2024'; EN lists ES as hreflang "
                "alternate.",
        licence_status="U.S. federal government work; MedlinePlus lists 'Medical test information', in English "
                       "and Spanish, among its public-domain content.",
        licence_url="https://medlineplus.gov/about/using/usingcontent/", licence_file="medlineplus_usingcontent.html",
        licence_quote="\"You may reproduce, redistribute, and link freely to non-copyrighted content\" "
                      "(MedlinePlus, Linking to and Using Content).",
        conditions="MedlinePlus asks for the credit 'Courtesy of MedlinePlus from the National Library of "
                   "Medicine'. The page's reference list (third-party citations) is not used.",
        attribution="Courtesy of MedlinePlus from the National Library of Medicine: \"Prothrombin Time Test and "
                    "INR (PT/INR)\" / \"Prueba de tiempo de protrombina e INR (TP/INR)\" (retrieved 2026-10-14). "
                    "Segmented and aligned by SecondRead; no NLM/NIH logo is reused and no endorsement is implied.",
        scope="Article body from 'What is a prothrombin time test with an INR (PT/INR)?' to the end of 'Is there "
              "anything else I need to know about a PT/INR test?'; 'References', related-topic links and the "
              "footer disclaimer are left out.",
        content="Warfarin monitoring: 'you may need to delay your daily dose until after testing', INR too low "
                "means clots and too high means bleeding, vitamin K foods, at-home fingertip test steps.",
    ),
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean_ref(doc_id: str, text: str) -> str:
    for pat, rep in REF_CLEAN.get(doc_id, []):
        text = re.sub(pat, rep, text)
    return re.sub(r"\s+", " ", text).strip()


def write_source(doc_id: str, rows: list[dict], dev: list[dict]) -> None:
    m = META[doc_id]
    d = GOLD_DOCS / doc_id
    files = []
    for side, url in (("en", m["en_url"]), ("es", m["es_url"])):
        p = d / "raw" / f"{side}.html"
        files.append(f"- `raw/{side}.html` ({p.stat().st_size} bytes, sha256 `{_sha(p)}`) from {url}")
    lines = [
        f"# {m['title']}", "",
        f"doc_id: `{doc_id}`  ",
        f"Publisher: {m['publisher']}  ",
        f"Retrieved: {RETRIEVED}  ",
        f"Aligned segments: {len(rows)} (`aligned.jsonl`, rows `{{id, src, ref_es, section}}`); "
        f"{len(dev)} pair(s) set aside in `ref_deviations.jsonl` because the official Spanish is not equivalent.",
        "", "## Sources", "",
        f"- English: {m['en_url']}",
        f"- Spanish (official): {m['es_url']}",
        "", "Fetched files:", "", *files, "",
        f"Version: {m['version']}",
        "", "## Licence / reuse", "",
        f"- Status: {m['licence_status']}",
        f"- Evidence: {m['licence_url']} (saved copy: `data/realdocs/gold/_licence/{m['licence_file']}`)",
        f"- Quote: {m['licence_quote']}",
        *([f"- Conditions: {m['conditions']}"] if m.get("conditions") else []),
        "", "Attribution line to show with any published excerpt:", "",
        f"> {m['attribution']}",
        "", "## How the pairs were made", "",
        f"- Scope: {m['scope']}",
        "- Built by `scripts/gold_align.py`, which runs `scripts/realdocs_align.py` unchanged: leaf blocks "
        "(headings, paragraphs, list items) are aligned EN<->ES with a monotone DP over length ratio, numbers "
        "and block type; blocks are then split into sentences and paired 1-1/1-2/2-1. A pair is kept only if "
        "both sides carry the same numbers (digits and number words) and the length ratio is plausible; "
        "anything else is dropped. Every kept pair was read by hand against the source.",
        f"- Content: {m['content']}",
        "", "## Pairs set aside (official translation not equivalent)", "",
    ]
    lines += [f"- EN: \"{x['src'][:110]}{'...' if len(x['src']) > 110 else ''}\" -- {x['note']}" for x in dev] or ["- none"]
    (d / "SOURCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reorder(blocks: list, moves: list[tuple[str, str | None, str]]) -> list:
    """Move the run [first block matching a, first later block matching b) before the block matching c."""
    def find(pat: str, start: int = 0) -> int:
        for i in range(start, len(blocks)):
            if re.search(pat, blocks[i].text):
                return i
        raise ValueError(f"reorder_es: no block matches {pat!r}")
    for a, b, c in moves:
        i = find(a)
        j = find(b, i + 1) if b else len(blocks)
        run, rest = blocks[i:j], blocks[:i] + blocks[j:]
        blocks = rest
        k = find(c)
        blocks = rest[:k] + run + rest[k:]
    return blocks


_base_load = ra.load


def _load(doc_id: str, side: str, cfg: dict) -> list:
    blocks = _base_load(doc_id, side, cfg)
    if side == "es" and cfg.get("reorder_es"):
        blocks = _reorder(blocks, cfg["reorder_es"])
    return blocks


def build(doc_id: str) -> tuple[list[dict], list[dict]]:
    rows, dev = ra.build(doc_id)
    for r in rows + dev:
        r["ref_es"] = clean_ref(doc_id, r["ref_es"])
    return rows, dev


def main(argv: list[str]) -> None:
    show = "--print" in argv
    ids = [a for a in argv if not a.startswith("--")] or list(NEW_DOCS)
    ra.ROOT = GOLD_DOCS
    ra.DOCS.update(NEW_DOCS)
    ra.load = _load  # build() looks load up at call time; adds only the declared reorder_es step
    for doc_id in ids:
        rows, dev = build(doc_id)
        d = GOLD_DOCS / doc_id
        for name, data in (("aligned.jsonl", rows), ("ref_deviations.jsonl", dev)):
            with (d / name).open("w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if doc_id in META:
            write_source(doc_id, rows, dev)
        print(f"{doc_id}: {len(rows)} segments, {len(dev)} set aside (reference not equivalent)")
        if show:
            for r in rows:
                print(f"  [{r['id'][-3:]}] {r['section']}\n     EN {r['src']}\n     ES {r['ref_es']}")
            for r in dev:
                print(f"  [SET ASIDE] {r['note']}\n     EN {r['src']}\n     ES {r['ref_es']}")


if __name__ == "__main__":
    main(sys.argv[1:])
