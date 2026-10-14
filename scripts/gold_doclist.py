"""Write data/realdocs/gold/DOC_LIST.json: the frozen document list of the gold annotation corpus.

usage: uv run --frozen python -I scripts/gold_doclist.py

The list is written once, after alignment and BEFORE any machine translation, annotation or scoring
of the added documents. It names every document of the corpus (the five original realdocs plus the
documents added by scripts/gold_align.py), with the sha256 of each aligned.jsonl so later steps can
check they use exactly these segments. Documents were accepted by the rule in ACCEPT, which looks
only at licence, the existence of an official human Spanish version and alignability; candidates
that failed it are listed with the reason. Refuses to overwrite an existing list unless --force.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "data" / "realdocs"
GOLD = REAL / "gold"

ORIGINAL = ["irs-failure-to-file-penalty", "fda-medicine-disposal", "eu-air-passenger-rights",
            "niddk-hypoglycemia", "ready-hurricanes"]
ADDED = ["uscis-common-scams", "ssa-disability-qualify", "cdc-carbon-monoxide", "ready-home-fires",
         "medlineplus-pt-inr"]

ACCEPT = {
    "kind": "public high-stakes instructions or rules for the general public (tax, benefits, immigration, "
            "medicines and tests, safety, passenger rights) with an English original",
    "licence": "reuse clearly allowed: U.S. federal government work (17 U.S.C. 105) with the publisher's own "
               "public-domain or free-reuse statement saved under _licence/, or EUR-Lex reuse under Decision "
               "2011/833/EU",
    "official_spanish": "a Spanish version published by the same agency on its own site as the counterpart of the "
                        "English page (hreflang alternate or explicit language link), written by people; pages "
                        "translated on the fly by a machine-translation widget do not count",
    "alignable": "HTML on both sides that scripts/realdocs_align.py aligns as the same document: block structure "
                 "matches and at least 40 candidate pairs survive its filters before the hand check",
    "one_per_publisher": "at most one ADDED document per publisher; when a publisher had several qualifying "
                         "candidates, the one with the most candidate pairs was taken, unless its topic duplicates "
                         "an already listed document",
    "set_aside": "within an accepted document, pairs whose official Spanish is not equivalent are set aside with "
                 "a reason (ref_deviations.jsonl) during the hand check; this is about the reference, not about "
                 "any MT output",
    "decided_on": "licence pages, page structure and candidate-pair counts only; no MT output, annotation or "
                  "system score of any added document existed when the list was frozen",
}

FREEZE = ("This list is final. Every listed document stays in the gold corpus whatever its MT quality, annotation "
          "or scores turn out to be; no document may be dropped, replaced or added because of results. Segments are "
          "fixed by the aligned.jsonl sha256 below.")

NOT_ACCEPTED = [
    {"candidate": "EPA, A Brief Guide to Mold, Moisture and Your Home",
     "en_url": "https://www.epa.gov/mold/brief-guide-mold-moisture-and-your-home",
     "es_url": "https://espanol.epa.gov/cai/una-breve-guia-para-el-moho-la-humedad-y-su-hogar",
     "reason": "not alignable: the Spanish page only links a PDF (5 HTML blocks against 100)"},
    {"candidate": "EPA, Radon Frequently Asked Questions",
     "en_url": "https://www.epa.gov/radon/radon-frequently-asked-questions",
     "es_url": "https://espanol.epa.gov/cai/preguntas-frecuentes-sobre-el-radon",
     "reason": "no official counterpart: the Spanish FAQ is a different, longer document (171 vs 39 blocks, "
               "0 candidate pairs)"},
    {"candidate": "EPA, Prevenga los envenenamientos en su hogar",
     "en_url": None, "es_url": "https://espanol.epa.gov/espanol/prevenga-los-envenenamientos-en-su-hogar",
     "reason": "no English original: the page links only 'related information in English'"},
    {"candidate": "OSHA, Heat (Working in Outdoor and Indoor Heat Environments)",
     "en_url": "https://www.osha.gov/heat-exposure", "es_url": None,
     "reason": "no official human Spanish: the page offers Spanish only through a machine-translation widget"},
    {"candidate": "IRS, Understanding your CP14 notice",
     "en_url": "https://www.irs.gov/individuals/understanding-your-cp14-notice",
     "es_url": "https://www.irs.gov/es/individuals/understanding-your-cp14-notice",
     "reason": "about 33 candidate pairs (< 40); IRS is also already listed"},
    {"candidate": "Ready.gov, Floods / Winter Weather / Power Outages / Extreme Heat",
     "en_url": "https://www.ready.gov/floods (and /winter-weather, /power-outages, /heat)",
     "es_url": "https://www.ready.gov/es/inundaciones (and /es/invierno, /es/apagones, /es/calor-extremo)",
     "reason": "one added document per publisher: Home Fires had the most candidate pairs (about 78 against "
               "53, 52, 46 and 44)"},
    {"candidate": "MedlinePlus, Blood Glucose Test",
     "en_url": "https://medlineplus.gov/lab-tests/blood-glucose-test/",
     "es_url": "https://medlineplus.gov/spanish/pruebas-de-laboratorio/prueba-de-glucosa-en-la-sangre/",
     "reason": "one added document per publisher, and the topic duplicates niddk-hypoglycemia; PT/INR taken"},
    {"candidate": "MedlinePlus drug information pages",
     "en_url": None, "es_url": None,
     "reason": "licence: drug information is not in MedlinePlus's public-domain list (licensed content)"},
    {"candidate": "CDC Vaccine Information Statements",
     "en_url": "https://www.cdc.gov/vaccines/hcp/current-vis/", "es_url": None,
     "reason": "not pursued: the Spanish VIS are distributed as PDF translations rather than CDC HTML "
               "counterparts, which the HTML aligner cannot read; CDC is represented by Carbon Monoxide "
               "Poisoning Basics"},
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _field(md: str, pat: str) -> str | None:
    m = re.search(pat, md, re.M)
    return m.group(1).strip() if m else None


def describe(doc_id: str, origin: str) -> dict:
    d = (REAL if origin == "realdocs-v1" else GOLD / "docs") / doc_id
    md = (d / "SOURCE.md").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in (d / "aligned.jsonl").open(encoding="utf-8")]
    dev = d / "ref_deviations.jsonl"
    n_dev = sum(1 for _ in dev.open(encoding="utf-8")) if dev.exists() else 0
    return {
        "doc_id": doc_id,
        "origin": origin,
        "title": _field(md, r"^# (.+)$"),
        "publisher": _field(md, r"^Publisher: (.+?)\s*$"),
        "dir": str(d.relative_to(REPO)),
        "aligned": str((d / "aligned.jsonl").relative_to(REPO)),
        "aligned_sha256": _sha(d / "aligned.jsonl"),
        "n_segments": len(rows),
        "n_set_aside_reference_not_equivalent": n_dev,
        "en_url": _field(md, r"^- English: (\S+)"),
        "es_url": _field(md, r"^- Spanish \(official\): (\S+)"),
        "retrieved": _field(md, r"^Retrieved: (\S+)"),
        "licence_status": _field(md, r"^- Status: (.+)$"),
        "licence_evidence": _field(md, r"^- Evidence: (\S+)"),
        "source_md": str((d / "SOURCE.md").relative_to(REPO)),
    }


def main(argv: list[str]) -> None:
    out = GOLD / "DOC_LIST.json"
    if out.exists() and "--force" not in argv:
        sys.exit(f"{out} exists and is frozen; not overwriting (pass --force only to repair, never because of results)")
    docs = [describe(x, "realdocs-v1") for x in ORIGINAL] + [describe(x, "gold-extension") for x in ADDED]
    doc = {
        "name": "SecondRead real-document gold corpus, EN->ES",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "freeze_rule": FREEZE,
        "acceptance_rule": ACCEPT,
        "documents": docs,
        "totals": {"documents": len(docs), "segments": sum(d["n_segments"] for d in docs),
                   "set_aside_reference_not_equivalent": sum(d["n_set_aside_reference_not_equivalent"] for d in docs)},
        "candidates_not_accepted": NOT_ACCEPTED,
        "built_by": ["scripts/realdocs_align.py (original five)", "scripts/gold_align.py (added documents)",
                     "scripts/gold_doclist.py (this list)"],
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {doc['totals']}")


if __name__ == "__main__":
    main(sys.argv[1:])
