"""Build the en-es real-document demo jobs: one XLIFF 1.2 file per document (plus a Trados
SDLXLIFF for the IRS page), demo/answer_key.en-es.json, and the per-document numbers quoted in
docs/REALDOCS.md.

usage: PYTHONPATH=. uv run --frozen python scripts/make_realdocs_demo.py [--out demo]
           [--wmt-out data/demo_wmt] [--check-only]

Inputs (data/ is git-ignored; see docs/REALDOCS.md for how each was made)
  data/realdocs/scored.jsonl               SecondRead v1 (backend agent), rule checks and Gemma 4
                                           31B direct scores per (doc, system, segment)
  data/realdocs/adjudication_final.jsonl   sampled adjudication against the official Spanish
  data/realdocs/adjudication_fill.jsonl    the rest of each chosen marian document, same rule
  data/realdocs/wmt24_*/aligned.jsonl      WMT24 en-es MT with human MQM gold labels

Only openly reusable text goes into demo/ (committed): the three public documents, with MT by
Helsinki-NLP/opus-mt-en-es (Apache-2.0), and their answer key demo/answer_key.en-es.json.
The WMT24 document (WMT test data is released for research use only; MT by the WMT24
submission MSLC) is written with its own answer key, answer_key.en-es.wmt24.json, to the
git-ignored data/demo_wmt/ and is not redistributed. NLLB-200 (CC-BY-NC-4.0) output is never
written anywhere.

Every file is parsed back with secondread.xliff and must give the same segments in the same
order; the answer key records each segment's key in each file. The build is deterministic.
"""
from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from pathlib import Path

from lxml import etree

from secondread import xliff
from secondread.xliff import SDL_NS, XLIFF12_NS

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "realdocs"
LP = "en-es"
NS_UUID = uuid.UUID("0c7b6f0e-2d55-4a59-9d4e-8a1f3b7e5e10")  # stable SDL ids across rebuilds
GEMMA_HI, GEMMA_LO = 0.971, 0.5

MARIAN = {
    "name": "OPUS-MT en-es (MarianMT)",
    "model_id": "Helsinki-NLP/opus-mt-en-es",
    "revision": "5bc4493d463cf000c1f0b50f8d56886a392ed4ab",
    "licence": "Apache-2.0 (model card front matter at the pinned revision)",
    "attribution": "Machine translation: Helsinki-NLP OPUS-MT en-es (Tiedemann & Thottingal, "
                   "2020, OPUS-MT, EAMT), Apache-2.0, run locally with beam 4, one sentence per "
                   "input (scripts/realdocs_mt.py).",
}

DOCS = {
    "irs-failure-to-file-penalty": {
        "system": "marian", "sdlxliff": True,
        "title": "Failure to File Penalty",
        "title_es": "Multa por incumplimiento de presentar la declaración de impuestos",
        "publisher": "Internal Revenue Service (IRS), U.S. Department of the Treasury",
        "domain": "tax (penalty rules, rates, deadlines)",
        "src_lang": "en-US", "tgt_lang": "es-US",
        "urls": {"en": "https://www.irs.gov/payments/failure-to-file-penalty",
                 "es": "https://www.irs.gov/es/payments/failure-to-file-penalty"},
        "licence": "U.S. federal government work, not subject to copyright (17 U.S.C. 105); "
                   "IRS: content made by federal employees \"is not subject to copyright and may "
                   "be freely copied. Credit is requested.\"",
        "licence_evidence": "https://www.irs.gov/about-irs/use-of-content-from-irsgov",
        "attribution": "Source: Internal Revenue Service, \"Failure to File Penalty\" / \"Multa "
                       "por incumplimiento de presentar la declaración de impuestos\", IRS.gov "
                       "(retrieved 2026-10-14). U.S. government work. Segmented and aligned by "
                       "SecondRead; no IRS seal or logo is reused and no endorsement is implied.",
        "scope": "Page body from the opening sentence to 'Get help'; penalty tables, link-only "
                 "items and 4 pairs whose official Spanish is not equivalent are left out.",
    },
    "fda-medicine-disposal": {
        "system": "marian",
        "title": "Where and How to Dispose of Unused Medicines",
        "title_es": "Cómo y dónde deshacerse de los medicamentos sin utilizar",
        "publisher": "U.S. Food and Drug Administration (FDA)",
        "domain": "consumer health (drug take-back, flushing, fentanyl patches)",
        "src_lang": "en-US", "tgt_lang": "es-US",
        "urls": {"en": "https://www.fda.gov/consumers/consumer-updates/where-and-how-dispose-unused-medicines",
                 "es": "https://www.fda.gov/consumers/articulos-para-el-consumidor-en-espanol/"
                       "como-y-donde-deshacerse-de-los-medicamentos-sin-utilizar"},
        "licence": "U.S. federal government work, public domain; FDA: \"They are in the public "
                   "domain and may be republished, reprinted\" (credit appreciated).",
        "licence_evidence": "https://www.fda.gov/about-fda/about-website/website-policies",
        "attribution": "Source: U.S. Food and Drug Administration, \"Where and How to Dispose of "
                       "Unused Medicines\" / \"Cómo y dónde deshacerse de los medicamentos sin "
                       "utilizar\" (retrieved 2026-10-14). Segmented and aligned by SecondRead; "
                       "no FDA logo is reused and no endorsement is implied.",
        "scope": "Article body from 'Is your medicine cabinet full...' to the end of 'Flushing "
                 "Drugs and the Water Supply'; 4 pairs whose official Spanish is not equivalent "
                 "are left out.",
    },
    "eu-air-passenger-rights": {
        "system": "marian",
        "title": "Regulation (EC) No 261/2004 (air passenger rights), Articles 1-19",
        "title_es": "Reglamento (CE) n.º 261/2004 (derechos de los pasajeros aéreos), artículos 1 a 19",
        "publisher": "European Parliament and Council of the EU; Publications Office of the EU (EUR-Lex)",
        "domain": "EU law (denied boarding, cancellation, delay, compensation)",
        "src_lang": "en-GB", "tgt_lang": "es-ES",
        "urls": {"en": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32004R0261",
                 "es": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32004R0261"},
        "licence": "EU legal document; reuse authorised under Commission Decision 2011/833/EU "
                   "with the source acknowledged. EUR-Lex legal notice: \"Unless otherwise "
                   "specified, you can re-use the legal documents published in EUR-Lex\".",
        "licence_evidence": "https://eur-lex.europa.eu/content/legal-notice/legal-notice.html",
        "attribution": "Source: EUR-Lex, Regulation (EC) No 261/2004, OJ L 46, 17.2.2004, p. 1. "
                       "© European Union, https://eur-lex.europa.eu, 1998-2026. Segmented and "
                       "aligned by SecondRead; only European Union documents published in the "
                       "Official Journal are deemed authentic.",
        "scope": "Articles 1-19 of the original OJ text (recitals, signatures, footnotes, the "
                 "Gibraltar paragraphs and 1 pair whose official Spanish is not equivalent are "
                 "left out).",
    },
    "wmt24_the_other_side_stormfall": {
        "system": "MSLC", "wmt": True,
        "title": "The Other Side: Stormfall (WMT24 literary test document, first chunk)",
        "publisher": "WMT24 General MT test set; English source story from Archive of Our Own",
        "domain": "literary (military science fiction)",
        "src_lang": "en-US", "tgt_lang": "es-MX",
        "wmt_doc": "test-en-literary_the_other_side_stormfall_chunk_1_words_992",
        "urls": {"wmt24": "https://www2.statmt.org/wmt24/translation-task.html",
                 "mqm": "https://github.com/google/wmt-mqm-human-evaluation"},
        "licence": "NOT openly licensed: WMT24 test data \"can be freely used for research "
                   "purposes\" (cite Kocmi et al., 2024); copyright in the story stays with its "
                   "AO3 author. MQM annotations: google/wmt-mqm-human-evaluation, Apache-2.0. "
                   "Generated locally from the WMT24 data for research use (data/demo_wmt/, "
                   "git-ignored); not redistributed and not for commercial material.",
        "licence_evidence": "https://www2.statmt.org/wmt24/translation-task.html",
        "attribution": "Source text, MT and reference: WMT24 General MT test set, en-es (Kocmi "
                       "et al., 2024), system MSLC (Larkin, Lo and Knowles, NRC Canada, 2024). "
                       "Human MQM labels: Unbabel raters, released by Google in "
                       "google/wmt-mqm-human-evaluation (Apache-2.0; Freitag et al., 2024). "
                       "Research use only.",
        "scope": "All 31 annotated segments of the document's first chunk, in order.",
    },
}

MSLC = {
    "name": "MSLC (WMT24 submission, NRC Canada)",
    "model_id": "MSLC",
    "licence": "WMT24 system output, distributed with the WMT24 test data for research use",
    "attribution": "Larkin, Lo and Knowles, 2024, MSLC24 Submissions to the General Machine "
                   "Translation Task, WMT 2024, https://aclanthology.org/2024.wmt-1.7",
}

LABEL_RULE_ADJ = ("Our adjudication against the official Spanish translation (bilingual reading "
                  "by Claude, two passes: A labels every segment in the review pool, B tries to "
                  "refute every major/unsure call and keeps 'major' only if a reader of the MT "
                  "would be misled about who must do what, when, how much, or what something is). "
                  "'Major' = major_error; minor_error and ok both count as not Major.")
LABEL_RULE_MQM = ("WMT24 human MQM (one Unbabel rater for the whole document): 'Major' = at least "
                  "one major or critical error in the meaning family (accuracy/terminology), the "
                  "training rule of scripts/convert.py.")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def pretty(root) -> None:
    """Indent, but keep seg-source/target children flush (whitespace between <mrk mtype="seg">
    elements would become inter-segment text)."""
    etree.indent(root, space="  ")
    for el in root.iter(f"{{{XLIFF12_NS}}}seg-source", f"{{{XLIFF12_NS}}}target"):
        if len(el):
            el.text = None
            for ch in el:
                ch.tail = None


def serialise(root, bom: bool = False) -> bytes:
    return (b"\xef\xbb\xbf" if bom else b"") + b'<?xml version="1.0" encoding="utf-8"?>\n' + \
        etree.tostring(root, encoding="utf-8") + b"\n"


def header_note(doc_id: str, meta: dict, mt: dict) -> str:
    official = "" if meta.get("wmt") else (
        " The Spanish targets are this machine translation, not the publisher's official Spanish "
        "(that is in demo/answer_key.en-es.json).")
    return (f"{meta['title']}. {meta['attribution']} MT: {mt['name']} ({mt['model_id']}), "
            f"{mt['licence']}.{official} SecondRead demo job, see demo/README.md.")


def build_xliff12(doc_id: str, meta: dict, mt: dict, rows: list[dict]):
    E = lambda tag: f"{{{XLIFF12_NS}}}{tag}"
    root = etree.Element(E("xliff"), version="1.2", nsmap={None: XLIFF12_NS})
    f = etree.SubElement(root, E("file"), {
        "original": f"{doc_id}.{'txt' if meta.get('wmt') else 'html'}",
        "datatype": "plaintext" if meta.get("wmt") else "html",
        "source-language": meta["src_lang"], "target-language": meta["tgt_lang"]})
    header = etree.SubElement(f, E("header"))
    etree.SubElement(header, E("tool"), {"tool-id": "secondread-demo",
                                         "tool-name": "SecondRead demo builder"})
    etree.SubElement(header, E("note")).text = header_note(doc_id, meta, mt)
    body = etree.SubElement(f, E("body"))
    for r in rows:
        tu = etree.SubElement(body, E("trans-unit"), id=r["id"])
        etree.SubElement(tu, E("source")).text = r["src"]
        etree.SubElement(tu, E("target"), {"state": "needs-review-translation",
                                           "state-qualifier": "leveraged-mt"}).text = r["mt"]
        if r.get("section"):
            etree.SubElement(tu, E("note"), {"from": "secondread-demo", "annotates": "source"}
                             ).text = f"Section: {r['section']}"
    pretty(root)
    return root


def build_sdlxliff(doc_id: str, meta: dict, mt: dict, rows: list[dict]):
    """What Trados Studio writes for a file pre-translated by MT: one trans-unit per paragraph,
    one <mrk mtype="seg"> per segment, sdl:seg conf=Draft origin=mt."""
    E = lambda tag: f"{{{XLIFF12_NS}}}{tag}"
    S = lambda tag: f"{{{SDL_NS}}}{tag}"
    root = etree.Element(E("xliff"), {"version": "1.2", S("version"): "1.0"},
                         nsmap={"sdl": SDL_NS, None: XLIFF12_NS})
    f = etree.SubElement(root, E("file"), {
        "original": f"{doc_id}.html", "datatype": "x-sdlfilterframework2",
        "source-language": meta["src_lang"], "target-language": meta["tgt_lang"]})
    header = etree.SubElement(f, E("header"))
    info = etree.SubElement(header, S("file-info"), nsmap={None: SDL_NS})
    for k, v in (("SDL:FileId", str(uuid.uuid5(NS_UUID, doc_id))),
                 ("SDL:OriginalFilePath", f"{doc_id}.html"),
                 ("SDL:OriginalEncoding", "utf-8"),
                 ("SDL:AutoClonedFlagSupported", "True")):
        etree.SubElement(info, S("value"), key=k).text = v
    ft = etree.SubElement(header, S("filetype-info"))
    etree.SubElement(ft, S("filetype-id")).text = "HTML 5 2.0.0.0"
    etree.SubElement(header, E("note")).text = header_note(doc_id, meta, mt)
    body = etree.SubElement(f, E("body"))
    for mid, r in enumerate(rows, 1):
        tu = etree.SubElement(body, E("trans-unit"), id=str(uuid.uuid5(NS_UUID, r["id"])))
        etree.SubElement(tu, E("source")).text = r["src"]
        seg_src = etree.SubElement(tu, E("seg-source"))
        etree.SubElement(seg_src, E("mrk"), mtype="seg", mid=str(mid)).text = r["src"]
        tgt = etree.SubElement(tu, E("target"))
        etree.SubElement(tgt, E("mrk"), mtype="seg", mid=str(mid)).text = r["mt"]
        defs = etree.SubElement(tu, S("seg-defs"))
        etree.SubElement(defs, S("seg"), {"id": str(mid), "conf": "Draft", "origin": "mt",
                                          "origin-system": mt["model_id"]})
    pretty(root)
    return root


def collect() -> dict[str, list[dict]]:
    scored = load_jsonl(REAL / "scored.jsonl")
    adj = {(r["id"], r["system"]): r for r in load_jsonl(REAL / "adjudication_final.jsonl")}
    fill_p = REAL / "adjudication_fill.jsonl"
    for r in load_jsonl(fill_p) if fill_p.exists() else []:
        assert (r["id"], r["system"]) not in adj, f"{r['id']} adjudicated twice"
        adj[(r["id"], r["system"])] = r
    out = {}
    for doc_id, meta in DOCS.items():
        rows = [r for r in scored if r["doc_id"] == doc_id and r["system"] == meta["system"]]
        assert rows, f"no scored rows for {doc_id}/{meta['system']}"
        assert all(r["public_demo"] for r in rows), f"{doc_id}: MT not cleared for public display"
        if meta.get("wmt"):
            gold = {g["id"]: g for g in load_jsonl(REAL / doc_id / "aligned.jsonl")}
            assert [r["id"] for r in rows] == list(gold), f"{doc_id}: rows differ from aligned.jsonl"
        for r in rows:
            a = adj.get((r["id"], r["system"]))
            r["adj"] = a
            if meta.get("wmt"):
                g = gold[r["id"]]
                assert (g["src"], g["mt"]) == (r["src"], r["mt"])
                r["label"] = g["gold_label"]
                r["gold_spans"], r["mqm_spans"] = g["gold_spans"], g["mqm_spans"]
                r["wmt_id"], r["rater"] = g["wmt_id"], g["rater"]
            else:
                assert a is not None, f"{r['id']} has no adjudication; run the fill first"
                r["label"] = "Major" if a["verdict"] == "major_error" else "OK"
        out[doc_id] = rows
    return out


def explain_mqm(r: dict) -> str:
    if r["label"] != "Major":
        return "OK in MQM: the rater marked no major meaning error."
    parts = []
    for s in r["gold_spans"]:
        parts.append(f"{s['category']} on '{s['text']}'" if s.get("text") else s["category"])
    return "MQM Major: " + "; ".join(parts) + "."


def stats(rows: list[dict], label_of) -> dict:
    thr = rows[0]["secondread"]["threshold"]
    maj = [r for r in rows if label_of(r) == "Major"]
    flag = [r for r in rows if r["secondread"]["flagged"]]
    g = lambda r, t: r["gemma"] is not None and r["gemma"]["risk"] >= t  # None: not scored
    by_risk = sorted(rows, key=lambda r: -r["secondread"]["risk"])
    top = max(1, round(0.1 * len(rows)))
    return {
        "segments": len(rows), "major": len(maj), "threshold": thr,
        "sr_flagged": len(flag), "sr_caught": sum(label_of(r) == "Major" for r in flag),
        "sr_false_flags": sum(label_of(r) != "Major" for r in flag),
        "sr_major_in_top10pct": sum(label_of(r) == "Major" for r in by_risk[:top]), "top10pct_n": top,
        "gemma_hi_flagged": sum(g(r, GEMMA_HI) for r in rows),
        "gemma_hi_caught": sum(g(r, GEMMA_HI) for r in maj),
        "gemma_lo_flagged": sum(g(r, GEMMA_LO) for r in rows),
        "gemma_lo_caught": sum(g(r, GEMMA_LO) for r in maj),
        "gemma_missing": sum(r["gemma"] is None for r in rows),
        "rule_hits": sum(bool(r["checks"]["number_check"]["codes"] or r["checks"]["negation_check"]["codes"])
                         for r in rows),
        "majors": [(r["id"], round(r["secondread"]["risk"], 3), r["secondread"]["flagged"],
                    r["gemma"] and round(r["gemma"]["risk"], 3)) for r in maj],
        "false_flags": [(r["id"], round(r["secondread"]["risk"], 3),
                         (r["adj"] or {}).get("verdict")) for r in flag if label_of(r) != "Major"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "demo"),
                    help="public jobs and answer_key.en-es.json (committed)")
    ap.add_argument("--wmt-out", default=str(ROOT / "data" / "demo_wmt"),
                    help="WMT-derived jobs and answer_key.en-es.wmt24.json (git-ignored, research "
                         "use only, never committed)")
    ap.add_argument("--check-only", action="store_true", help="print the numbers, write nothing")
    args = ap.parse_args()
    docs = collect()

    # two answer keys: the public documents (demo/) and the WMT document (data/demo_wmt/)
    keyset = {False: (Path(args.out), [], []), True: (Path(args.wmt_out), [], [])}
    summary = {}
    for doc_id, rows in docs.items():
        meta = DOCS[doc_id]
        mt = MSLC if meta.get("wmt") else MARIAN
        out, key_docs, key_segs = keyset[bool(meta.get("wmt"))]
        files = {"xliff12": (f"secondread-demo.{LP}.{doc_id}.xlf", build_xliff12, False)}
        if meta.get("sdlxliff"):
            files["sdlxliff"] = (f"secondread-demo.{LP}.{doc_id}.sdlxliff", build_sdlxliff, True)
        keys = {}
        for fmt, (name, build, bom) in files.items():
            data = serialise(build(doc_id, meta, mt, rows), bom)  # Trados writes a UTF-8 BOM
            parsed = xliff.load(data, name=name)
            assert parsed.format == fmt and parsed.lp == LP, (name, parsed.format, parsed.lp)
            got = [(s.src, s.tgt) for s in parsed.segments]
            assert got == [(r["src"], r["mt"]) for r in rows], f"{name}: segments do not round-trip"
            assert all(s.src_plain == s.src and s.tgt_plain == s.tgt for s in parsed.segments), name
            keys[fmt] = [s.key for s in parsed.segments]
            if not args.check_only:
                out.mkdir(parents=True, exist_ok=True)
                (out / name).write_bytes(data)
                print(f"{out / name}: {len(data):,} bytes, {len(parsed.segments)} segments")

        label_main = lambda r: r["label"]
        st = stats(rows, label_main)
        if meta.get("wmt"):  # the same document under our stricter adjudication, for comparison
            st_adj = stats([r for r in rows if r["adj"]],
                           lambda r: "Major" if r["adj"]["verdict"] == "major_error" else "OK")
            st["our_adjudication"] = {k: st_adj[k] for k in ("segments", "major", "sr_flagged",
                                                              "sr_caught", "sr_false_flags", "majors")}
        summary[doc_id] = st
        verdicts = Counter((r["adj"] or {}).get("verdict") for r in rows)
        key_docs.append({
            "doc_id": doc_id, "title": meta["title"], **({"title_es": meta["title_es"]} if "title_es" in meta else {}),
            "publisher": meta["publisher"], "domain": meta["domain"], "urls": meta["urls"],
            **({"wmt_doc": meta["wmt_doc"]} if "wmt_doc" in meta else {}),
            "licence": meta["licence"], "licence_evidence": meta["licence_evidence"],
            "attribution": meta["attribution"], "scope": meta["scope"],
            "mt_system": mt,
            "labels": "WMT24 human MQM" if meta.get("wmt") else
                      "our adjudication against the official Spanish translation",
            "label_rule": LABEL_RULE_MQM if meta.get("wmt") else LABEL_RULE_ADJ,
            "files": {fmt: name for fmt, (name, _, _) in files.items()},
            "counts": {"segments": len(rows), "Major": st["major"], "OK": len(rows) - st["major"],
                       **({} if meta.get("wmt") else {
                           "verdicts": {"major_error": verdicts["major_error"],
                                        "minor_error": verdicts["minor_error"], "ok": verdicts["ok"]}})},
        })
        for i, r in enumerate(rows):
            a = r["adj"]
            seg = {"doc_id": doc_id, "id": r["id"], "keys": {fmt: keys[fmt][i] for fmt in files},
                   "section": r.get("section"), "label": r["label"]}
            if meta.get("wmt"):
                seg["explanation"] = explain_mqm(r)
                seg["mqm"] = {"rater": r["rater"], "wmt_id": r["wmt_id"],
                              "major_spans": [{k: s[k] for k in ("start", "end", "text", "category", "severity")}
                                              for s in r["gold_spans"]]}
                if a:
                    seg["our_reading"] = {"verdict": a["verdict"], "error_type": a["error_type"],
                                          "span_mt": a["span_mt"], "explanation": a["explanation"]}
            else:
                seg.update(verdict=a["verdict"], error_type=a["error_type"], span_mt=a["span_mt"],
                           explanation=a["explanation"], label_pass=a.get("final_source"))
            seg.update(src=r["src"], mt=r["mt"],
                       **({"reference_es": r["ref_es"]} if meta.get("wmt") else {"official_es": r["ref_es"]}))
            key_segs.append(seg)

    descriptions = {
        False: "Answer key for the SecondRead en-es demo jobs (real public documents with their "
               "official Spanish). label 'Major' = a meaning error that would mislead a reader; "
               "see each document's 'labels' and 'label_rule'. Explanations quote the MT and the "
               "official Spanish.",
        True: "Answer key for the SecondRead WMT24 en-es demo job, generated locally from the "
              "WMT24 test data (research use only; not redistributed). label 'Major' = at least "
              "one major or critical MQM meaning error; see 'label_rule'.",
    }
    names = {False: f"answer_key.{LP}.json", True: f"answer_key.{LP}.wmt24.json"}
    for wmt, (out, key_docs, key_segs) in keyset.items():
        if not key_docs:
            continue
        key = {
            "description": descriptions[wmt],
            "lp": LP,
            "documents": key_docs,
            "counts": {"segments": len(key_segs), "Major": sum(s["label"] == "Major" for s in key_segs)},
            "segments": key_segs,
        }
        if not args.check_only:
            p = out / names[wmt]
            p.write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"{p}: {len(key_segs)} segments, {key['counts']['Major']} Major")
    for doc_id, st in summary.items():
        print(f"\n{doc_id}: {st['segments']} segments, {st['major']} Major | SecondRead "
              f"(thr {st['threshold']}): {st['sr_flagged']} flagged, {st['sr_caught']} Major caught, "
              f"{st['sr_false_flags']} false flags, {st['sr_major_in_top10pct']} Major in its top "
              f"{st['top10pct_n']} by risk | Gemma >= {GEMMA_HI}: {st['gemma_hi_flagged']} flagged, "
              f"{st['gemma_hi_caught']} caught | Gemma >= {GEMMA_LO}: {st['gemma_lo_flagged']} "
              f"flagged, {st['gemma_lo_caught']} caught | rule hits {st['rule_hits']}"
              + (f" | WARNING: {st['gemma_missing']} rows without a Gemma score" if st["gemma_missing"] else ""))
        print("  Major (id, SR risk, SR flag, Gemma risk):", st["majors"])
        print("  SecondRead false flags (id, risk, our verdict):", st["false_flags"])
        if "our_adjudication" in st:
            o = st["our_adjudication"]
            print(f"  under our adjudication ({o['segments']} adjudicated): {o['major']} Major, "
                  f"SecondRead {o['sr_flagged']} flagged, {o['sr_caught']} caught, "
                  f"{o['sr_false_flags']} false flags; Major: {o['majors']}")


if __name__ == "__main__":
    main()
