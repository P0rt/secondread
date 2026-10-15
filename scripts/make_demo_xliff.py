"""Build the SecondRead demo job from public WMT22 en-ru test data, as XLIFF 1.2, XLIFF 2.0 and
Trados SDLXLIFF, plus a gold answer key.

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/make_demo_xliff.py
           [--split data/splits/en-ru.test.jsonl] [--system eTranslation] [--docs DOC,DOC,...]
           [--out data/demo_wmt]

WMT releases its test sets for research use only, so this job is generated locally into the
git-ignored data/demo_wmt/ and is not redistributed with the repository (demo/ holds only
openly reusable documents, see demo/README.md).

One MT system, whole documents, segments in document order. WMT22 MQM documents are
10-segment excerpts, so ~50 segments takes five of them. All documents come from the held-out
test split (checked against data/splits/en-ru.docs.json), so the demo never shows training data.
Every output file is parsed back with secondread.xliff and must give the same segments in the
same order; the answer key records each segment's key in every format.
"""
import argparse
import json
import uuid
from collections import Counter
from pathlib import Path

from lxml import etree

from secondread import xliff
from secondread.xliff import SDL_NS, XLIFF12_NS, XLIFF2_NS

# Mixed job: two product listings, a support chat, two news articles. ~27% Major, close to
# eTranslation's 26.7% over the whole en-ru test split.
DEFAULT_DOCS = [
    "ecommerce_en_36",  # watch dial listing
    "ecommerce_en_13",  # dog joint supplement listing
    "en_fr_CLIENT-02_default_2020-12-20-16_doc06",  # e-reader support chat
    "brisbanetimes.com.au.47160",  # cricket broadcast power failure
    "guardian.44758",  # Libyan election
]
LANGS = {"en": "en-US", "ru": "ru-RU", "es": "es-ES", "de": "de-DE"}
NS_UUID = uuid.UUID("6f1d3a52-6a0e-4f7e-9a59-5ecd0d2ead00")  # stable ids across rebuilds


def seg_no(row) -> int:
    return int(row["id"].split("|")[2])  # id = lp|system|global segment number|rater


def untsv(text: str, spans: list, side: str) -> str:
    """Undo CSV quoting left in a few WMT TSV fields (a field wrapped in quotes with inner
    quotes doubled), but only when no gold span on that side would shift."""
    if (len(text) > 1 and text[0] == text[-1] == '"' and '""' in text
            and not any(s["side"] == side and s["start"] >= 0 for s in spans)):
        return text[1:-1].replace('""', '"')
    return text


def pretty(root) -> None:
    """Indent the tree, but keep seg-source/target flush: whitespace between <mrk mtype="seg">
    elements would become inter-segment text."""
    etree.indent(root, space="  ")
    for el in root.iter(f"{{{XLIFF12_NS}}}seg-source", f"{{{XLIFF12_NS}}}target"):
        if len(el):
            el.text = None
            for ch in el:
                ch.tail = None


def build_xliff12(docs, src_lang, tgt_lang, system):
    E = lambda tag: f"{{{XLIFF12_NS}}}{tag}"
    root = etree.Element(E("xliff"), version="1.2", nsmap={None: XLIFF12_NS})
    for doc, rows in docs:
        f = etree.SubElement(root, E("file"), {"original": f"{doc}.txt", "datatype": "plaintext",
                                               "source-language": src_lang,
                                               "target-language": tgt_lang})
        header = etree.SubElement(f, E("header"))
        etree.SubElement(header, E("tool"), {"tool-id": "secondread-demo",
                                             "tool-name": "SecondRead demo builder"})
        etree.SubElement(header, E("note")).text = (
            f"WMT22 General MT test set, document {doc} ({rows[0]['domain']}), "
            f"MT by {system}. Research use only, see demo/README.md.")
        body = etree.SubElement(f, E("body"))
        for r in rows:
            tu = etree.SubElement(body, E("trans-unit"), id=str(seg_no(r)))
            etree.SubElement(tu, E("source")).text = r["src"]
            etree.SubElement(tu, E("target"), {"state": "needs-review-translation",
                                               "state-qualifier": "leveraged-mt"}).text = r["mt"]
    pretty(root)
    return root


def build_xliff20(docs, src_lang, tgt_lang, system):
    E = lambda tag: f"{{{XLIFF2_NS}}}{tag}"
    root = etree.Element(E("xliff"), version="2.0", srcLang=src_lang, trgLang=tgt_lang,
                         nsmap={None: XLIFF2_NS})
    for i, (doc, rows) in enumerate(docs, 1):
        f = etree.SubElement(root, E("file"), id=f"f{i}", original=f"{doc}.txt")
        notes = etree.SubElement(f, E("notes"))
        etree.SubElement(notes, E("note"), category="source").text = (
            f"WMT22 General MT test set, document {doc} ({rows[0]['domain']}), "
            f"MT by {system}. Research use only, see demo/README.md.")
        for r in rows:
            unit = etree.SubElement(f, E("unit"), id=str(seg_no(r)))
            seg = etree.SubElement(unit, E("segment"), id="1", state="translated")
            etree.SubElement(seg, E("source")).text = r["src"]
            etree.SubElement(seg, E("target")).text = r["mt"]
    pretty(root)
    return root


def build_sdlxliff(docs, src_lang, tgt_lang, system):
    """Mimics what Trados Studio writes for a plain-text file pre-translated by MT: one
    trans-unit per line, one <mrk mtype="seg"> per segment, sdl:seg conf=Draft origin=mt."""
    E = lambda tag: f"{{{XLIFF12_NS}}}{tag}"
    S = lambda tag: f"{{{SDL_NS}}}{tag}"
    root = etree.Element(E("xliff"), {"version": "1.2", S("version"): "1.0"},
                         nsmap={"sdl": SDL_NS, None: XLIFF12_NS})
    mid = 0
    for doc, rows in docs:
        f = etree.SubElement(root, E("file"), {
            "original": f"{doc}.txt", "datatype": "x-sdlfilterframework2",
            "source-language": src_lang, "target-language": tgt_lang})
        header = etree.SubElement(f, E("header"))
        info = etree.SubElement(header, S("file-info"), nsmap={None: SDL_NS})
        for k, v in (("SDL:FileId", str(uuid.uuid5(NS_UUID, doc))),
                     ("SDL:OriginalFilePath", f"{doc}.txt"),
                     ("SDL:OriginalEncoding", "utf-8"),
                     ("SDL:AutoClonedFlagSupported", "True")):
            etree.SubElement(info, S("value"), key=k).text = v
        ft = etree.SubElement(header, S("filetype-info"))
        etree.SubElement(ft, S("filetype-id")).text = "Plain Text v 1.0.0.0"
        body = etree.SubElement(f, E("body"))
        for r in rows:
            mid += 1
            tu = etree.SubElement(body, E("trans-unit"), id=str(uuid.uuid5(NS_UUID, r["id"])))
            etree.SubElement(tu, E("source")).text = r["src"]
            seg_src = etree.SubElement(tu, E("seg-source"))
            etree.SubElement(seg_src, E("mrk"), mtype="seg", mid=str(mid)).text = r["src"]
            tgt = etree.SubElement(tu, E("target"))
            etree.SubElement(tgt, E("mrk"), mtype="seg", mid=str(mid)).text = r["mt"]
            defs = etree.SubElement(tu, S("seg-defs"))
            etree.SubElement(defs, S("seg"), {"id": str(mid), "conf": "Draft", "origin": "mt",
                                              "origin-system": f"{system} (WMT22)"})
    pretty(root)
    return root


def write(root, path: Path, bom: bool = False) -> bytes:
    data = (b"\xef\xbb\xbf" if bom else b"") + b'<?xml version="1.0" encoding="utf-8"?>\n' + \
        etree.tostring(root, encoding="utf-8") + b"\n"
    path.write_bytes(data)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="data/splits/en-ru.test.jsonl")
    ap.add_argument("--system", default="eTranslation")
    ap.add_argument("--docs", default=",".join(DEFAULT_DOCS))
    ap.add_argument("--out", default="data/demo_wmt",
                    help="output folder; keep it git-ignored (WMT text, research use only)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.split, encoding="utf-8")]
    lp = rows[0]["lp"]
    split_of = json.load(open(Path(args.split).parent / f"{lp}.docs.json"))
    docs = []
    for doc in args.docs.split(","):
        assert split_of.get(doc) == "test", f"{doc} is not a test-split document"
        rs = sorted((r for r in rows if r["doc"] == doc and r["system"] == args.system), key=seg_no)
        assert rs, f"no {args.system} rows for {doc}"
        nums = [seg_no(r) for r in rs]
        assert nums == list(range(nums[0], nums[0] + len(nums))), f"gap in {doc}: {nums}"
        for r in rs:
            r["src"] = untsv(r["src"], r["spans"], "src")
            r["mt"] = untsv(r["mt"], r["spans"], "mt")
        docs.append((doc, rs))
    src_lang, tgt_lang = (LANGS[x] for x in lp.split("-"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"secondread-demo.{lp}"
    files = {
        "xliff12": (f"{stem}.xliff12.xlf", build_xliff12, False),
        "xliff20": (f"{stem}.xliff20.xlf", build_xliff20, False),
        "sdlxliff": (f"{stem}.sdlxliff", build_sdlxliff, True),  # Trados writes a UTF-8 BOM
    }
    flat = [r for _, rs in docs for r in rs]
    keys = {}
    for fmt, (name, build, bom) in files.items():
        data = write(build(docs, src_lang, tgt_lang, args.system), out / name, bom)
        parsed = xliff.load(data, name=name)
        assert parsed.format == fmt and parsed.lp == lp, (fmt, parsed.format, parsed.lp)
        got = [(s.src, s.tgt) for s in parsed.segments]
        assert got == [(r["src"], r["mt"]) for r in flat], f"{fmt}: segments do not round-trip"
        keys[fmt] = [s.key for s in parsed.segments]
        print(f"{out / name}: {len(data):,} bytes, {len(parsed.segments)} segments")

    labels = Counter(r["label"] for r in flat)
    key = {
        "description": "Gold labels for the SecondRead demo job. 'Major' = at least one major or "
                       "critical MQM error in the meaning family (accuracy/terminology: "
                       "mistranslation, omission, addition, untranslated, wrong named entity, "
                       "number/date/unit formats ...), the same rule as training "
                       "(scripts/convert.py). One MQM annotator per segment, so expect label noise.",
        "source": "WMT22 General MT test set (en-ru) with MQM annotations from "
                  "google/wmt-mqm-human-evaluation. Research use only; see demo/README.md.",
        "lp": lp, "system": args.system,
        "documents": [{"doc": d, "domain": rs[0]["domain"], "segments": len(rs),
                       "major": sum(r["label"] == "Major" for r in rs)} for d, rs in docs],
        "files": {fmt: name for fmt, (name, _, _) in files.items()},
        "counts": {"segments": len(flat), "Major": labels["Major"], "OK": labels["OK"]},
        "segments": [{
            "n": i + 1,
            "keys": {fmt: keys[fmt][i] for fmt in files},
            "doc": r["doc"], "domain": r["domain"], "wmt_segment": seg_no(r),
            "label": r["label"], "label_any_major": r["label_any_major"],
            "src": r["src"], "mt": r["mt"],
            "errors": [{k: s[k] for k in ("side", "start", "end", "text", "category", "family",
                                          "severity")} for s in r["spans"]],
        } for i, r in enumerate(flat)],
    }
    (out / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    print(f"{out / 'answer_key.json'}: {len(flat)} segments, {labels['Major']} Major "
          f"({100 * labels['Major'] / len(flat):.1f}%)")


if __name__ == "__main__":
    main()
