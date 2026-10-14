"""Tests for secondread.xliff (go/no-go test G10).

Runs under pytest, or without it (pytest is not a project dependency yet):
    HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python tests/test_xliff.py
Set SECONDREAD_OKAPI_SDLXLIFF=/path/to/test.txt_en-US_fr-FR.sdlxliff to also check the real
Okapi sample (Apache-2.0, okapi/filters/xliff test resources); it is skipped otherwise.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secondread import xliff  # noqa: E402
from secondread.xliff import SDL_NS, SR_NS, XLIFF12_NS, XLIFF2_NS, XliffError  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
DEMO = Path(__file__).resolve().parents[1] / "demo"
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
FIXTURES = ["sample12.xlf", "sample20.xlf", "trados_like.sdlxliff"]
SDL = f"{{{SDL_NS}}}"
X12 = f"{{{XLIFF12_NS}}}"


def c14n(data: bytes) -> bytes:
    return etree.tostring(etree.fromstring(data), method="c14n")


def c14n_exclusive(data: bytes) -> bytes:
    """Canonical XML without unused namespace declarations."""
    return etree.tostring(etree.fromstring(data), method="c14n", exclusive=True)


def raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        return e
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def alternating(doc, hi=0.9, lo=0.1):
    return {s.key: hi if i % 2 == 0 else lo for i, s in enumerate(doc.segments)}


# -- reading -----------------------------------------------------------------------------------

def test_xliff12_segments():
    doc = xliff.load(FIX / "sample12.xlf")
    assert doc.format == "xliff12" and doc.lp == "en-ru"
    # tu3 is translate="no"; alt-trans source/target must not leak in; tu5 has two mrk segments
    assert [s.key for s in doc.segments] == ["0:tu1:1", "0:tu2:1", "0:tu4:1", "0:tu5:0",
                                             "0:tu5:1"]
    s = {x.key: x for x in doc.segments}
    tu1 = s["0:tu1:1"]
    assert tu1.src == "Do {1}not{/1} remove the battery."
    assert tu1.tgt == "Извлеките {1}аккумулятор{/1}."
    assert tu1.src_plain == "Do not remove the battery."
    assert tu1.src_tags == {"{1}": '<g id="1" ctype="bold">', "{/1}": "</g>"}
    assert (tu1.state, tu1.origin) == ("translated", "leveraged-mt")
    tu2 = s["0:tu2:1"]
    assert tu2.src == "Press {1}Start{/1} and wait 5\xa0seconds.{2}Then release."
    assert tu2.tgt == "Нажмите {1}Пуск{/1} и подождите 15\xa0секунд.{2}Затем отпустите."
    assert tu2.src_plain == "Press Start and wait 5\xa0seconds. Then release."  # NBSP kept
    assert tu2.src_tags["{1}"] == "<bpt id=\"1\">&lt;b&gt;</bpt>"
    assert tu2.src_tags["{2}"] == '<x id="2" ctype="lb"/>'
    assert tu2.state == "needs-review-translation"
    tu4 = s["0:tu4:1"]
    assert tu4.tgt is None and tu4.tgt_plain is None
    assert tu4.src == "Price: {1} per unit." and tu4.src_plain == "Price: per unit."
    assert (s["0:tu5:0"].src, s["0:tu5:0"].tgt) == ("Two sentences.", "Два предложения.")
    assert (s["0:tu5:1"].src, s["0:tu5:1"].tgt) == ("Second one.", "Второе.")
    assert [x.index for x in doc.segments] == [1, 2, 3, 4, 5]


def test_xliff20_segments():
    doc = xliff.load(FIX / "sample20.xlf")
    assert doc.format == "xliff20" and doc.lp == "en-ru"
    assert [s.key for s in doc.segments] == ["0:u1:s1", "0:u1:s2", "0:u2:1", "0:u4:s1"]
    s1, s2, u2, u4 = doc.segments
    assert (s1.src, s1.tgt) == ("Do {1}not{/1} touch.", "Не {1}трогайте{/1}.")
    assert s1.src_tags["{1}"].startswith("<pc id=\"1\"")
    assert s2.src == "Line one{1}line two." and s2.src_plain == "Line one line two."
    assert s2.tgt_plain == "Строка один строка два."
    assert (s1.state, s2.state) == ("translated", "initial")
    # sc/ec pair up via startRef; a term <mrk> is transparent
    assert u2.src == "{1}Save{/1} changes?" and u2.tgt == "{1}Сохранить{/1} изменения?"
    assert u4.tgt is None and u4.src == "Untranslated\x0btext."  # <cp hex="000B"/>


def test_sdlxliff_trados_like():
    """Hand-crafted per the SDLXLIFF layout Trados Studio writes: BOM + CRLF, doc-info with
    track-change and comment defs, groups with sdl:cxts, seg mrks nested in <g>, locked and
    empty segments, a segment missing from the target, translate="no" structure units, two
    <file>s."""
    doc = xliff.load(FIX / "trados_like.sdlxliff")
    assert doc.format == "sdlxliff" and doc.lp == "en-ru"
    segs = doc.segments
    assert len(segs) == 8  # the two translate="no" structure units are skipped
    assert [s.seg_id for s in segs] == ["1", "2", "3", "4", "5", "6", "7", "1"]
    assert segs[-1].key.startswith("1:")  # second <file>: mid restarts, key stays unique
    assert len({s.key for s in segs}) == 8
    s1, s2, s3, s4, s5, s6, s7, f2 = segs
    assert s1.src == "Do {1}not{/1} remove the battery while the device is on."
    assert s1.tgt == "Извлеките {1}аккумулятор{/1}, пока устройство включено."  # human comment kept
    assert s1.tgt_tags == {"{1}": '<g id="pt1">', "{/1}": "</g>"}
    assert (s1.state, s1.origin, s1.locked) == ("Draft", "mt", False)
    assert s2.tgt == "Подождите 10 секунд."  # track changes applied: "1" deleted, "10" added
    assert (s2.state, s2.origin) == ("Translated", "interactive")
    assert (s3.src, s3.tgt, s3.locked, s3.match_percent) == ("Warning:", "Внимание:", True, 100.0)
    assert s4.src == "keep away from children under 3."
    assert s5.tgt == "" and s5.state is None
    assert (s6.state, s6.origin, s6.match_percent) == ("Translated", "tm", 87.0)
    assert s7.tgt is None
    assert f2.file_original.endswith("faq.html") and f2.tgt == "Гарантия действует 12 месяцев."


def test_memoq_flavour():
    data = f'''<?xml version="1.0" encoding="utf-8"?>
<xliff version="1.2" xmlns="{XLIFF12_NS}" xmlns:mq="MQXliff">
 <file original="a.docx" source-language="eng" target-language="rus" datatype="x-docx">
  <body>
   <trans-unit id="1" mq:status="ManuallyConfirmed" mq:locked="locked">
    <source>Hello</source><target>Привет</target>
   </trans-unit>
   <trans-unit id="2" mq:status="NotStarted"><source>Bye</source><target/></trans-unit>
  </body>
 </file>
</xliff>'''.encode()
    a, b = xliff.load(data).segments
    assert (a.state, a.locked, b.state, b.tgt) == ("ManuallyConfirmed", True, "NotStarted", "")


def test_namespaceless_xliff12():
    data = b'<xliff version="1.2"><file source-language="en" target-language="es"><body>' \
           b'<trans-unit id="a"><source>No <g id="1">smoking</g></source>' \
           b'<target>Prohibido <g id="1">fumar</g></target></trans-unit></body></file></xliff>'
    doc = xliff.load(data)
    assert doc.format == "xliff12" and doc.lp == "en-es"
    assert doc.segments[0].tgt == "Prohibido {1}fumar{/1}"
    doc.annotate({"0:a:1": 0.8}, now=NOW)
    again = xliff.load(doc.to_bytes())
    assert again.segments[0].risk == 0.8
    assert again.root.find("file/body/trans-unit/note").get("from") == "SecondRead"


def test_untrusted_input_is_not_expanded():
    secret = Path(__file__).resolve()
    data = f'''<?xml version="1.0"?>
<!DOCTYPE xliff [<!ENTITY xxe SYSTEM "file://{secret}"><!ENTITY boom "BOOM">]>
<xliff version="1.2" xmlns="{XLIFF12_NS}"><file source-language="en" target-language="ru">
<body><trans-unit id="1"><source>a &xxe; b &boom;</source><target>x</target></trans-unit>
</body></file></xliff>'''.encode()
    seg = xliff.load(data).segments[0]
    assert "import" not in seg.src and "BOOM" not in seg.src  # neither entity was expanded
    assert seg.src == "a &xxe; b &boom;"


def test_errors():
    raises(XliffError, xliff.load, b"<html><body/></html>")
    raises(XliffError, xliff.load, b"<xliff version='1.2'><file>")
    doc = xliff.load(FIX / "sample12.xlf")
    raises(KeyError, doc.annotate, {"nope": 0.5})
    for bad in (1.5, -0.1, float("nan"), "high", None):
        raises(ValueError, doc.annotate, {"0:tu1:1": bad})
    raises(ValueError, doc.annotate, {"0:tu1:1": 0.5}, comments="some")


# -- writing -----------------------------------------------------------------------------------

def test_noop_roundtrip_preserves_file():
    for name in FIXTURES:
        raw = (FIX / name).read_bytes()
        out = xliff.load(FIX / name).to_bytes()
        assert c14n(out) == c14n(raw), name
        assert out.startswith(raw[:raw.index(b"?>") + 2]), name  # BOM + declaration verbatim
        assert (b"\r\n" in out) == (b"\r\n" in raw), name
        assert (out.count(b"\n") == out.count(b"\r\n")) == (raw.count(b"\n") == raw.count(b"\r\n"))
        assert b"secondread" not in out and b"xmlns:mda" not in out, name  # nothing added
        top_comment = b"<!-- Hand-made"
        assert (top_comment in out) == (top_comment in raw), name


def test_annotate_roundtrip_and_idempotence():
    for name in FIXTURES:
        doc = xliff.load(FIX / name)
        before = [(s.key, s.src, s.tgt) for s in doc.segments]
        noop = doc.to_bytes()
        risks = {s.key: r for s, r in zip(doc.segments, [0.91, 0.12, 0.55, 0.49, 0.8, 0.3, 0.7, 1.0])}
        doc.annotate(risks, threshold=0.5, now=NOW)
        out = doc.to_bytes()
        again = xliff.load(out, name=name)
        assert [(s.key, s.src, s.tgt) for s in again.segments] == before, name
        for s in again.segments:  # scores read back for every segment that has a target
            expect = risks[s.key] if s.tgt else None
            assert s.risk == expect, (name, s.key, s.risk, expect)
        # annotating an annotated file replaces, never duplicates
        again.annotate(risks, threshold=0.5, now=NOW)
        assert again.to_bytes() == out, name
        # clearing in the same session gives back the untouched bytes exactly...
        doc.clear_annotations()
        assert doc.to_bytes() == noop, name
        assert all(s.risk is None for s in doc.segments)
        # ...and on a reloaded file the same XML; only an unused xmlns:mda may stay behind,
        # since that namespace is not ours to remove
        again.clear_annotations()
        cleared = again.to_bytes()
        assert c14n_exclusive(cleared) == c14n_exclusive(noop), name
        assert b"secondread" not in cleared.lower() and b"SecondRead" not in cleared, name


def test_xliff12_note_placement():
    data = f'''<xliff version="1.2" xmlns="{XLIFF12_NS}" xmlns:ext="urn:example:ext">
  <file source-language="en" target-language="ru">
    <body>
      <trans-unit id="1">
        <source>Ten</source>
        <target>Десять</target>
        <note from="pm">Keep short.</note>
        <ext:data>tool-specific</ext:data>
      </trans-unit>
    </body>
  </file>
</xliff>'''.encode()
    doc = xliff.load(data)
    assert doc.annotate({"0:1:1": 0.97}, details={"0:1:1": "number 10 vs 15"}) == 1
    tu = etree.fromstring(doc.to_bytes()).find(f".//{X12}trans-unit")
    kids = [etree.QName(c).localname for c in tu]
    assert kids == ["source", "target", "note", "note", "data"], kids  # before extensions
    note = tu[3]
    assert note.get("from") == "SecondRead" and note.get("annotates") == "target"
    assert note.text == "SecondRead: possible major meaning error (risk 0.97). number 10 vs 15"
    assert tu.get(f"{{{SR_NS}}}risk") == "0.9700" and tu.get(f"{{{SR_NS}}}label") == "Major"
    assert "\n        <note from=\"SecondRead\"" in doc.to_bytes().decode()  # indentation mirrored


def test_comment_modes():
    doc = xliff.load(FIX / "sample12.xlf")
    keys = [s.key for s in doc.segments if s.tgt]
    assert doc.annotate({k: 0.2 for k in keys}, comments="flagged") == 0
    assert doc.annotate({k: 0.2 for k in keys}, comments="all") == len(keys)
    assert doc.annotate({k: 0.9 for k in keys}, comments="none") == 0
    out = xliff.load(doc.to_bytes())
    assert all(s.risk == 0.9 for s in out.segments if s.tgt)  # scores still stored


def test_xliff20_structure():
    doc = xliff.load(FIX / "sample20.xlf")
    doc.annotate({"0:u1:s1": 0.9, "0:u1:s2": 0.2, "0:u2:1": 0.75}, now=NOW)
    root = etree.fromstring(doc.to_bytes())
    ns = {"x": XLIFF2_NS, "mda": "urn:oasis:names:tc:xliff:metadata:2.0"}
    u1 = root.find(".//x:unit[@id='u1']", ns)
    order = [etree.QName(c).localname for c in u1]
    # Core schema order: extensions/modules, notes, originalData, segment|ignorable
    assert order[:3] == ["metadata", "notes", "originalData"], order
    groups = u1.findall("mda:metadata/mda:metaGroup", ns)
    assert [[m.text for m in g] for g in groups] == [["s1", "0.9000", "Major"],
                                                    ["s2", "0.2000", "OK"]]
    notes = u1.findall("x:notes/x:note", ns)
    assert len(notes) == 1 and notes[0].get("category") == "SecondRead"
    assert notes[0].text.startswith("[segment s1] SecondRead: possible major")
    assert root.find(".//x:file/x:notes/x:note", ns).get("id") == "n1"  # file note untouched
    u2 = root.find(".//x:unit[@id='u2']", ns)
    assert u2.find("x:notes/x:note", ns).text.startswith("SecondRead:")  # single segment: no prefix


def test_sdl_comment_structure():
    doc = xliff.load(FIX / "trados_like.sdlxliff")
    k = {s.seg_id + ("b" if s.key.startswith("1:") else ""): s.key for s in doc.segments}
    risks = {k["1"]: 0.93, k["2"]: 0.1, k["4"]: 0.88, k["5"]: 0.9, k["7"]: 0.9, k["1b"]: 0.97}
    assert doc.annotate(risks, now=NOW) == 3  # 5 is empty and 7 has no target: no comment
    out = doc.to_bytes()
    assert out.startswith(b"\xef\xbb\xbf") and b"\r\n" in out
    root = etree.fromstring(out)
    info = root[0]
    assert info.tag == f"{SDL}doc-info"  # still the first child of <xliff>
    assert [etree.QName(c).localname for c in info] == ["rev-defs", "cmt-defs"]
    defs = {d.get("id"): d for d in info.find(f"{SDL}cmt-defs")}
    assert len(defs) == 4  # the reviewer's comment + ours
    ours = {i: d for i, d in defs.items() if d.find("Comments/Comment").get("user") == "SecondRead"}
    assert len(ours) == 3
    for d in ours.values():
        comments = d.find("Comments")  # Trados keeps <Comments> in no namespace
        assert comments is not None and comments.tag == "Comments"
        c = comments.find("Comment")
        assert c.get("severity") == "Medium" and c.get("version") == "1.0"
        assert c.get("date") == "2026-10-14T12:00:00.000+00:00"
    # each comment is anchored by an x-sdl-comment marker spanning the whole target segment
    for cid in ours:
        marker = root.find(f".//{X12}mrk[@mtype='x-sdl-comment'][@{SDL}cid='{cid}']")
        seg_mrk = marker.getparent()
        assert seg_mrk.get("mtype") == "seg" and len(seg_mrk) == 1 and not seg_mrk.text
        assert seg_mrk.getparent().tag in (f"{X12}target", f"{X12}g")
    # scores on the target seg mrk
    m4 = root.find(f".//{X12}target//{X12}mrk[@mid='4']")
    assert m4.get(f"{{{SR_NS}}}risk") == "0.8800" and m4.get(f"{{{SR_NS}}}label") == "Major"
    # the reviewer's comment marker inside segment 1 survived (now nested in ours)
    assert root.find(f".//{X12}mrk[@{SDL}cid='3f6b8f1e-5a2c-4d47-9f0e-2c8a1b7d6e54']") is not None
    # SDL never gets XLIFF <note>s
    assert root.find(f".//{X12}trans-unit/{X12}note") is None


def test_sdl_doc_info_created_when_missing():
    doc = xliff.load(FIX / "trados_like.sdlxliff")
    info = doc.root.find(f"{SDL}doc-info")
    doc.root.remove(info)  # simulate a file Trados wrote without any comments or revisions
    doc.annotate({doc.segments[0].key: 0.9}, now=NOW)
    root = etree.fromstring(doc.to_bytes())
    assert root[0].tag == f"{SDL}doc-info"
    assert root[0][0].tag == f"{SDL}cmt-defs" and root[1].tag == f"{X12}file"
    assert root.find(f"{SDL}doc-info/{SDL}cmt-defs/{SDL}cmt-def/Comments/Comment") is not None


def test_prefix_collision():
    data = f'''<xliff version="1.2" xmlns="{XLIFF12_NS}" xmlns:sr="urn:someone:else">
<file source-language="en" target-language="ru"><body>
<trans-unit id="1" sr:keep="yes"><source>One</source><target>Один</target></trans-unit>
</body></file></xliff>'''.encode()
    doc = xliff.load(data)
    doc.annotate({"0:1:1": 0.6})
    out = doc.to_bytes()
    root = etree.fromstring(out)
    tu = root.find(f".//{X12}trans-unit")
    assert tu.get("{urn:someone:else}keep") == "yes" and tu.get(f"{{{SR_NS}}}risk") == "0.6000"
    assert b'xmlns:srx="urn:secondread:xliff:1"' in out


def test_sdl_reply_to_our_comment_survives():
    doc = xliff.load(FIX / "trados_like.sdlxliff")
    key = doc.segments[3].key  # mid 4, plain segment
    doc.annotate({key: 0.9}, now=NOW)
    reply = etree.SubElement(next(c.getparent() for c in doc.root.iter("Comment")
                                  if c.get("user") == "SecondRead"), "Comment",
                             severity="Low", user="ACME\\translator", version="1.0")
    reply.text = "Checked, the client wants it this way."
    doc.clear_annotations()
    root = etree.fromstring(doc.to_bytes())
    users = [c.get("user") for c in root.iter("Comment")]
    assert "SecondRead" not in users and "ACME\\translator" in users
    m4 = root.find(f".//{X12}target//{X12}mrk[@mid='4']")
    assert m4[0].get("mtype") == "x-sdl-comment"  # the thread's marker is still anchored


def test_utf16_input():
    raw = (FIX / "sample12.xlf").read_text(encoding="utf-8").replace(
        'encoding="UTF-8"', 'encoding="UTF-16"').encode("utf-16")
    doc = xliff.load(raw)
    assert len(doc.segments) == 5 and doc.segments[0].tgt == "Извлеките {1}аккумулятор{/1}."
    doc.annotate({doc.segments[0].key: 0.9})
    out = doc.to_bytes()
    assert out.decode("utf-16").startswith("<?xml")
    assert xliff.load(out).segments[0].risk == 0.9


def test_reannotating_a_saved_file():
    doc = xliff.load(FIX / "trados_like.sdlxliff")
    doc.annotate(alternating(doc), now=NOW)
    saved = doc.to_bytes()
    doc2 = xliff.load(saved, name="job.sdlxliff")
    assert doc2.segments[0].risk == 0.9
    doc2.annotate({s.key: 0.95 for s in doc2.segments}, now=NOW)
    root = etree.fromstring(doc2.to_bytes())
    ours = [c for c in root.iter("Comment") if c.get("user") == "SecondRead"]
    flagged = [s for s in doc2.segments if s.tgt]
    assert len(ours) == len(flagged)  # old SecondRead comments were replaced, not stacked
    assert doc2.to_bytes().count(b'xmlns:sr="') == 1


# -- demo and real-world samples -----------------------------------------------------------------

def test_demo_files_agree():
    key_path = DEMO / "answer_key.json"
    if not key_path.exists():
        print("  (skipped: run scripts/make_demo_xliff.py first)")
        return
    key = json.loads(key_path.read_text(encoding="utf-8"))
    parsed = {fmt: xliff.load(DEMO / name) for fmt, name in key["files"].items()}
    texts = {fmt: [(s.src, s.tgt) for s in d.segments] for fmt, d in parsed.items()}
    assert texts["xliff12"] == texts["xliff20"] == texts["sdlxliff"]
    assert len(texts["xliff12"]) == key["counts"]["segments"] == len(key["segments"])
    for fmt, d in parsed.items():
        assert d.format == fmt and d.lp == key["lp"]
        assert [s.key for s in d.segments] == [g["keys"][fmt] for g in key["segments"]]
        assert all(s.origin == "mt" for s in d.segments if fmt == "sdlxliff")
    assert [(g["src"], g["mt"]) for g in key["segments"]] == texts["xliff12"]
    assert sum(g["label"] == "Major" for g in key["segments"]) == key["counts"]["Major"]


def test_okapi_sample():
    path = os.environ.get("SECONDREAD_OKAPI_SDLXLIFF")
    if not path:
        print("  (skipped: set SECONDREAD_OKAPI_SDLXLIFF)")
        return
    doc = xliff.load(path)
    assert doc.format == "sdlxliff" and doc.lp == "en-fr"
    assert [s.seg_id for s in doc.segments] == ["1", "2", "3", "4", "5"]  # not one merged paragraph
    assert [s.origin for s in doc.segments] == ["interactive", "mt", "tm", "document-match",
                                                "document-match"]
    assert [s.locked for s in doc.segments] == [False, False, False, True, False]
    doc.annotate({s.key: 0.9 for s in doc.segments}, now=NOW)
    again = xliff.load(doc.to_bytes(), name="x.sdlxliff")
    assert [(s.src, s.tgt) for s in again.segments] == [(s.src, s.tgt) for s in doc.segments]


if __name__ == "__main__":
    import traceback
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"ok    {n}")
        except Exception:
            failed += 1
            print(f"FAIL  {n}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
