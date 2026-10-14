"""Read XLIFF 1.2, XLIFF 2.x and Trados SDLXLIFF segments; write SecondRead risk scores back.

    from secondread import xliff
    doc = xliff.load("job.sdlxliff")
    for seg in doc.segments:                      # document order, translate="no" units skipped
        if seg.tgt_plain and not seg.locked:
            risks[seg.key] = score(seg.src_plain, seg.tgt_plain, doc.lp)
    doc.annotate(risks, threshold=0.5)            # idempotent: replaces earlier SecondRead output
    doc.save("job.secondread.sdlxliff")

Segments
    XLIFF 1.2   one <trans-unit> = one segment, or one segment per <mrk mtype="seg"> when the unit
                has a <seg-source> (Okapi, memoQ, Trados). Target <mrk>s are aligned by `mid`.
    SDLXLIFF    XLIFF 1.2 + <sdl:seg-defs>: conf -> state, origin (mt/tm/interactive...),
                locked, percent. Track changes are applied (x-sdl-deleted text is dropped).
    XLIFF 2.x   one <segment> = one segment; <ignorable> is skipped.

Inline codes become placeholders, numbered per segment and shared by source and target (the same
code id gets the same number): paired codes are `{1}…{/1}`, standalone codes `{2}`. `src_tags` /
`tgt_tags` map each placeholder to the original XML. `src_plain` / `tgt_plain` drop the codes and
collapse whitespace; feed those to the model and the regex checks ({1} looks like a number).

Write-back adds, never rewrites, translation text (the one exception: SDL comments must wrap the
commented text in a marker element). What is written:
    xliff12   <note from="SecondRead" annotates="target"> in the trans-unit, and sr:risk / sr:label
              on the trans-unit (or on the target <mrk mtype="seg"> of a segmented unit)
    xliff20   <note category="SecondRead" appliesTo="target"> in the unit's <notes>, and one
              <mda:metaGroup category="SecondRead"> per segment (XLIFF 2 Metadata module)
    sdlxliff  Trados comments (<doc-info><cmt-defs>, <mrk mtype="x-sdl-comment" sdl:cid>), and
              sr:risk / sr:label on the target <mrk mtype="seg">
`sr` is our own namespace (SR_NS). Untouched files serialise back unchanged apart from lxml's
normalisation (attribute quotes, character references); BOM, XML declaration and CRLF line
endings are preserved.

Client files are untrusted: the parser loads no DTDs, expands no entities and never hits the network.

CLI (dump segments as JSON lines): python -m secondread.xliff FILE
"""
from __future__ import annotations

import math
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping
from xml.sax.saxutils import escape, quoteattr

from lxml import etree

XLIFF12_NS = "urn:oasis:names:tc:xliff:document:1.2"
XLIFF2_NS = "urn:oasis:names:tc:xliff:document:2.0"  # XLIFF 2.1 keeps the 2.0 core namespace
MDA_NS = "urn:oasis:names:tc:xliff:metadata:2.0"
SDL_NS = "http://sdl.com/FileTypes/SdlXliff/1.0"
MQ_NS = "MQXliff"  # memoQ's extension namespace (not a URI, but that is what memoQ writes)
SR_NS = "urn:secondread:xliff:1"
XML_NS = "http://www.w3.org/XML/1998/namespace"
TOOL = "SecondRead"

Format = Literal["xliff12", "xliff20", "sdlxliff"]
CommentMode = Literal["flagged", "all", "none"]

_SR_RISK = f"{{{SR_NS}}}risk"
_SR_LABEL = f"{{{SR_NS}}}label"
_SDL = f"{{{SDL_NS}}}"
_MDA = f"{{{MDA_NS}}}"


class XliffError(ValueError):
    """The input is not an XLIFF file this module can read."""


@dataclass
class Segment:
    index: int  # 1-based position in document order
    key: str  # "<file#>:<unit id>:<segment id>", unique in the document; annotate() takes these
    unit_id: str
    seg_id: str  # mrk@mid, 2.0 segment@id, else the 1-based position inside the unit
    src: str  # source with inline codes as placeholders
    tgt: str | None  # None when there is no target for this segment at all
    src_plain: str  # codes removed, whitespace collapsed: what the model sees
    tgt_plain: str | None
    src_tags: dict[str, str] = field(default_factory=dict)  # placeholder -> original XML
    tgt_tags: dict[str, str] = field(default_factory=dict)
    state: str | None = None  # 1.2 target@state | 2.0 segment@state | SDL conf | memoQ status
    locked: bool = False
    origin: str | None = None  # SDL origin ("mt", "tm", ...) or 1.2 state-qualifier
    match_percent: float | None = None  # SDL TM match percent
    file_original: str | None = None
    risk: float | None = None  # SecondRead risk already stored in the file, if any

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Ref:
    """Where a segment lives in the tree, for write-back."""
    unit: etree._Element  # <trans-unit> (1.2) or <unit> (2.0)
    target: etree._Element | None  # element whose content is this segment's target text
    segmented: bool  # several segments share the unit (1.2 seg-source, 2.0 multi-segment)


def load(source: str | Path | bytes, name: str | None = None) -> "XliffDocument":
    """Parse a file path or raw bytes. `name` (file name) only helps detect .sdlxliff."""
    if isinstance(source, (bytes, bytearray)):
        return XliffDocument(bytes(source), name)
    path = Path(source)
    return XliffDocument(path.read_bytes(), name or path.name)


# ----------------------------------------------------------------------------------------------
# helpers

_PROLOG = re.compile(rb"(\xef\xbb\xbf)?(<\?xml[^>]*\?>[ \t\r\n]*)?")
# Tokens before the root element (comments, PIs, DOCTYPE) and the root start tag itself.
_ROOT_TAG = re.compile(
    rb"<!--.*?-->|<\?.*?\?>|<!DOCTYPE(?:[^\[>]|\[.*?\])*>"
    rb"|(<(?:[\w.-]+:)?xliff)((?:\s+[^\s=>/]+\s*=\s*(?:\"[^\"]*\"|'[^']*'))*)\s*/?>", re.S)
_SPACES = re.compile(r"[ \t\r\n]+")  # XML whitespace only: keep NBSP, Russian typography uses it
_GAP = "\0"  # stands for a standalone code in plain text; XML text can never contain U+0000
_GAP_DROP = re.compile(r"\0+(?=[\s.,;:!?\u2026)\]}\u00bb\u201d'\"]|$)|(?:^|(?<=[\s(\[{\u00ab\u201c]))\0+")


def _plain(s: str) -> str:
    """Model input: a standalone code (line break, image, variable) separates words, so it
    becomes a space unless whitespace or punctuation is already there; then collapse spaces."""
    s = _GAP_DROP.sub("", s).replace(_GAP, " ")
    return _SPACES.sub(" ", s).strip()


def _declare_on_root(data: bytes, wanted: dict[str, str]) -> tuple[bytes, list[bytes]]:
    """Add xmlns:prefix declarations to the root start tag of the raw input.

    lxml cannot add a declaration to an existing element, so a new sr: attribute would get a
    local xmlns:ns0 on every element. etree.cleanup_namespaces() would fix that but is unsafe
    here: it drops xmlns="" undeclarations, which moves Trados' <Comments xmlns=""> into the
    XLIFF namespace. Returns the new bytes and the exact declarations added (to strip on save
    if unused). Non-UTF-8 input is returned unchanged.
    """
    m = next((m for m in _ROOT_TAG.finditer(data) if m.group(1)), None)
    if m is None:
        return data, []
    attrs, added = m.group(2), []
    for prefix, uri in wanted.items():
        if uri.encode() in attrs:
            continue  # already declared (e.g. a file SecondRead annotated before)
        while re.search(rb"xmlns:%s\s*=" % re.escape(prefix.encode()), attrs):
            prefix += "x"
        added.append(b' xmlns:%s="%s"' % (prefix.encode(), uri.encode()))
    return data[:m.end(1)] + b"".join(added) + data[m.end(1):], added


def _parser() -> etree.XMLParser:
    return etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True,
                           remove_blank_text=False, strip_cdata=False, huge_tree=False)


def _local(el) -> str | None:
    return etree.QName(el).localname if isinstance(el.tag, str) else None


def _ns(el) -> str | None:
    return etree.QName(el).namespace if isinstance(el.tag, str) else None


def _ws(s: str | None) -> bool:
    """True for formatting whitespace (contains a newline, nothing else)."""
    return s is not None and "\n" in s and not s.strip()


def _float(s: str | None) -> float | None:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def _translatable(el, stop) -> bool:
    """translate= is inherited from groups (and the file in 2.0); the nearest value wins."""
    while el is not None:
        t = el.get("translate")
        if t is not None:
            return t != "no"
        if el is stop:
            break
        el = el.getparent()
    return True


def _attr_name(el, name: str) -> str:
    if not name.startswith("{"):
        return name
    uri, local = name[1:].split("}", 1)
    if uri == XML_NS:
        return "xml:" + local
    prefix = next((p for p, u in el.nsmap.items() if u == uri and p), None)
    return f"{prefix}:{local}" if prefix else local


def _start_tag(el, empty: bool = False) -> str:
    attrs = "".join(f" {_attr_name(el, k)}={quoteattr(v)}" for k, v in el.attrib.items())
    return f"<{_local(el)}{attrs}{'/' if empty else ''}>"


def _code_xml(el) -> str:
    """Compact namespace-free serialisation of an inline code, for display and mapping."""
    kids = [c for c in el if isinstance(c.tag, str)]
    if not kids and not el.text:
        return _start_tag(el, empty=True)
    inner = escape(el.text or "") + "".join(_code_xml(c) + escape(c.tail or "") for c in kids)
    return f"{_start_tag(el)}{inner}</{_local(el)}>"


class _Codes:
    """Numbers inline codes within one segment. The same code id gets the same number in the
    source and the target, so {2} in the source is the {2} of the target."""

    def __init__(self) -> None:
        self._num: dict[str, int] = {}
        self._anon = 0

    def num(self, code_id: str | None) -> int:
        if code_id is None:  # a code without id (invalid, but seen): never aligned
            self._anon += 1
            code_id = f"\0{self._anon}"
        return self._num.setdefault(code_id, len(self._num) + 1)


def _flatten(el, codes: _Codes, ns: str | None, v2: bool) -> tuple[str, str, dict[str, str]]:
    out: list[str] = []
    plain: list[str] = []
    tags: dict[str, str] = {}
    _walk(el, codes, ns, v2, out, plain, tags)
    return "".join(out), _plain("".join(plain)), tags


def _walk(node, codes: _Codes, ns, v2: bool, out: list, plain: list, tags: dict) -> None:
    def text(s):
        if s:
            out.append(s)
            plain.append(s)

    def code(kind: str, code_id: str | None, xml: str) -> None:
        n = codes.num(code_id)
        ph = f"{{/{n}}}" if kind == "close" else f"{{{n}}}"
        out.append(ph)
        tags.setdefault(ph, xml)
        if kind == "one":
            plain.append(_GAP)

    text(node.text)
    for ch in node:
        if isinstance(ch, etree._Entity):
            text(ch.text)  # unresolved entity: keep "&name;" literally
        elif isinstance(ch.tag, str):  # comments and PIs are skipped, their tail is kept below
            local = _local(ch) if _ns(ch) == ns else None  # foreign elements are transparent
            cid, mtype = ch.get("id"), ch.get("mtype")
            if local in (("pc",) if v2 else ("g",)):  # paired code around translatable text
                code("open", cid, _start_tag(ch))
                _walk(ch, codes, ns, v2, out, plain, tags)
                code("close", cid, f"</{local}>")
            elif local == "ph" or (local == "x" and not v2):
                code("one", cid, _code_xml(ch))  # 1.2 <ph> content is native code, not text
            elif (local == "sc" and v2) or (local in ("bx", "bpt") and not v2):
                code("open", ch.get("rid") or cid, _code_xml(ch))
            elif local == "ec" and v2:
                code("close", ch.get("startRef") or cid, _code_xml(ch))
            elif local in ("ex", "ept") and not v2:
                code("close", ch.get("rid") or cid, _code_xml(ch))
            elif local == "it" and not v2:
                pos = ch.get("pos")
                code("open" if pos == "open" else "close" if pos == "close" else "one", cid,
                     _code_xml(ch))
            elif local == "cp" and v2:  # a character XML cannot carry literally
                try:
                    c = chr(int(ch.get("hex", ""), 16))
                except ValueError:
                    c = ""
                text(c if c != _GAP else "")
            elif local in ("sm", "em") and v2:
                pass  # annotation start/end markers carry no text
            elif local == "mrk" and mtype in ("x-sdl-deleted", "x-sdl-location"):
                pass  # Trados track changes: deleted text is no longer part of the segment
            else:  # mrk (comments, terms, x-sdl-added), sub, unknown elements: keep the text
                _walk(ch, codes, ns, v2, out, plain, tags)
        text(ch.tail)


def _child_indent(parent, step: str | None) -> str | None:
    """Whitespace that precedes `parent`'s children in a pretty-printed file, else None."""
    if len(parent) and _ws(parent.text):
        return parent.text[parent.text.rfind("\n"):]
    own = _own_indent(parent)
    return own + step if own is not None and step else None


def _own_indent(el) -> str | None:
    prev, parent = el.getprevious(), el.getparent()
    s = prev.tail if prev is not None else parent.text if parent is not None else None
    return s[s.rfind("\n"):] if _ws(s) else None


def _insert(parent, index: int, el, step: str | None) -> None:
    """Insert `el` at `index`, mirroring the sibling indentation so pretty files stay pretty."""
    index = max(0, min(index, len(parent)))
    indent = _child_indent(parent, step)
    if indent is not None:
        if index == len(parent):  # new last child takes over the closing indentation
            closing = parent[-1].tail if len(parent) else parent.text
            if not _ws(closing):
                closing = _own_indent(parent) or ""
            if len(parent):
                parent[-1].tail = indent
            else:
                parent.text = indent
            el.tail = closing
        else:
            el.tail = indent
    parent.insert(index, el)
    if indent is not None and step:
        etree.indent(el, space=step, level=sum(1 for _ in el.iterancestors()))


def _remove(el) -> None:
    """Remove `el` without leaving broken indentation or losing mixed-content text."""
    parent, prev, tail = el.getparent(), el.getprevious(), el.tail
    if el.getnext() is None and _ws(tail):
        keep = tail  # last child: its tail is the parent's closing indentation
    elif tail and tail.strip():
        keep = ((prev.tail if prev is not None else parent.text) or "") + tail
    else:
        keep = None
    if keep is not None:
        if prev is not None:
            prev.tail = keep
        else:
            parent.text = keep
    parent.remove(el)


def _unwrap(el) -> None:
    """Replace `el` by its content (text, children, tail) in place."""
    parent, prev = el.getparent(), el.getprevious()
    lead = (el.text or "")
    if prev is not None:
        prev.tail = (prev.tail or "") + lead
    else:
        parent.text = (parent.text or "") + lead
    idx = parent.index(el)
    kids = list(el)
    for i, ch in enumerate(kids):
        parent.insert(idx + i, ch)  # moves ch together with its tail
    last = kids[-1] if kids else el.getprevious()
    if el.tail:
        if last is not None:
            last.tail = (last.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    el.tail = None
    parent.remove(el)


def _note_text(risk: float, flagged: bool, detail: str | None) -> str:
    head = (f"{TOOL}: possible major meaning error (risk {risk:.2f})" if flagged
            else f"{TOOL}: no major meaning error found (risk {risk:.2f})")
    return f"{head}. {detail}" if detail else head


# ----------------------------------------------------------------------------------------------

class XliffDocument:
    """A parsed XLIFF file: `segments` to score, `annotate()` + `save()` to write scores back."""

    def __init__(self, data: bytes, name: str | None = None):
        self.name = name
        m = _PROLOG.match(data)
        self._bom, self._decl = m.group(1) or b"", m.group(2) or b""
        self._crlf = b"\r\n" in data and data.count(b"\n") == data.count(b"\r\n")
        self._final_newline = data.endswith(b"\n")
        patched, self._added_ns = _declare_on_root(data, {"sr": SR_NS, "mda": MDA_NS})
        try:
            self.root = etree.fromstring(patched, _parser())
        except etree.XMLSyntaxError as e:
            raise XliffError(f"not well-formed XML: {e}") from e
        self.format: Format = self._detect()
        self._xns = _ns(self.root)  # None for namespace-less XLIFF 1.2
        t = self.root.text  # indentation step ("  ", "\t", ...) if the file is pretty-printed
        self._step = (t[t.rfind("\n") + 1:] or None) if _ws(t) else None
        self.source_language: str | None = None
        self.target_language: str | None = None
        self.segments: list[Segment] = []
        self._refs: dict[str, _Ref] = {}
        (self._extract20 if self.format == "xliff20" else self._extract12)()

    # -- reading -------------------------------------------------------------------------------

    @property
    def lp(self) -> str | None:
        """Language pair as primary subtags, e.g. "en-ru" for en-US -> ru-RU."""
        s, t = (x and x.replace("_", "-").split("-")[0].lower()
                for x in (self.source_language, self.target_language))
        return f"{s}-{t}" if s and t else None

    def _detect(self) -> Format:
        q = etree.QName(self.root)
        if q.localname != "xliff":
            raise XliffError(f"root element is <{q.localname}>, not <xliff>")
        ns, version = q.namespace, self.root.get("version", "")
        if ns and ns.startswith("urn:oasis:names:tc:xliff:document:2."):
            return "xliff20"
        if (ns and ns.startswith("urn:oasis:names:tc:xliff:document:1.")) or \
                (ns is None and version.startswith("1.")):
            sdl = (SDL_NS in self.root.nsmap.values()
                   or (self.name or "").lower().endswith(".sdlxliff")
                   or next(self.root.iter(f"{_SDL}seg-defs"), None) is not None)
            return "sdlxliff" if sdl else "xliff12"
        raise XliffError(f"unsupported XLIFF namespace/version: {ns!r} {version!r}")

    def _q(self, local: str) -> str:
        return f"{{{self._xns}}}{local}" if self._xns else local

    def _add(self, ref: _Ref, file_idx: int, file_el, unit_id: str, seg_id: str,
             src_el, tgt_el, **meta) -> None:
        v2 = self.format == "xliff20"
        codes = _Codes()  # source first, so target codes reuse the source numbers
        src, src_plain, src_tags = _flatten(src_el, codes, self._xns, v2)
        if tgt_el is not None:
            tgt, tgt_plain, tgt_tags = _flatten(tgt_el, codes, self._xns, v2)
        else:
            tgt, tgt_plain, tgt_tags = None, None, {}
        key = base = f"{file_idx}:{unit_id}:{seg_id}"
        n = 1
        while key in self._refs:  # duplicate ids happen in the wild; keep keys unique
            n += 1
            key = f"{base}#{n}"
        self._refs[key] = ref
        self.segments.append(Segment(
            index=len(self.segments) + 1, key=key, unit_id=unit_id, seg_id=seg_id,
            src=src, tgt=tgt, src_plain=src_plain, tgt_plain=tgt_plain, src_tags=src_tags,
            tgt_tags=tgt_tags, file_original=file_el.get("original"), **meta))

    def _seg_mrks(self, el) -> list:
        return [m for m in el.iter(self._q("mrk")) if m.get("mtype") == "seg"] \
            if el is not None else []

    def _extract12(self) -> None:
        q = self._q
        for fi, f in enumerate(self.root.iter(q("file"))):
            if fi == 0:
                self.source_language = f.get("source-language")
                self.target_language = f.get("target-language")
            for tu in f.iter(q("trans-unit")):
                source = tu.find(q("source"))
                if source is None or not _translatable(tu, f):
                    continue
                uid, target = tu.get("id", ""), tu.find(q("target"))
                mq_locked = tu.get(f"{{{MQ_NS}}}locked", "").lower() in ("locked", "true", "1")
                mq_status = tu.get(f"{{{MQ_NS}}}status")
                src_mrks = self._seg_mrks(tu.find(q("seg-source")))
                if not src_mrks:  # the whole trans-unit is one segment
                    state = target.get("state") if target is not None else None
                    if state is None:
                        state = mq_status or ("approved" if tu.get("approved") == "yes" else None)
                    self._add(_Ref(tu, target, False), fi, f, uid, "1", source, target,
                              state=state, locked=mq_locked, risk=_float(tu.get(_SR_RISK)),
                              origin=target.get("state-qualifier") if target is not None else None)
                    continue
                tgt_mrks = {m.get("mid"): m for m in self._seg_mrks(target)}
                defs = {s.get("id"): s for s in tu.iterfind(f"{_SDL}seg-defs/{_SDL}seg")}
                for pos, m in enumerate(src_mrks, 1):
                    mid = m.get("mid") or str(pos)
                    tm = tgt_mrks.get(mid)
                    if tm is None and len(src_mrks) == 1 and not tgt_mrks and target is not None:
                        tm = target  # a tool segmented the source but not the target
                    d = defs.get(mid)
                    if d is not None:
                        meta = dict(state=d.get("conf"), origin=d.get("origin"),
                                    locked=d.get("locked") == "true",
                                    match_percent=_float(d.get("percent")))
                    else:
                        meta = dict(state=(target.get("state") if target is not None else None)
                                    or mq_status, locked=mq_locked,
                                    origin=target.get("state-qualifier") if target is not None
                                    else None)
                    risk = _float(tm.get(_SR_RISK)) if tm is not None else None
                    self._add(_Ref(tu, tm, True), fi, f, uid, mid, m, tm, risk=risk, **meta)

    def _extract20(self) -> None:
        q = self._q
        self.source_language = self.root.get("srcLang")
        self.target_language = self.root.get("trgLang")
        for fi, f in enumerate(self.root.iter(q("file"))):
            for unit in f.iter(q("unit")):
                if not _translatable(unit, f):
                    continue
                uid = unit.get("id", "")
                stored = {}
                for g in unit.iterfind(f"{_MDA}metadata/{_MDA}metaGroup[@category='{TOOL}']"):
                    meta = {m.get("type"): m.text for m in g.iterfind(f"{_MDA}meta")}
                    stored[meta.get("segment")] = _float(meta.get("risk"))
                segs = unit.findall(q("segment"))
                for pos, s in enumerate(segs, 1):
                    sid = s.get("id") or str(pos)
                    target = s.find(q("target"))
                    self._add(_Ref(unit, target, len(segs) > 1), fi, f, uid, sid,
                              s.find(q("source")), target, state=s.get("state"),
                              risk=stored.get(sid))

    # -- writing -------------------------------------------------------------------------------

    def annotate(self, risks: Mapping[str, float], *, threshold: float = 0.5,
                 details: Mapping[str, str] | None = None, comments: CommentMode = "flagged",
                 store_scores: bool = True, now: datetime | None = None) -> int:
        """Write risk scores (segment key -> P(major) in [0, 1]) into the document.

        comments: "flagged" adds a note/comment for risk >= threshold, "all" for every scored
        segment, "none" for none. store_scores adds the machine-readable sr:/mda: form.
        details: optional extra text per key (e.g. "number 5 vs 15") appended to the comment.
        now: timestamp for SDL comments (pass a fixed value for reproducible output).
        Earlier SecondRead annotations are removed first. Returns the number of comments written.
        """
        unknown = [k for k in risks if k not in self._refs]
        if unknown:
            raise KeyError(f"unknown segment keys: {unknown[:5]}")
        for k, r in risks.items():
            try:
                ok = math.isfinite(float(r)) and 0.0 <= float(r) <= 1.0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"risk for {k!r} must be a number in [0, 1], got {r!r}")
        if comments not in ("flagged", "all", "none"):
            raise ValueError(f"comments must be 'flagged', 'all' or 'none', got {comments!r}")
        self.clear_annotations()
        details = details or {}
        stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="milliseconds")
        written = 0
        for seg in self.segments:
            ref = self._refs[seg.key]
            if seg.key not in risks or ref.target is None or not seg.tgt_plain:
                continue  # empty or missing target: nothing was translated, nothing to flag
            risk = float(risks[seg.key])
            flagged = risk >= threshold
            label = "Major" if flagged else "OK"
            want = comments == "all" or (comments == "flagged" and flagged)
            note = _note_text(risk, flagged, details.get(seg.key)) if want else None
            if self.format == "xliff20":
                written += self._annotate20(seg, ref, risk, label, note, store_scores)
            elif self.format == "sdlxliff" and ref.segmented:
                written += self._annotate_sdl(seg, ref, risk, label, note, store_scores, stamp)
            else:
                written += self._annotate12(seg, ref, risk, label, note, store_scores)
            seg.risk = risk
        return written

    def _annotate12(self, seg, ref, risk, label, note, store) -> int:
        if store:
            el = ref.target if ref.segmented else ref.unit
            el.set(_SR_RISK, f"{risk:.4f}")
            el.set(_SR_LABEL, label)
        if note is None or self.format == "sdlxliff":  # Trados ignores <note>; see _annotate_sdl
            return 0
        n = etree.Element(self._q("note"), {"from": TOOL, "annotates": "target"})
        n.text = f"[segment {seg.seg_id}] {note}" if ref.segmented else note
        # Schema order: source, seg-source, target, (context-group|count-group|prop-group|note|
        # alt-trans)*, then extension elements such as sdl:seg-defs -> after the last XLIFF child.
        idx = max((i + 1 for i, ch in enumerate(ref.unit)
                   if isinstance(ch.tag, str) and _ns(ch) == self._xns), default=0)
        _insert(ref.unit, idx, n, self._step)
        return 1

    def _annotate20(self, seg, ref, risk, label, note, store) -> int:
        unit = ref.unit
        if store:
            md = unit.find(f"{_MDA}metadata")  # at most one per unit; extensions come first
            if md is None:
                md = etree.Element(f"{_MDA}metadata", nsmap={"mda": MDA_NS})
                _insert(unit, 0, md, self._step)
            g = etree.Element(f"{_MDA}metaGroup", {"category": TOOL, "appliesTo": "target"},
                              nsmap={"mda": MDA_NS})
            for t, v in (("segment", seg.seg_id), ("risk", f"{risk:.4f}"), ("label", label)):
                etree.SubElement(g, f"{_MDA}meta", type=t).text = v
            _insert(md, len(md), g, self._step)
        if note is None:
            return 0
        notes = unit.find(self._q("notes"))
        if notes is None:  # <notes> goes after extensions, before originalData and segments
            notes = etree.Element(self._q("notes"))
            idx = next((i for i, ch in enumerate(unit) if _ns(ch) == self._xns and
                        _local(ch) in ("originalData", "segment", "ignorable")), len(unit))
            _insert(unit, idx, notes, self._step)
        n = etree.Element(self._q("note"), {"category": TOOL, "appliesTo": "target"})
        n.text = f"[segment {seg.seg_id}] {note}" if ref.segmented else note
        _insert(notes, len(notes), n, self._step)
        return 1

    def _annotate_sdl(self, seg, ref, risk, label, note, store, stamp) -> int:
        mrk = ref.target  # the target <mrk mtype="seg">
        if store:
            mrk.set(_SR_RISK, f"{risk:.4f}")
            mrk.set(_SR_LABEL, label)
        if note is None:
            return 0
        # A Trados comment = <cmt-def> in <doc-info> + an x-sdl-comment marker around the text.
        cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"secondread:{self.name}:{seg.key}"))
        severity = "Medium" if label == "Major" else "Low"  # Studio: Warning / For information
        # Built from a string because <Comments> must stay in no namespace (xmlns=""), which lxml
        # cannot express for an element created under a default-namespace parent.
        cdef = etree.fromstring(
            f'<cmt-def xmlns="{SDL_NS}" id={quoteattr(cid)}><Comments xmlns="">'
            f'<Comment severity="{severity}" user="{TOOL}" date={quoteattr(stamp)} version="1.0">'
            f'{escape(note)}</Comment></Comments></cmt-def>')
        defs = self._sdl_cmt_defs()
        _insert(defs, len(defs), cdef, self._step)
        wrap = etree.SubElement(mrk, self._q("mrk"), {"mtype": "x-sdl-comment", f"{_SDL}cid": cid})
        wrap.text, mrk.text = mrk.text, None
        for ch in list(mrk)[:-1]:  # everything before the new marker moves inside it
            wrap.append(ch)
        return 1

    def _sdl_cmt_defs(self):
        doc_info = self.root.find(f"{_SDL}doc-info")
        if doc_info is None:  # <doc-info> is the first child of <xliff>
            doc_info = etree.fromstring(f'<doc-info xmlns="{SDL_NS}"/>')
            _insert(self.root, 0, doc_info, self._step)
        defs = doc_info.find(f"{_SDL}cmt-defs")
        if defs is None:
            defs = etree.fromstring(f'<cmt-defs xmlns="{SDL_NS}"/>')
            rev = doc_info.find(f"{_SDL}rev-defs")
            _insert(doc_info, doc_info.index(rev) + 1 if rev is not None else 0, defs, self._step)
        return defs

    def clear_annotations(self) -> None:
        """Remove everything SecondRead wrote earlier; other tools' notes/comments stay."""
        q = self._q
        for el in self.root.iter():
            if isinstance(el.tag, str):
                for a in [a for a in el.attrib if a.startswith(f"{{{SR_NS}}}")]:
                    del el.attrib[a]
        if self.format == "xliff20":
            for n in [n for n in self.root.iter(q("note")) if n.get("category") == TOOL]:
                notes = n.getparent()
                _remove(n)
                if not len(notes):
                    _remove(notes)
            for g in list(self.root.iter(f"{_MDA}metaGroup")):
                if g.get("category") == TOOL:
                    md = g.getparent()
                    _remove(g)
                    if not len(md):
                        _remove(md)
        else:
            for n in [n for n in self.root.iter(q("note")) if n.get("from") == TOOL]:
                _remove(n)
        if self.format == "sdlxliff":
            ours = set()
            for cdef in list(self.root.iter(f"{_SDL}cmt-def")):
                thread = [c for c in cdef.iter() if _local(c) == "Comment"]
                mine = [c for c in thread if c.get("user") == TOOL]
                if mine and len(mine) == len(thread):
                    ours.add(cdef.get("id"))
                    _remove(cdef)
                else:  # someone replied in Trados: keep the thread and its marker, drop ours
                    for c in mine:
                        _remove(c)
            for m in [m for m in self.root.iter(q("mrk"))
                      if m.get("mtype") == "x-sdl-comment" and m.get(f"{_SDL}cid") in ours]:
                _unwrap(m)
            for tag in ("cmt-defs", "doc-info"):
                for el in list(self.root.iter(f"{_SDL}{tag}")):
                    if not len(el):
                        _remove(el)
        for s in self.segments:
            s.risk = None

    # -- serialising ---------------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        tree = self.root.getroottree()
        enc = (tree.docinfo.encoding or "utf-8").lower()
        if enc not in ("utf-8", "utf8") or tree.docinfo.internalDTD is not None:
            return etree.tostring(tree, encoding=tree.docinfo.encoding, xml_declaration=True)
        parts = [tree.docinfo.doctype.encode()] if tree.docinfo.doctype else []
        parts += [etree.tostring(n, encoding="utf-8")
                  for n in reversed(list(self.root.itersiblings(preceding=True)))]
        parts.append(self._root_bytes())
        parts += [etree.tostring(n, encoding="utf-8") for n in self.root.itersiblings()]
        body = b"\n".join(parts) + (b"\n" if self._final_newline else b"")
        if self._crlf:  # literal CRs in content are serialised as &#13;, so this is safe
            body = body.replace(b"\n", b"\r\n")
        return self._bom + self._decl + body

    def _root_bytes(self) -> bytes:
        out = etree.tostring(self.root, encoding="utf-8")
        used = {_ns(el) for el in self.root.iter()} | {
            etree.QName(a).namespace for el in self.root.iter() if isinstance(el.tag, str)
            for a in el.attrib}
        head, rest = out.split(b">", 1)  # lxml escapes ">" in attribute values
        for decl in self._added_ns:  # declarations we injected at load time
            if decl.split(b'"')[1].decode() not in used:
                head = head.replace(decl, b"", 1)
        if SR_NS not in used:  # our own namespace, also when an earlier save declared it
            head = re.sub(rb'\s+xmlns:[\w.-]+="%s"' % re.escape(SR_NS.encode()), b"", head)
        return head + b">" + rest

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())


def _main(argv: list[str]) -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Dump XLIFF / SDLXLIFF segments as JSON lines.")
    ap.add_argument("file")
    args = ap.parse_args(argv)
    doc = load(args.file)
    print(f"# {doc.format} {doc.lp} {len(doc.segments)} segments", file=sys.stderr)
    for s in doc.segments:
        print(json.dumps(s.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    _main(sys.argv[1:])
