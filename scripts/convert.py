"""Convert Google/Unbabel MQM TSV (wmt-mqm-human-evaluation) into SecondRead JSONL.

usage: python3 -I convert.py IN.tsv OUT.jsonl LP YEAR
One output line per (system, segment, rater) rating unit:
 {id, lp, year, system, doc, domain, rater, src, mt,
  label: "Major"|"OK", label_any_major: bool,
  spans: [{start,end,text,side,category,family,severity}]}
"""
import csv, sys, json, re, collections

inp, outp, lp, year = sys.argv[1:5]
csv.field_size_limit(10**9)

ACCURACY = {
    # Unbabel flat typology (WMT22 en-ru, WMT24 en-es)
    "mistranslation", "omission", "addition", "untranslated", "mt_hallucination",
    "wrong_named_entity", "wrong_term", "do_not_translate",
    "number_format", "measurement_format", "currency_format", "date_time_format",
}
def family(cat):
    c = cat.strip()
    lc = c.lower()
    if c.startswith("Accuracy/") or lc in ACCURACY or c.startswith("Terminology/"):
        return "meaning"
    if c.startswith("Non-translation"):
        return "meaning"
    if lc.startswith("source") or lc == "source_issue":
        return "source"
    if lc in ("no-error",):
        return "none"
    return "fluency_style"

TAG = re.compile(r"<v>|</v>")

def locate(marked):
    """strip <v>..</v>; return clean text and list of (start,end) char spans in clean text"""
    spans, out, pos, start = [], [], 0, None
    i = 0
    for m in TAG.finditer(marked):
        out.append(marked[i:m.start()])
        pos += m.start() - i
        if m.group() == "<v>":
            start = pos
        else:
            spans.append((start, pos))
            start = None
        i = m.end()
    out.append(marked[i:])
    return "".join(out), spans

units = collections.OrderedDict()
stats = collections.Counter()
with open(inp, encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(r)
    for row in r:
        if len(row) < 9:
            stats["short_row"] += 1
            continue
        system, doc, _, gseg, rater, src_m, tgt_m, cat, sev = row[:9]
        meta = row[9] if len(row) > 9 else "{}"
        if "CANARY" in src_m:
            stats["canary_rows_dropped"] += 1
            continue
        src, sspans = locate(src_m)
        mt, tspans = locate(tgt_m)
        key = (system, gseg, rater)
        u = units.get(key)
        if u is None:
            dom = ""
            try:
                m = meta.strip()
                if m.startswith('"'):
                    m = m[1:-1].replace('""', '"')
                dom = json.loads(m).get("domain", "") if m else ""
            except Exception:
                pass
            if not dom:
                dom = doc.split("-")[1] if doc.startswith("test-") and "-" in doc else ""
                dom = doc.split("_")[0].split("-")[-1] if doc.startswith("test-") else dom
            u = units[key] = dict(id="%s|%s|%s|%s" % (lp, system, gseg, rater), lp=lp, year=int(year),
                                  system=system, doc=doc, domain=dom, rater=rater, src=src, mt=mt, spans=[])
        else:
            if u["mt"] != mt:
                stats["mt_mismatch_within_unit"] += 1
                u["_bad"] = True
            if u["src"] != src:
                stats["src_mismatch_within_unit"] += 1
        s = sev.strip().lower()
        if s == "no-error":
            continue
        side_spans = [("src", a, b) for a, b in sspans] + [("mt", a, b) for a, b in tspans]
        if not side_spans:
            stats["error_without_span"] += 1
            side_spans = [("mt", -1, -1)]
        for side, a, b in side_spans:
            text = (src if side == "src" else mt)[a:b] if a >= 0 else ""
            if a == b and a >= 0:
                stats["empty_span_insertion_point"] += 1
            u["spans"].append(dict(start=a, end=b, text=text, side=side, category=cat.strip(),
                                   family=family(cat), severity=s))

DIG = re.compile(r"\d+(?:[.,]\d+)?")
lab = collections.Counter()
with open(outp, "w", encoding="utf-8") as fo:
    for k in [k for k, u in units.items() if u.get("_bad")]:
        del units[k]
        stats["units_dropped_inconsistent_mt"] += 1
    for u in units.values():
        majors = [sp for sp in u["spans"] if sp["severity"] in ("major", "critical")]
        u["label_any_major"] = bool(majors)
        u["label"] = "Major" if any(sp["family"] == "meaning" for sp in majors) else "OK"
        lab[(u["label"], u["label_any_major"])] += 1
        # digit-set mismatch heuristic (numbers present in src but missing from mt)
        sd = set(x.replace(",", ".") for x in DIG.findall(u["src"]))
        md = set(x.replace(",", ".") for x in DIG.findall(u["mt"]))
        if sd - md:
            stats["units_src_digits_missing_in_mt"] += 1
            if u["label"] == "Major":
                stats["  ...of_which_label_MAJOR"] += 1
        fo.write(json.dumps(u, ensure_ascii=False) + "\n")

n = len(units)
print("units:", n)
for k, v in sorted(lab.items()):
    print("  label=%s any_major=%s: %d (%.1f%%)" % (k[0], k[1], v, 100 * v / n))
print("domains:", collections.Counter(u["domain"] for u in units.values()).most_common())
print("meaning-family major/critical span categories:",
      collections.Counter(sp["category"] for u in units.values() for sp in u["spans"]
                          if sp["severity"] in ("major", "critical") and sp["family"] == "meaning").most_common())
print("stats:", dict(stats))
print("src chars mean/max:", sum(len(u["src"]) for u in units.values()) / n, max(len(u["src"]) for u in units.values()))
