import csv, sys, json, re, collections

path, segcol, globcol = sys.argv[1], sys.argv[2], sys.argv[3]
csv.field_size_limit(10**9)
rows = []
with open(path, encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(r)
    for row in r:
        rows.append(row)
print("header:", header, "ncols in rows:", collections.Counter(len(x) for x in rows))
H = ["system", "doc", segcol, globcol, "rater", "source", "target", "category", "severity", "metadata"]
recs = [dict(zip(H, x)) for x in rows]
print("annotation rows:", len(recs))

V = re.compile(r"<v>(.*?)</v>", re.S)
def strip(s):
    return s.replace("<v>", "").replace("</v>", "")

systems = collections.Counter(x["system"] for x in recs)
raters = collections.Counter(x["rater"] for x in recs)
docs = set(x["doc"] for x in recs)
srcsegs = set(x[globcol] for x in recs)
print("systems (%d):" % len(systems), sorted(systems))
print("raters (%d):" % len(raters), dict(raters))
print("documents:", len(docs))
print("unique source segments (by %s):" % globcol, len(srcsegs))
print("unique source texts:", len(set(strip(x["source"]) for x in recs)))

# segment-rating unit = (system, globalseg, rater)
units = collections.defaultdict(list)
for x in recs:
    units[(x["system"], x[globcol], x["rater"])].append(x)
print("rated (system,segment,rater) units:", len(units))
sysseg = collections.defaultdict(set)
for (s, g, ra) in units:
    sysseg[(s, g)].add(ra)
print("unique (system,segment) translations:", len(sysseg))
print("raters per translation:", collections.Counter(len(v) for v in sysseg.values()))

def sev(x):
    return x["severity"].strip().lower()
def is_majorplus(x):
    return sev(x) in ("major", "critical")

def unit_label(lst):
    return any(is_majorplus(x) for x in lst)

nmaj = sum(unit_label(v) for v in units.values())
nerr = sum(any(sev(x) not in ("no-error",) for x in v) for v in units.values())
noerr = sum(all(sev(x) == "no-error" for x in v) for v in units.values())
print("units with >=1 Major/Critical: %d (%.1f%%)" % (nmaj, 100 * nmaj / len(units)))
print("units with only No-error rows: %d (%.1f%%)" % (noerr, 100 * noerr / len(units)))
nonly_minor = sum((not unit_label(v)) and any(sev(x) in ("minor",) for x in v) for v in units.values())
print("units with minor/neutral but no major: %d" % (len(units) - nmaj - noerr))
# translation-level (any rater major)
tmaj = 0
for (s, g), rs in sysseg.items():
    if any(unit_label(units[(s, g, ra)]) for ra in rs):
        tmaj += 1
print("translations with >=1 Major/Critical by any rater: %d (%.1f%%)" % (tmaj, 100 * tmaj / len(sysseg)))

# accuracy-type major categories
acc_keys = ["omission", "addition", "untranslated", "mistranslation", "hallucination", "number", "named_entity", "accuracy"]
cat_major = collections.Counter()
cat_all = collections.Counter()
for x in recs:
    if sev(x) == "no-error":
        continue
    c = x["category"].strip()
    cat_all[(c, sev(x))] += 1
    if is_majorplus(x):
        cat_major[c] += 1
print("Major+Critical error counts by category:", cat_major.most_common())
tot_err = sum(cat_all.values())
print("total error annotations (excl No-error):", tot_err)
print("severity totals:", collections.Counter(sev(x) for x in recs))

# units with major meaning errors (accuracy family)
def is_meaning(c):
    c = c.lower()
    return any(k in c for k in acc_keys)
nm = sum(any(is_majorplus(x) and is_meaning(x["category"]) for x in v) for v in units.values())
print("units with >=1 Major/Critical ACCURACY-family error (%s): %d (%.1f%%)" % ("/".join(acc_keys), nm, 100 * nm / len(units)))

# where spans are marked
where = collections.Counter()
nspan = collections.Counter()
for x in recs:
    if sev(x) == "no-error":
        continue
    ins = "<v>" in x["source"]
    int_ = "<v>" in x["target"]
    where[(x["category"], "src" if ins else "", "tgt" if int_ else "")] += 1
    nspan[(len(V.findall(x["source"])), len(V.findall(x["target"])))] += 1
print("span location by category (category, in src, in tgt): count")
for k, v in sorted(where.items(), key=lambda kv: (kv[0][0], -kv[1])):
    print("   ", k, v)
print("(#spans in src, #spans in tgt) distribution:", nspan.most_common(10))

# empty span
empt = sum(1 for x in recs if "<v></v>" in x["target"] or "<v></v>" in x["source"])
print("rows with empty <v></v>:", empt)

# number-related: heuristic, digits inside span
numspan = 0
for x in recs:
    if is_majorplus(x):
        sp = V.findall(x["target"]) + V.findall(x["source"])
        if any(re.search(r"\d", s) for s in sp):
            numspan += 1
print("Major/Critical errors whose span contains a digit:", numspan)

# per-unit error count distribution
print("errors per unit:", sorted(collections.Counter(sum(1 for x in v if sev(x) != "no-error") for v in units.values()).items()))

# lengths
print("mean src chars:", sum(len(strip(x["source"])) for x in recs) / len(recs))

# per-system major rate
print("per-system major rate:")
ps = collections.defaultdict(lambda: [0, 0])
for (s, g, ra), v in units.items():
    ps[s][1] += 1
    ps[s][0] += unit_label(v)
for s, (a, b) in sorted(ps.items(), key=lambda kv: kv[1][0] / kv[1][1]):
    print("   %-28s %5d/%5d  %.1f%%" % (s, a, b, 100 * a / b))

if len(sys.argv) > 4:
    # a few examples for inspection
    k = 0
    for x in recs:
        if x["category"].lower() in ("omission", "addition") and is_majorplus(x):
            print(json.dumps({c: x[c] for c in ["category", "severity", "source", "target"]}, ensure_ascii=False)[:700])
            k += 1
            if k >= 4:
                break
