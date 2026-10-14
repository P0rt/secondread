import csv, sys, json, collections
csv.field_size_limit(10**9)
c = collections.Counter()
maj = collections.Counter()
anyspan = collections.Counter()
ann = collections.defaultdict(set)
for row in csv.reader(sys.stdin):
    if len(row) < 10:
        continue
    user, system, item, side, sl, tl, score, doc, isbad, spans = row[:10]
    if "tutorial" in system or "tutorial" in doc:
        continue
    lp = sl + "-" + tl
    c[lp] += 1
    ann[lp].add(user)
    try:
        sp = json.loads(spans)
    except Exception:
        sp = []
    if sp:
        anyspan[lp] += 1
    if any(s.get("severity") == "major" for s in sp):
        maj[lp] += 1
for lp, n in sorted(c.items()):
    print("%-10s items=%6d with_span=%6d with_major=%6d (%.1f%%) annotators=%d" % (lp, n, anyspan[lp], maj[lp], 100 * maj[lp] / n, len(ann[lp])))
