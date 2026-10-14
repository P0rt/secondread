import sys, json, collections
segs = collections.Counter()
ann = collections.Counter()
maj = collections.Counter()
ann_ids = collections.defaultdict(set)
systems = collections.defaultdict(set)
keys = collections.Counter()
errkeys = collections.Counter()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    lp = d["doc_id"].split("_#_")[0]
    segs[lp] += 1
    for sysname, lst in d.get("scores", {}).items():
        systems[lp].add(sysname)
        for a in lst or []:
            ann[lp] += 1
            ann_ids[lp].add(a.get("annotator"))
            for k in a:
                keys[k] += 1
            errs = a.get("errors") or []
            for e in errs:
                for k in e:
                    errkeys[(k, e.get("severity") if k == "severity" else "")] += 1
            if any(e.get("severity") == "major" for e in errs):
                maj[lp] += 1
for lp in sorted(segs):
    print("%-14s src_segs=%6d annotations=%7d with_major=%6d (%.1f%%) systems=%d annotators=%d" % (
        lp, segs[lp], ann[lp], maj[lp], 100 * maj[lp] / max(ann[lp], 1), len(systems[lp]), len(ann_ids[lp])))
print("annotation keys:", dict(keys))
print("error keys:", dict(errkeys))
