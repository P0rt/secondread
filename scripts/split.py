"""Deterministic document-level train/dev/test split.

usage: python3 -I scripts/split.py IN.jsonl TRAIN DEV TEST
Every source document goes to exactly one split (each source sentence appears
once per MT system, so a segment-level split would leak). The doc->split map is
written to data/splits/<lp>.docs.json so it can be committed and reused before
any synthetic data is generated.
"""
import hashlib, json, os, sys, collections

inp, train, dev, test = sys.argv[1], *map(float, sys.argv[2:5])
assert abs(train + dev + test - 1) < 1e-9
rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
lp = rows[0]["lp"]
docs = sorted({r["doc"] for r in rows})

def bucket(doc):
    h = int(hashlib.sha256(f"secondread|{lp}|{doc}".encode()).hexdigest(), 16) / 2**256
    return "train" if h < train else "dev" if h < train + dev else "test"

doc_split = {d: bucket(d) for d in docs}
os.makedirs("data/splits", exist_ok=True)
json.dump(doc_split, open(f"data/splits/{lp}.docs.json", "w"), indent=0, sort_keys=True)

out = {s: open(f"data/splits/{lp}.{s}.jsonl", "w", encoding="utf-8") for s in ("train", "dev", "test")}
stat = collections.Counter()
for r in rows:
    s = doc_split[r["doc"]]
    out[s].write(json.dumps(r, ensure_ascii=False) + "\n")
    stat[(s, r["label"])] += 1
for s in ("train", "dev", "test"):
    n_docs = sum(v == s for v in doc_split.values())
    maj, ok = stat[(s, "Major")], stat[(s, "OK")]
    print(f"{lp} {s:5s} docs={n_docs:3d} units={maj + ok:5d} Major={maj:4d} ({100 * maj / max(1, maj + ok):.1f}%)")
