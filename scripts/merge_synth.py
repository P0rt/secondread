"""Merge per-pair synth_llm.py outputs into data/synth/llm.train.jsonl (+ rejects). Run from repo root.
usage: merge_synth.py OUT_PREFIX PART_DIR [PART_DIR ...]   e.g. data/synth/llm.train data/synth/llm_parts/en-ru ..."""
import json, sys, collections
out, parts = sys.argv[1], sys.argv[2:]
SPLIT = out.rsplit(".", 1)[-1]
acc, rej = {}, {}
for d in parts:
    for l in open(f"{d}/llm.{SPLIT}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in acc and acc[r["id"]] != r:
            sys.exit(f"conflicting duplicate id {r['id']}")
        acc[r["id"]] = r
    for l in open(f"{d}/llm.{SPLIT}.rejects.jsonl", encoding="utf-8"):
        r = json.loads(l)
        rej[(r["parent"], r["synth_type"])] = r
done = {(r["synth_parent"], r["synth_type"]) for r in acc.values()}
rej = {k: v for k, v in rej.items() if k not in done}
rows = sorted(acc.values(), key=lambda r: r["id"])
with open(f"{out}.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{out}.rejects.jsonl", "w", encoding="utf-8") as f:
    for r in rej.values(): f.write(json.dumps(r, ensure_ascii=False) + "\n")
seg = collections.Counter((r["lp"], r["doc"], r["src"]) for r in rows)
print(f"wrote {out}.jsonl ({len(rows)}) and rejects ({len(rej)}); max edits per source segment {max(seg.values())}, "
      f"segments with >4 edits {sum(v > 4 for v in seg.values())}")
print(dict(sorted(collections.Counter((r['lp'], r['synth_type']) for r in rows).items())))
