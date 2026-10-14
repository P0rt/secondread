"""Independent split-hygiene check + stats for data/synth/llm.*.jsonl (run from repo root)."""
import json, sys, collections
def rj(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
norm = lambda s: " ".join(s.lower().split())
problems = 0
for split in ("train", "test"):
    recs = rj(f"data/synth/llm.{split}.jsonl")
    rej = rj(f"data/synth/llm.{split}.rejects.jsonl")
    for lp in ("en-ru", "en-es"):
        docs = json.load(open(f"data/splits/{lp}.docs.json"))
        own = {r["id"]: r for r in rj(f"data/splits/{lp}.{split}.jsonl")}
        other_src = set()
        other_seg = set()
        for s in ("train", "dev", "test"):
            if s != split:
                for r in rj(f"data/splits/{lp}.{s}.jsonl"):
                    other_src.add(norm(r["src"]))
        R = [r for r in recs if r["lp"] == lp]
        bad_doc = [r["id"] for r in R if docs.get(r["doc"]) != split]
        bad_parent = [r["id"] for r in R if r["synth_parent"] not in own]
        bad_parent_lbl = [r["id"] for r in R if r["synth_parent"] in own and own[r["synth_parent"]]["label"] != "OK"]
        bad_src = [r["id"] for r in R if norm(r["src"]) in other_src]
        bad_flag = [r["id"] for r in R if not r.get("synthetic") or r.get("synth_model") is None]
        dup_id = len(R) - len({r["id"] for r in R})
        n = len(bad_doc) + len(bad_parent) + len(bad_parent_lbl) + len(bad_src) + len(bad_flag) + dup_id
        problems += n
        print(f"{lp} {split}: {len(R)} rows, docs {len({r['doc'] for r in R})}; doc-not-in-split {len(bad_doc)}, "
              f"parent-not-in-split {len(bad_parent)}, parent-not-OK {len(bad_parent_lbl)}, src-in-other-split {len(bad_src)}, "
              f"missing-provenance {len(bad_flag)}, dup-ids {dup_id}")
        acc = collections.Counter(r["synth_type"] for r in R)
        rj_ = collections.Counter(r["synth_type"] for r in rej if r["lp"] == lp)
        reasons = collections.defaultdict(collections.Counter)
        for r in rej:
            if r["lp"] == lp: reasons[r["synth_type"]][r["reason"]] += 1
        for t in ("neg_remove", "neg_insert", "omission", "addition", "ok_paraphrase"):
            a, b = acc[t], rj_[t]
            print(f"   {t:14s} accepted {a:4d} / {a+b:4d} ({100*a/max(1,a+b):5.1f}%)  rejects: {dict(reasons[t].most_common())}")
# cross-split: no source segment shared between train synth and test synth
tr = {(r["lp"], norm(r["src"])) for r in rj("data/synth/llm.train.jsonl")}
te = {(r["lp"], norm(r["src"])) for r in rj("data/synth/llm.test.jsonl")}
print("train/test synth shared sources:", len(tr & te)); problems += len(tr & te)
print("PROBLEMS:", problems)
sys.exit(1 if problems else 0)
