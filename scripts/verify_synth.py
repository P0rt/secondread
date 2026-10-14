"""Independent split-hygiene check + stats for data/synth/<NAME>.*.jsonl (run from repo root).

usage: python3 -I scripts/verify_synth.py [NAME] [--integrity]
  NAME         llm (default) or sense
  --integrity  also fail on rows from real-document stories (data/realdocs, demo answer key) and on
               rows whose source or MT contains a word of the real-document integrity list (copy of
               scripts/synth_sense.py FORBIDDEN_RE); on by default for NAME=sense
"""
import collections
import json
import re
import sys
from pathlib import Path


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


norm = lambda s: " ".join(s.lower().split())  # noqa: E731
names = [a for a in sys.argv[1:] if not a.startswith("--")]
NAME = names[0] if names else "llm"
INTEGRITY = "--integrity" in sys.argv or NAME == "sense"
TYPE_ORDER = {"llm": ("neg_remove", "neg_insert", "omission", "addition", "ok_paraphrase"),
              "sense": ("wrong_sense", "false_friend", "antonym_term", "synonym_ok")}

# Copied from scripts/synth_sense.py on purpose, so that this check does not import the generator.
FORBIDDEN_RE = re.compile(
    r"(?<!\w)(?:return\w*|devoluci\w*|flush\w*|lav(?:ar|o|a|as|an|ado|ada|ados|adas|ando|e|en|es|aron|ó|é|"
    r"arse|arlo|arla|arlos|arlas)|paperless|impres[ao]s?|partner\w*|parejas?|take[\s-]*back)(?!\w)", re.IGNORECASE)
story = lambda d: re.sub(r"_chunk_\d+.*$", "", d)  # noqa: E731
real_stories = set()
for f in Path("data/realdocs").glob("*/aligned.jsonl"):
    real_stories |= {story(r["doc"]) for r in rj(f) if r.get("doc")}
key = Path("demo/answer_key.en-es.json")
if key.exists():
    real_stories |= {story(m) for m in re.findall(r'"(test-en-[^"\s]+)"', key.read_text(encoding="utf-8"))}

problems = 0
present = {}
for split in ("train", "test"):
    recs = present[split] = rj(f"data/synth/{NAME}.{split}.jsonl")
    rej = rj(f"data/synth/{NAME}.{split}.rejects.jsonl")
    for lp in ("en-ru", "en-es"):
        docs = json.load(open(f"data/splits/{lp}.docs.json"))
        own = {r["id"]: r for r in rj(f"data/splits/{lp}.{split}.jsonl")}
        other_src = set()
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
        extra = ""
        if INTEGRITY:
            bad_real = [r["id"] for r in R if story(r["doc"]) in real_stories]
            bad_word = [r["id"] for r in R if FORBIDDEN_RE.search(r["src"]) or FORBIDDEN_RE.search(r["mt"])]
            bad_span = [r["id"] for r in R for sp in r["spans"] if r["mt"][sp["start"]:sp["end"]] != sp["text"]]
            n += len(bad_real) + len(bad_word) + len(bad_span)
            extra = (f", realdoc-story {len(bad_real)}, integrity-word {len(bad_word)}, "
                     f"span-text-mismatch {len(bad_span)}")
        problems += n
        print(f"{lp} {split}: {len(R)} rows, docs {len({r['doc'] for r in R})}; doc-not-in-split {len(bad_doc)}, "
              f"parent-not-in-split {len(bad_parent)}, parent-not-OK {len(bad_parent_lbl)}, src-in-other-split "
              f"{len(bad_src)}, missing-provenance {len(bad_flag)}, dup-ids {dup_id}{extra}")
        acc = collections.Counter(r["synth_type"] for r in R)
        rj_ = collections.Counter(r["synth_type"] for r in rej if r["lp"] == lp)
        reasons = collections.defaultdict(collections.Counter)
        for r in rej:
            if r["lp"] == lp:
                reasons[r["synth_type"]][r["reason"]] += 1
        for t in TYPE_ORDER.get(NAME, sorted(acc)):
            a, b = acc[t], rj_[t]
            print(f"   {t:14s} accepted {a:4d} / {a + b:4d} ({100 * a / max(1, a + b):5.1f}%)  "
                  f"rejects: {dict(reasons[t].most_common())}")
# cross-split: no source segment shared between train synth and test synth
tr = {(r["lp"], norm(r["src"])) for r in present["train"]}
te = {(r["lp"], norm(r["src"])) for r in present["test"]}
print("train/test synth shared sources:", len(tr & te))
problems += len(tr & te)
print("PROBLEMS:", problems)
sys.exit(1 if problems else 0)
