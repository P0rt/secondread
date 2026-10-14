"""Subsample synthetic train sets to capped shares per language pair, for scripts/build_sft.py.

usage (repo root):
  python3 -I scripts/mix_synth.py --human data/sft_v1.jsonl \
      --group sense:0.12:data/synth/sense.train.jsonl \
      --group v2:0.08:data/synth/rules.train.jsonl,data/synth/llm.train.jsonl \
      --out-dir data/synth/mix_v3b [--seed 0]
  then: py scripts/build_sft.py --synthetic data/synth/mix_v3b/sense.train.jsonl \
        data/synth/mix_v3b/v2.train.jsonl --out data/sft_v3b.jsonl

Share rule (per pair): with H human rows (counted in --human, a human-only build_sft output,
which is exactly the human part of any build with the same --seed and --es-ok-ratio) and n_g rows
taken from group g, every group satisfies n_g <= cap_g * (H + sum_g n_g). The largest such n_g
are found by a fixed-point iteration from the pool sizes downwards.

Before sampling, every pool row is dropped if (integrity rule 1, docs/REALDOCS.md):
  * its document belongs to a real-document story (data/realdocs/*/aligned.jsonl docs and the
    WMT ids in demo/answer_key.en-es.json, matched per story, i.e. every chunk), or
  * its source or MT contains a word of the integrity list (same regex as scripts/synth_sense.py
    FORBIDDEN_RE and scripts/verify_synth.py), or
  * it is an en-ru neg_insert row whose MT contains "?" (build_sft.py drops these by default with
    --drop-neg-questions en-ru; removing them here keeps the shares exact).
Sampling is stratified by (file, synth_type) with largest-remainder allocation, rows sorted by
id, random.Random(f"{seed}:{group}:{lp}:{file}:{type}"). Nothing is read from dev or test, and no
score or result file is consulted: selection uses counts only.

Writes OUT_DIR/<group>.train.jsonl and OUT_DIR/mix.json (counts, filters, shares).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
from pathlib import Path

# Copied from scripts/synth_sense.py FORBIDDEN_RE (also in scripts/verify_synth.py); keep in sync.
FORBIDDEN_RE = re.compile(
    r"(?<!\w)(?:return\w*|devoluci\w*|flush\w*|lav(?:ar|o|a|as|an|ado|ada|ados|adas|ando|e|en|es|aron|ó|é|"
    r"arse|arlo|arla|arlos|arlas)|paperless|impres[ao]s?|partner\w*|parejas?|take[\s-]*back)(?!\w)", re.IGNORECASE)


def rj(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def story(doc: str) -> str:
    return re.sub(r"_chunk_\d+.*$", "", doc or "")


def real_stories() -> set[str]:
    out = set()
    for f in Path("data/realdocs").glob("*/aligned.jsonl"):
        out |= {story(r["doc"]) for r in rj(f) if r.get("doc")}
    key = Path("demo/answer_key.en-es.json")
    if key.exists():
        out |= {story(m) for m in re.findall(r'"(test-en-[^"\s]+)"', key.read_text(encoding="utf-8"))}
    return out


def allocate(sizes: dict, n: int) -> dict:
    """Largest-remainder split of n over strata proportional to their sizes (ties by key order)."""
    tot = sum(sizes.values())
    if n >= tot:
        return dict(sizes)
    quota = {k: n * v / tot for k, v in sizes.items()}
    alloc = {k: math.floor(q) for k, q in quota.items()}
    rest = n - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", required=True, help="human-only build_sft.py output (e.g. data/sft_v1.jsonl)")
    ap.add_argument("--group", action="append", required=True, metavar="NAME:CAP:FILE[,FILE...]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-question-lps", default="en-ru", help="'none' to keep them")
    a = ap.parse_args()

    human = rj(a.human)
    if any(r.get("source") != "wmt" for r in human):
        sys.exit(f"{a.human} is not human-only")
    H = collections.Counter(r["lp"] for r in human)
    lps = sorted(H)
    stories = real_stories()
    negq = set() if a.neg_question_lps.lower() == "none" else set(a.neg_question_lps.split(","))

    groups, pools, filt = [], {}, {}
    for spec in a.group:
        name, cap, files = spec.split(":", 2)
        groups.append((name, float(cap), files.split(",")))
    if sum(c for _, c, _ in groups) >= 1:
        sys.exit("caps must sum to < 1")
    seen_ids = set()
    for name, _, files in groups:
        pools[name] = collections.defaultdict(list)
        filt[name] = collections.Counter()
        for path in files:
            for r in rj(path):
                if r["id"] in seen_ids:
                    sys.exit(f"duplicate id across pools: {r['id']}")
                seen_ids.add(r["id"])
                why = None
                if story(r.get("doc")) in stories:
                    why = "realdoc_story"
                elif FORBIDDEN_RE.search(r["src"]) or FORBIDDEN_RE.search(r["mt"]):
                    why = "integrity_word"
                elif r.get("synth_type") == "neg_insert" and r["lp"] in negq and "?" in r["mt"]:
                    why = "neg_insert_question"
                if why:
                    filt[name][(r["lp"], why)] += 1
                    continue
                pools[name][r["lp"]].append({**r, "_file": os.path.basename(path)})

    # largest n_g with n_g <= cap_g * (H + sum n) for every group, per pair
    take = {}
    for lp in lps:
        n = {g: len(pools[g][lp]) for g, _, _ in groups}
        for _ in range(1000):
            new = {}
            for g, cap, _ in groups:
                others = sum(v for k, v in n.items() if k != g)
                new[g] = min(len(pools[g][lp]), math.floor(cap * (H[lp] + others) / (1 - cap) + 1e-9))
            if new == n:
                break
            n = new
        T = H[lp] + sum(n.values())
        for g, cap, _ in groups:
            assert n[g] <= cap * T + 1e-9, (lp, g, n, T)
        take[lp] = n

    os.makedirs(a.out_dir, exist_ok=True)
    report = {"human": a.human, "human_rows": dict(H), "seed": a.seed, "groups": {}, "pairs": {}}
    for g, cap, files in groups:
        out_rows = []
        strata_report = {}
        for lp in lps:
            strata = collections.defaultdict(list)
            for r in pools[g][lp]:
                strata[(r["_file"], r.get("synth_type"))].append(r)
            alloc = allocate({k: len(v) for k, v in strata.items()}, take[lp][g])
            for k in sorted(strata):
                rows = sorted(strata[k], key=lambda r: r["id"])
                pick = random.Random(f"{a.seed}:{g}:{lp}:{k[0]}:{k[1]}").sample(rows, alloc[k])
                out_rows += pick
                strata_report[f"{lp} {k[0]} {k[1]}"] = [alloc[k], len(rows)]
        out_rows.sort(key=lambda r: (r["lp"], r["id"]))
        path = os.path.join(a.out_dir, f"{g}.train.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in out_rows:
                r.pop("_file")
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["groups"][g] = {"cap": cap, "files": files, "out": path,
                               "filtered": {f"{lp} {why}": c for (lp, why), c in sorted(filt[g].items())},
                               "pool_after_filter": {lp: len(pools[g][lp]) for lp in lps},
                               "strata_taken_of_pool": strata_report}
    for lp in lps:
        T = H[lp] + sum(take[lp].values())
        report["pairs"][lp] = {"human": H[lp], "total": T,
                               **{g: {"rows": take[lp][g], "share": round(take[lp][g] / T, 4)} for g, _, _ in groups},
                               "synthetic_share": round(sum(take[lp].values()) / T, 4)}
    with open(os.path.join(a.out_dir, "mix.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"filtered": {g: report["groups"][g]["filtered"] for g, _, _ in groups},
                      "pool_after_filter": {g: report["groups"][g]["pool_after_filter"] for g, _, _ in groups},
                      "pairs": report["pairs"]}, indent=1))


if __name__ == "__main__":
    main()
