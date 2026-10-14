"""Seeded subsample of a synthetic train file under a per-pair share cap (the v3a recipe).

usage: python3 -I scripts/subsample_synth.py data/synth/sense.train.jsonl \
           --out data/synth/sense.v3a.train.jsonl [--max-share 0.12] [--major-per-ok 3] [--seed 0]

Rule, fixed before training (v3a):
  * H = human rows that scripts/build_sft.py takes for the pair with its defaults: every en-ru
    train unit; for en-es every Major train unit plus round(--es-ok-ratio x Major) OK units.
  * S_max = the largest S with S / (H + S) <= --max-share.
  * Major:OK about --major-per-ok : 1 inside the synthetic part. If the Major pool is the limit,
    all Major rows are taken and round(Major / ratio) OK rows; otherwise round(S_max / (ratio + 1))
    OK rows and S_max minus that many Major rows.
  * Inside each label the count is split over synth_type in proportion to the pool (largest
    remainder), and rows are drawn per type with random.Random(f"{seed}:{lp}:{type}") from the
    id-sorted pool, so the result does not depend on the input order.
The output keeps the input's rows unchanged and in input order.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random


def rj(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def human_rows(splits_dir: str, lp: str, es_ok_ratio: float) -> int:
    rows = rj(os.path.join(splits_dir, f"{lp}.train.jsonl"))
    if lp != "en-es":
        return len(rows)
    n_major = sum(r["label"] == "Major" for r in rows)
    return n_major + min(len(rows) - n_major, round(es_ok_ratio * n_major))  # = build_sft.select_wmt


def split_by_type(n: int, pool: dict[str, list[dict]]) -> dict[str, int]:
    total = sum(len(v) for v in pool.values())
    quota = {t: n * len(v) / total for t, v in pool.items()}
    out = {t: math.floor(q) for t, q in quota.items()}
    for t in sorted(pool, key=lambda t: (-(quota[t] - out[t]), t))[: n - sum(out.values())]:
        out[t] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("synthetic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--max-share", type=float, default=0.12)
    ap.add_argument("--major-per-ok", type=float, default=3.0)
    ap.add_argument("--es-ok-ratio", type=float, default=3.0, help="must match build_sft.py --es-ok-ratio")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = rj(a.synthetic)
    keep: set[str] = set()
    for lp in sorted({r["lp"] for r in rows}):
        h = human_rows(a.splits_dir, lp, a.es_ok_ratio)
        s_max = math.floor(a.max_share * h / (1 - a.max_share) + 1e-9)
        pools = {lab: collections.defaultdict(list) for lab in ("Major", "OK")}
        for r in sorted((r for r in rows if r["lp"] == lp), key=lambda r: r["id"]):
            pools[r["label"]][r["synth_type"]].append(r)
        n_maj_pool = sum(len(v) for v in pools["Major"].values())
        n_ok_pool = sum(len(v) for v in pools["OK"].values())
        if n_maj_pool + round(n_maj_pool / a.major_per_ok) <= s_max:
            n_major, n_ok = n_maj_pool, round(n_maj_pool / a.major_per_ok)
        else:
            n_ok = round(s_max / (a.major_per_ok + 1))
            n_major = s_max - n_ok
        n_ok = min(n_ok, n_ok_pool)
        assert n_major + n_ok <= s_max and (n_major + n_ok) / (h + n_major + n_ok) <= a.max_share
        for lab, n in (("Major", n_major), ("OK", n_ok)):
            for t, k in sorted(split_by_type(n, pools[lab]).items()):
                pick = random.Random(f"{a.seed}:{lp}:{t}").sample(pools[lab][t], k)
                keep |= {r["id"] for r in pick}
                print(f"{lp} {lab:5s} {t:14s} {k:5d} of {len(pools[lab][t]):5d}")
        print(f"{lp}: human {h}, synthetic {n_major + n_ok} (Major {n_major}, OK {n_ok}, "
              f"{n_major / max(n_ok, 1):.2f}:1), share {(n_major + n_ok) / (h + n_major + n_ok):.4f} "
              f"(cap {a.max_share}, S_max {s_max})")

    out = [r for r in rows if r["id"] in keep]
    assert len(out) == len(keep), "duplicate ids in the input"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {len(out)} rows")


if __name__ == "__main__":
    main()
