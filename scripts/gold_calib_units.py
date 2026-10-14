"""Build the blind calibration set for the real-document gold panel.

Source: data/splits/en-es.dev.jsonl (WMT24 en-es, human MQM, one rater per doc/system).
Output (all under data/realdocs/gold/calib/, git-ignored):
  units.jsonl        every calibration unit, blind (annotator-facing fields only)
  batch_01.jsonl     first half  (~120 units), blind
  batch_02.jsonl     second half (~120 units), blind
  key_human.json     uid -> human MQM label, spans, categories, system, sampling weight
                     (NEVER shown to annotators)
  SAMPLING.json      population, strata, targets, seeds tried, draw, weights, batch split

Unit fields are the same as the real-document units:
  {uid, doc_id, seg_index, context_src, src, mt, ref_es}
  - ref_es is "" (WMT24 en-es has no official reference for annotators here)
  - context_src = the previous 2 source segments of the same WMT document (oldest first);
    the source text is identical across systems, so this is also "same document/system"
  - seg_index = 1-based position of the segment in its WMT document
  - uid is positional (cal-a001 .. cal-b120) and carries no system or label information

Sampling (seeded, recorded in SAMPLING.json):
  - A "chunk" is a contiguous run of <= CHUNK segments of one WMT document; a "stream" is a
    chunk as translated by one system. Streams are sampled whole, so annotators always see
    documents, never isolated error segments.
  - Strata: A1 = MSLC streams with >= 1 human Major, A2 = other systems' streams with >= 1
    Major, B1 = Major-free multi-segment streams (news/social), B2 = Major-free speech
    singletons. The natural Major rate (5.8%) would give ~14 Majors in 240 units; the panel
    needs >= 40, so A is sampled at f_A = TARGET_MAJORS / all Majors (same fraction in A1 and
    A2, which keeps the natural system mix of Majors), and B fills the rest of the budget in
    proportion to its population.
  - At most MAX_VERSIONS systems per source chunk (the real-document set has 2 MT versions
    per segment; more versions would turn annotation into side-by-side comparison).
  - Restricted randomisation: seeds 0, 1, 2, ... are tried in order and the first draw that
    meets every acceptance rule is kept (40-46 Majors, 232-248 units, each stratum within 6
    units of its target, A1/A2 Majors within 3 of f_A x their population Majors); rejected
    seeds are recorded.
  - Post-stratification weight per unit w_h = population units / sampled units of its stratum,
    so calibration metrics can also be reported at the natural Major rate.

usage: PYTHONPATH=. uv run --frozen python scripts/gold_calib_units.py
"""
import collections
import hashlib
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "data/splits/en-es.dev.jsonl")
OUT = os.path.join(ROOT, "data/realdocs/gold/calib")

CHUNK = 8              # max segments per document chunk
MAX_VERSIONS = 3       # max systems sampled per source chunk
TARGET_UNITS = 240
TARGET_MAJORS = 43     # midpoint of the accepted range below
ACCEPT_MAJORS = (40, 46)
ACCEPT_UNITS = (232, 248)
STRATUM_TOL = 6        # accepted |sampled units - target| per stratum
MAJOR_TOL = 3          # accepted |sampled Majors - f_A * population Majors| in A1 and A2
DRAW_SLACK = 4         # a stream may overshoot its stratum target by this many units
MAX_SEEDS = 1000
SPLIT_SEED = 0
SPLIT_ITERS = 20000
UNIT_FIELDS = ("uid", "doc_id", "seg_index", "context_src", "src", "mt", "ref_es")
SEV_RANK = {"none": 0, "neutral": 1, "minor": 2, "major": 3, "critical": 4}


def seg_no(r):
    return int(r["id"].split("|")[2])


def chunk_key(s):
    return f"{s['chunk'][0]}#{s['chunk'][1]}"


def load():
    rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
    doc_order, doc_segs = [], collections.defaultdict(set)
    for r in rows:
        if r["doc"] not in doc_segs:
            doc_order.append(r["doc"])
        doc_segs[r["doc"]].add(seg_no(r))
    # source text must be identical across systems for a WMT segment number
    src_of = {}
    for r in rows:
        assert src_of.setdefault(seg_no(r), r["src"]) == r["src"], r["id"]
    pos, chunk_of = {}, {}
    for d in doc_order:
        ix = sorted(doc_segs[d])
        assert ix == list(range(ix[0], ix[0] + len(ix))), f"gap in {d}"
        k = math.ceil(len(ix) / CHUNK)
        size = math.ceil(len(ix) / k)
        for i, g in enumerate(ix):
            pos[g] = (d, i + 1)
            chunk_of[g] = (d, i // size)
    return rows, doc_order, doc_segs, src_of, pos, chunk_of


def build_streams(rows, chunk_of):
    streams = collections.defaultdict(list)
    for r in rows:
        streams[(chunk_of[seg_no(r)], r["system"])].append(r)
    out = []
    for (ch, system), rs in streams.items():
        rs.sort(key=seg_no)
        majors = sum(r["label"] == "Major" for r in rs)
        domain = rs[0]["domain"]
        if majors:
            stratum = "A1" if system == "MSLC" else "A2"
        else:
            stratum = "B2" if domain == "speech" else "B1"
        out.append(dict(key=f"{ch[0]}#{ch[1]}|{system}", chunk=ch, system=system,
                        rows=rs, n=len(rs), majors=majors, stratum=stratum, domain=domain))
    out.sort(key=lambda s: s["key"])
    return out


def targets(streams):
    pop = collections.Counter()
    for s in streams:
        pop[s["stratum"]] += s["n"]
    all_majors = sum(s["majors"] for s in streams)
    f_a = TARGET_MAJORS / all_majors
    t = {h: round(f_a * pop[h]) for h in ("A1", "A2")}
    t_b = TARGET_UNITS - t["A1"] - t["A2"]
    pop_b = pop["B1"] + pop["B2"]
    t["B1"] = round(t_b * pop["B1"] / pop_b)
    t["B2"] = t_b - t["B1"]
    return pop, all_majors, f_a, t


def draw(streams, t, f_a, pop_majors, seed):
    rng = random.Random(seed)
    versions = collections.Counter()
    taken, got = [], collections.Counter()
    for h in ("A1", "A2", "B1", "B2"):          # scarce strata first
        cand = [s for s in streams if s["stratum"] == h]
        rng.shuffle(cand)
        for s in cand:
            if got[h] >= t[h]:
                break
            if got[h] + s["n"] > t[h] + DRAW_SLACK:
                continue
            if versions[s["chunk"]] >= MAX_VERSIONS:
                continue
            taken.append(s)
            got[h] += s["n"]
            versions[s["chunk"]] += 1
    majors = sum(s["majors"] for s in taken)
    units = sum(got.values())
    why = []
    if not ACCEPT_MAJORS[0] <= majors <= ACCEPT_MAJORS[1]:
        why.append(f"majors={majors}")
    if not ACCEPT_UNITS[0] <= units <= ACCEPT_UNITS[1]:
        why.append(f"units={units}")
    for h in t:
        if abs(got[h] - t[h]) > STRATUM_TOL:
            why.append(f"{h}={got[h]} vs {t[h]}")
    for h in ("A1", "A2"):                       # keep the natural system mix of Majors
        m_h = sum(s["majors"] for s in taken if s["stratum"] == h)
        if abs(m_h - f_a * pop_majors[h]) > MAJOR_TOL:
            why.append(f"{h} majors={m_h} vs {f_a * pop_majors[h]:.1f}")
    return taken, got, majors, units, why


def split_batches(groups):
    """2 bins, groups (source chunks) kept whole; balance units, Majors and B units."""
    rng = random.Random(SPLIT_SEED)
    keys = sorted(groups)

    def cost(assign):
        u, m, b = [0, 0], [0, 0], [0, 0]
        for k, a in zip(keys, assign):
            g = groups[k]
            u[a] += g["n"]
            m[a] += g["majors"]
            b[a] += g["b_units"]
        return (abs(u[0] - u[1]) + 3 * abs(m[0] - m[1]) + 0.5 * abs(b[0] - b[1]), u, m, b)

    best = None
    for _ in range(SPLIT_ITERS):
        assign = [rng.randrange(2) for _ in keys]
        c = cost(assign)
        if best is None or c[0] < best[0][0]:
            best = (c, assign)
    (c, u, m, b), assign = best
    return {k: a for k, a in zip(keys, assign)}, dict(cost=c, units=u, majors=m, b_units=b)


def main():
    rows, doc_order, doc_segs, src_of, pos, chunk_of = load()
    streams = build_streams(rows, chunk_of)
    pop, all_majors, f_a, t = targets(streams)
    pop_majors = collections.Counter()
    for s in streams:
        pop_majors[s["stratum"]] += s["majors"]

    tried = []
    for seed in range(MAX_SEEDS):
        taken, got, majors, units, why = draw(streams, t, f_a, pop_majors, seed)
        if not why:
            break
        tried.append(dict(seed=seed, rejected=why))
    else:
        raise SystemExit("no seed met the acceptance rules")

    # weights: population units / sampled units, per stratum
    weight = {h: pop[h] / got[h] for h in got}

    # group sampled streams by source chunk (all sampled versions of a chunk together)
    groups = collections.defaultdict(lambda: dict(streams=[], n=0, majors=0, b_units=0))
    for s in taken:
        g = groups[chunk_key(s)]
        g["streams"].append(s)
        g["n"] += s["n"]
        g["majors"] += s["majors"]
        g["b_units"] += s["n"] if s["stratum"].startswith("B") else 0
    assign, split_stats = split_batches(groups)

    doc_rank = {d: i for i, d in enumerate(doc_order)}
    batches = {0: [], 1: []}
    key_human = {}
    for gk in sorted(groups, key=lambda k: (doc_rank[k.split("#")[0]], int(k.split("#")[1]))):
        g = groups[gk]
        rs = [(r, s) for s in g["streams"] for r in s["rows"]]
        random.Random(f"0|{gk}").shuffle(rs)       # shuffle within the document chunk (seed 0)
        batches[assign[gk]].extend((r, s, gk) for r, s in rs)

    units_all, batch_files = [], []
    for b in (0, 1):
        letter = "ab"[b]
        out_units = []
        for i, (r, s, gk) in enumerate(batches[b], 1):
            g = seg_no(r)
            doc, idx = pos[g]
            first = min(doc_segs[doc])
            ctx = [src_of[x] for x in range(max(first, g - 2), g)]
            uid = f"cal-{letter}{i:03d}"
            u = dict(uid=uid, doc_id=f"wmt24-en-es/{doc}", seg_index=idx, context_src=ctx,
                     src=r["src"], mt=r["mt"], ref_es="")
            out_units.append(u)
            spans = [dict(text=sp["text"], start=sp["start"], end=sp["end"], side=sp["side"],
                          category=sp["category"], family=sp["family"], severity=sp["severity"])
                     for sp in r["spans"]]
            maxsev = max((sp["severity"] for sp in spans), key=SEV_RANK.get, default="none")
            mean_sev = max((sp["severity"] for sp in spans if sp["family"] == "meaning"),
                           key=SEV_RANK.get, default="none")
            key_human[uid] = dict(
                label=r["label"], label_any_major=r["label_any_major"],
                max_severity=maxsev, meaning_max_severity=mean_sev,
                categories=sorted({sp["category"] for sp in spans}),
                major_categories=sorted({sp["category"] for sp in spans
                                         if sp["family"] == "meaning"
                                         and sp["severity"] in ("major", "critical")}),
                spans=spans, wmt_id=r["id"], system=r["system"], rater=r["rater"],
                doc=r["doc"], domain=r["domain"], chunk=gk, stratum=s["stratum"],
                weight=round(weight[s["stratum"]], 6))
        batch_files.append((f"batch_{b + 1:02d}.jsonl", out_units))
        units_all.extend(out_units)

    # blindness checks: annotator files carry only unit fields, no system names, no labels
    systems = sorted({r["system"] for r in rows})
    for u in units_all:
        assert tuple(u) == UNIT_FIELDS
        for f in ("uid", "doc_id"):
            assert not any(sys_.lower() in u[f].lower() for sys_ in systems), u[f]
    assert len({u["uid"] for u in units_all}) == len(units_all) == len(key_human)

    os.makedirs(OUT, exist_ok=True)

    def dump_jsonl(name, us):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for u in us:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")

    dump_jsonl("units.jsonl", units_all)
    for name, us in batch_files:
        dump_jsonl(name, us)
    with open(os.path.join(OUT, "key_human.json"), "w", encoding="utf-8") as f:
        json.dump(key_human, f, ensure_ascii=False, indent=1)

    # sampling record
    n_major = sum(v["label"] == "Major" for v in key_human.values())
    w_major = sum(v["weight"] for v in key_human.values() if v["label"] == "Major")
    w_all = sum(v["weight"] for v in key_human.values())
    per_batch = []
    for name, us in batch_files:
        ks = [key_human[u["uid"]] for u in us]
        per_batch.append(dict(file=name, units=len(us),
                              majors=sum(k["label"] == "Major" for k in ks),
                              chunks=len({k["chunk"] for k in ks}),
                              by_stratum=dict(collections.Counter(k["stratum"] for k in ks)),
                              by_domain=dict(collections.Counter(k["domain"] for k in ks))))
    dup_pairs = collections.Counter((u["src"], u["mt"]) for u in units_all)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    record = dict(
        source=dict(file="data/splits/en-es.dev.jsonl", sha256=sha(DEV), rows=len(rows),
                    majors=all_majors, natural_major_rate=round(all_majors / len(rows), 4),
                    docs=len(doc_order), systems=systems,
                    note="one MQM rater per doc/system; human-human agreement cannot be "
                         "measured on this split. 'Major' = label field (any major/critical "
                         "span in the meaning family, scripts/convert.py)."),
        design=dict(chunk_max_segments=CHUNK, max_versions_per_source_chunk=MAX_VERSIONS,
                    target_units=TARGET_UNITS, target_majors=TARGET_MAJORS,
                    accept_majors=ACCEPT_MAJORS, accept_units=ACCEPT_UNITS,
                    stratum_tolerance=STRATUM_TOL, draw_slack=DRAW_SLACK,
                    stratum_major_tolerance=MAJOR_TOL,
                    strata={"A1": "MSLC streams with >=1 human Major",
                            "A2": "non-MSLC streams with >=1 human Major",
                            "B1": "Major-free multi-segment streams (news, social)",
                            "B2": "Major-free speech singletons"},
                    f_A=round(f_a, 4),
                    why="natural rate gives ~14 Majors in 240 units; >=40 needed, so "
                        "Major-bearing streams are oversampled at one common fraction f_A "
                        "(keeps the natural system mix of Majors) and Major-free streams "
                        "fill the rest in proportion to their population. Streams are "
                        "whole document chunks; the within-chunk Major rate is natural."),
        population_units=dict(pop), population_majors=dict(pop_majors),
        stratum_targets=t, sampled_units=dict(got),
        sampled_majors=dict(collections.Counter(
            s["stratum"] for s in taken for r in s["rows"] if r["label"] == "Major")),
        weights={h: round(w, 6) for h, w in weight.items()},
        seed=dict(accepted=seed, rejected=tried,
                  rule="seeds tried in order 0,1,2,...; first draw meeting all acceptance "
                       "rules kept"),
        result=dict(units=len(units_all), majors=n_major,
                    major_rate=round(n_major / len(units_all), 4),
                    major_rate_reweighted=round(w_major / w_all, 4),
                    majors_by_system=dict(collections.Counter(
                        v["system"] for v in key_human.values() if v["label"] == "Major")),
                    units_by_system=dict(collections.Counter(
                        v["system"] for v in key_human.values())),
                    units_by_domain=dict(collections.Counter(
                        v["domain"] for v in key_human.values())),
                    ok_with_nonmeaning_major=sum(v["label"] == "OK" and v["label_any_major"]
                                                 for v in key_human.values()),
                    identical_src_mt_pairs=sum(c - 1 for c in dup_pairs.values() if c > 1),
                    distinct_sources=len({u["src"] for u in units_all}),
                    source_chunks=len(groups)),
        streams=[dict(stream=s["key"], stratum=s["stratum"], units=s["n"], majors=s["majors"],
                      batch="batch_%02d" % (assign[chunk_key(s)] + 1))
                 for s in sorted(taken, key=lambda s: s["key"])],
        batch_split=dict(seed=SPLIT_SEED, iterations=SPLIT_ITERS,
                         objective="|units diff| + 3*|Majors diff| + 0.5*|B units diff|, "
                                   "source chunks kept whole",
                         stats=split_stats, batches=per_batch),
        order="batches list source chunks in document order; units are shuffled within "
              "each source chunk with random.Random('0|<doc>#<chunk>')",
        blind="batch_*.jsonl and units.jsonl hold only unit fields; uid is positional; "
              "system, rater and labels are only in key_human.json",
        script="scripts/gold_calib_units.py",
        outputs={name: sha(os.path.join(OUT, name))
                 for name in ["units.jsonl"] + [n for n, _ in batch_files] + ["key_human.json"]},
    )
    with open(os.path.join(OUT, "SAMPLING.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)

    print(json.dumps(dict(seed=seed, rejected=len(tried), result=record["result"],
                          sampled=dict(got), targets=t, batches=per_batch), indent=1))


if __name__ == "__main__":
    main()
