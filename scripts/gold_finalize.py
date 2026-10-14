"""Finalise the real-document gold labels (protocol v2, section 4, steps 3 to 5) and compute the gold statistics.

usage:
  uv run --frozen python -I scripts/gold_finalize.py

Inputs (data/realdocs/gold/, git-ignored):
  annotations_raw.jsonl     three lens records per uid, the vote and the queue reasons (scripts/gold_merge.py)
  to_adjudicate.jsonl       the adjudication queue (scripts/gold_merge.py)
  adjudication.jsonl        the senior adjudicator's decisions, one per queued uid, written blind BEFORE this
                            script joins the system key (its sha256 is recorded in gold_stats.json)
  adjudicator_audit.jsonl   optional, non-binding post-hoc second opinions; summarised, never applied
  key_systems.json          uid -> MT system and segment_id; read only here, after adjudication
  ../adjudication_final.jsonl, ../adjudication_fill.jsonl
                            the earlier real-document labels; only id, system, doc_id, verdict, mt and explanation
                            are read, for the agreement table (never the SecondRead or Gemma fields)

Outputs (data/realdocs/gold/):
  labels.jsonl      one line per unit: {uid, doc_id, seg_index, system, label, severity, category, span_mt, reason,
                    votes, adjudicated} plus segment_id, n_major, adjudication_reason, label_preregistered
  gold_stats.json   every number in docs/REALDOCS_GOLD.md

Final label (section 4.4): the adjudicated label where one exists, else the majority vote (Major when >= 2 of 3
lenses say major|critical). For a unit that is not adjudicated, severity is the median of the three lens severities
on none < minor < major < critical (so it is major or critical exactly when the majority says Major), and
category/span_mt come from the first lens, in the order mqm, posteditor, reader, whose severity equals that median.
label_preregistered uses the majority vote on minority_major_low_conf units (amendment A1) and the primary label
elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "data" / "realdocs" / "gold"
OLD = [REPO / "data" / "realdocs" / "adjudication_final.jsonl", REPO / "data" / "realdocs" / "adjudication_fill.jsonl"]
LENSES = ["mqm", "posteditor", "reader"]
SEV_ORDER = ["none", "minor", "major", "critical"]
MAJOR = {"major", "critical"}
N_BOOT, SEED = 10000, 0


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def fleiss_kappa(table: list[list[int]]) -> float | None:
    if not table:
        return None
    n, N, k = sum(table[0]), len(table), len(table[0])
    p_j = [sum(row[j] for row in table) / (N * n) for j in range(k)]
    P_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in table]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    return None if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def cohen_kappa(a: list, b: list) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(a) | set(b)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def cls3(sev: str) -> str:
    return "major+" if sev in MAJOR else sev


def doc_bootstrap(rows: list[dict], stat, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile 95% CI of stat(rows) when whole documents are resampled with replacement."""
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    docs = sorted(by_doc)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for d in (rng.choice(docs) for _ in docs) for r in by_doc[d]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    point = stat(rows)
    if not vals:
        return {"point": point, "ci95": None, "n_boot_used": 0}
    lo, hi = vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]
    return {"point": point, "ci95": [lo, hi], "n_boot_used": len(vals), "n_clusters": len(docs)}


def lens_agreement(rows: list[dict]) -> dict:
    sev = {l: [r["votes"][l] for r in rows] for l in LENSES}
    bin_ = {l: [s in MAJOR for s in sev[l]] for l in LENSES}
    c3 = ["none", "minor", "major+"]

    def fk_bin(rs):
        return fleiss_kappa([[sum(r["votes"][l] not in MAJOR for l in LENSES),
                              sum(r["votes"][l] in MAJOR for l in LENSES)] for r in rs])

    def fk_3(rs):
        return fleiss_kappa([[sum(cls3(r["votes"][l]) == c for l in LENSES) for c in c3] for r in rs])

    return {
        "n_units": len(rows),
        "fleiss_kappa_binary": doc_bootstrap(rows, fk_bin),
        "fleiss_kappa_3class": doc_bootstrap(rows, fk_3),
        "cohen_kappa_binary": {f"{a}-{b}": cohen_kappa(bin_[a], bin_[b]) for a, b in combinations(LENSES, 2)},
        "cohen_kappa_3class": {f"{a}-{b}": cohen_kappa([cls3(s) for s in sev[a]], [cls3(s) for s in sev[b]])
                               for a, b in combinations(LENSES, 2)},
        "unanimous_binary_share": sum(len({r["votes"][l] in MAJOR for l in LENSES}) == 1 for r in rows) / len(rows),
        "unanimous_3class_share": sum(len({cls3(r["votes"][l]) for l in LENSES}) == 1 for r in rows) / len(rows),
        "major_votes_per_lens": {l: sum(bin_[l]) for l in LENSES},
        "severity_counts_per_lens": {l: {s: sev[l].count(s) for s in SEV_ORDER} for l in LENSES},
    }


def main() -> None:
    raw = _jsonl(GOLD / "annotations_raw.jsonl")
    queue = {q["uid"]: q for q in _jsonl(GOLD / "to_adjudicate.jsonl")}
    adj_path = GOLD / "adjudication.jsonl"
    adj = {a["uid"]: a for a in _jsonl(adj_path)}
    units = {u["uid"]: u for u in _jsonl(GOLD / "units.jsonl")}
    doc_order = [d["doc_id"] for d in json.loads((GOLD / "DOC_LIST.json").read_text(encoding="utf-8"))["documents"]]

    if len(raw) != len(units) or {r["uid"] for r in raw} != set(units):
        sys.exit("annotations_raw.jsonl and units.jsonl do not hold the same uids")
    if any(r["missing_lenses"] for r in raw):
        sys.exit("some units miss a lens annotation; annotate them before finalising (protocol 4.3)")
    queued = {r["uid"] for r in raw if r["adjudication_reasons"]}
    if queued != set(queue) or set(adj) != queued:
        sys.exit(f"adjudication must cover the queue exactly: queue {len(queued)}, decisions {len(adj)}, "
                 f"missing {sorted(queued - set(adj))}, extra {sorted(set(adj) - queued)}")
    for uid, a in adj.items():
        if a["label"] not in ("Major", "OK") or (a["label"] == "Major") != (a["severity"] in MAJOR):
            sys.exit(f"{uid}: label {a['label']} does not match severity {a['severity']}")
        if a["span_mt"] and a["span_mt"] not in units[uid]["mt"]:
            sys.exit(f"{uid}: adjudicated span_mt is not in the MT")
        if not a.get("reason"):
            sys.exit(f"{uid}: adjudication has no reason")

    # ---- labels, still blind to the MT system ----
    labels = []
    for r in raw:
        uid, ann = r["uid"], r["annotations"]
        votes = {l: ann[l]["severity"] for l in LENSES}
        n_major = sum(v in MAJOR for v in votes.values())
        majority = "Major" if n_major >= 2 else "OK"
        if majority != r["majority_label"] or n_major != r["n_major"]:
            sys.exit(f"{uid}: vote recount differs from annotations_raw.jsonl")
        if uid in adj:
            a = adj[uid]
            label, severity, category, span, reason = a["label"], a["severity"], a["category"], a["span_mt"], a["reason"]
        else:
            label = majority
            severity = sorted(votes.values(), key=SEV_ORDER.index)[1]
            lens = next(l for l in LENSES if votes[l] == severity)
            category = ann[lens]["category"] if severity != "none" else "none"
            span = ann[lens]["span_mt"] if severity != "none" else ""
            expl = ann[lens]["explanation"] or ("no lens found an error" if severity == "none" else "")
            reason = f"panel {n_major}/3 Major; {lens}: {expl}"
        reason_q = r["adjudication_reason"]
        label_prereg = majority if reason_q == "minority_major_low_conf" else label
        labels.append({"uid": uid, "doc_id": r["doc_id"], "seg_index": r["seg_index"], "system": None,
                       "label": label, "severity": severity, "category": category, "span_mt": span,
                       "reason": reason, "votes": votes, "adjudicated": uid in adj,
                       "segment_id": None, "n_major": n_major, "adjudication_reason": reason_q,
                       "label_preregistered": label_prereg})

    # ---- join the system key only now ----
    key_path = GOLD / "key_systems.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    for row in labels:
        k = key["units"][row["uid"]]
        row["system"], row["segment_id"] = k["system"], k["segment_id"]
        if k["segment_id"] != f"{row['doc_id']}-{row['seg_index'] + 1:03d}":
            sys.exit(f"{row['uid']}: key segment_id {k['segment_id']} does not match doc_id/seg_index")
    pairs = Counter((r["doc_id"], r["seg_index"], r["system"]) for r in labels)
    if max(pairs.values()) != 1:
        sys.exit("a (doc, segment, system) triple appears twice")
    labels.sort(key=lambda r: (doc_order.index(r["doc_id"]), r["seg_index"], r["system"]))
    out = GOLD / "labels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    systems = sorted({r["system"] for r in labels})

    def n_maj(rs, field="label"):
        return sum(r[field] == "Major" for r in rs)

    def rate(rs):
        return n_maj(rs) / len(rs) if rs else None

    per_doc = {}
    for d in doc_order:
        rs = [r for r in labels if r["doc_id"] == d]
        per_doc[d] = {"units": len(rs), "segments": len(rs) // len(systems), "major": n_maj(rs),
                      "major_rate": rate(rs), "critical": sum(r["severity"] == "critical" for r in rs),
                      "adjudicated": sum(r["adjudicated"] for r in rs),
                      "by_system": {s: {"units": sum(r["system"] == s for r in rs),
                                        "major": n_maj([r for r in rs if r["system"] == s])} for s in systems}}
    per_system = {}
    for s in systems:
        rs = [r for r in labels if r["system"] == s]
        per_system[s] = {"units": len(rs), "major": n_maj(rs), "major_preregistered": n_maj(rs, "label_preregistered"),
                         "critical": sum(r["severity"] == "critical" for r in rs),
                         "major_rate": doc_bootstrap(rs, rate),
                         "docs_with_zero_major": sum(1 for d in doc_order
                                                     if not n_maj([r for r in rs if r["doc_id"] == d])),
                         "major_categories": dict(Counter(r["category"] for r in rs if r["label"] == "Major")
                                                  .most_common())}
    # both systems Major on the same segment
    seg_major = defaultdict(set)
    for r in labels:
        if r["label"] == "Major":
            seg_major[(r["doc_id"], r["seg_index"])].add(r["system"])
    both = sum(len(v) == len(systems) for v in seg_major.values())

    # ---- adjudication ----
    adj_rows = [r for r in labels if r["adjudicated"]]
    adjudication = {
        "n": len(adj_rows), "share": len(adj_rows) / len(labels),
        "by_reason": {k: {"n": v, "share": v / len(labels)}
                      for k, v in Counter(r["adjudication_reason"] for r in adj_rows).items()},
        "preregistered_only": {"n": sum(r["adjudication_reason"] in ("critical", "minority_major_conf>=0.7")
                                        for r in adj_rows)},
        "outcome": dict(Counter(f"{r['adjudication_reason']}: majority {('Major' if r['n_major'] >= 2 else 'OK')}"
                                f" -> final {r['label']}" for r in adj_rows)),
        "overturned_majority": sum((r["n_major"] >= 2) != (r["label"] == "Major") for r in adj_rows),
        "decisions_file": {"path": _rel(adj_path), "sha256": _sha(adj_path),
                           "decided_at": sorted({a["decided_at"] for a in adj.values()})},
        "adjudicator_confidence": {uid: adj[uid]["adjudicator_confidence"] for uid in sorted(adj)},
    }

    # ---- non-binding audit ----
    audit = None
    audit_path = GOLD / "adjudicator_audit.jsonl"
    if audit_path.exists():
        a_rows = _jsonl(audit_path)
        final = {r["uid"]: r["label"] for r in labels}
        audit = {"path": _rel(audit_path), "sha256": _sha(audit_path), "binding": False, "groups": {}}
        for g in sorted({a["group"] for a in a_rows}):
            rs = [a for a in a_rows if a["group"] == g]
            audit["groups"][g] = {"n": len(rs), "agree_with_gold": sum(a["audit_label"] == final[a["uid"]] for a in rs),
                                  "would_flip": sorted(a["uid"] for a in rs if a["audit_label"] != final[a["uid"]])}

    # ---- agreement with the earlier labels (OPUS-MT only) ----
    old = {}
    for p in OLD:
        for o in _jsonl(p):
            if o["system"] != "marian":
                continue
            k = (o["id"], o["system"])
            if k in old:
                sys.exit(f"{k} appears twice in the earlier labels")
            old[k] = {"verdict": o["verdict"], "mt": o["mt"], "doc_id": o["doc_id"], "explanation": o.get("explanation"),
                      "source": o.get("final_source"), "file": p.name}
    overlap = [(r, old[(r["segment_id"], "marian")]) for r in labels
               if r["system"] == "marian" and (r["segment_id"], "marian") in old]
    mt_mismatch = [r["uid"] for r, o in overlap if " ".join(o["mt"].split()) != " ".join(units[r["uid"]]["mt"].split())]

    def old_bin(v):
        return "Major" if v == "major_error" else "OK"

    def confusion(prs):
        c = Counter((r["label"], old_bin(o["verdict"])) for r, o in prs)
        return {"gold_Major_old_Major": c[("Major", "Major")], "gold_Major_old_OK": c[("Major", "OK")],
                "gold_OK_old_Major": c[("OK", "Major")], "gold_OK_old_OK": c[("OK", "OK")]}

    whole_doc = {"irs-failure-to-file-penalty", "fda-medicine-disposal", "eu-air-passenger-rights"}
    old_cmp = {
        "old_files": [_rel(p) for p in OLD],
        "old_marian_records": len(old),
        "overlap_units": len(overlap),
        "gold_marian_units_on_original_five_docs": sum(r["system"] == "marian" and r["doc_id"] in
                                                       {o["doc_id"] for o in old.values()} for r in labels),
        "mt_text_mismatches": mt_mismatch,
        "confusion": confusion(overlap),
        "cohen_kappa": cohen_kappa([r["label"] for r, _ in overlap], [old_bin(o["verdict"]) for _, o in overlap]),
        "raw_agreement": sum(r["label"] == old_bin(o["verdict"]) for r, o in overlap) / len(overlap) if overlap else None,
        "gold_vs_old_3way": {f"{lab}|{v}": n for (lab, v), n in sorted(Counter((r["label"], o["verdict"])
                                                                          for r, o in overlap).items())},
        "gold_severity_vs_old_3way": {f"{s}|{v}": n for (s, v), n in sorted(Counter((r["severity"], o["verdict"])
                                                                            for r, o in overlap).items())},
        "by_coverage": {
            "whole_document (IRS, FDA, EU: every OPUS-MT segment labelled)":
                confusion([(r, o) for r, o in overlap if r["doc_id"] in whole_doc]),
            "review_pool_only (NIDDK, Ready hurricanes)":
                confusion([(r, o) for r, o in overlap if r["doc_id"] not in whole_doc]),
        },
        "by_doc": {d: confusion([(r, o) for r, o in overlap if r["doc_id"] == d])
                   for d in doc_order if any(r["doc_id"] == d for r, _ in overlap)},
        "disagreements_by_old_pass": dict(Counter(o["source"] for r, o in overlap
                                                  if r["label"] != old_bin(o["verdict"]))),
        "disagreements_by_panel_votes": {f"{k}/3": v for k, v in sorted(Counter(
            r["n_major"] for r, o in overlap if r["label"] != old_bin(o["verdict"])).items())},
        "disagreements": [{"uid": r["uid"], "segment_id": r["segment_id"], "gold": r["label"],
                           "gold_votes": r["votes"], "gold_reason": r["reason"], "old_verdict": o["verdict"],
                           "old_source": o["source"], "old_explanation": o["explanation"]}
                          for r, o in overlap if r["label"] != old_bin(o["verdict"])],
    }

    stats = {
        "built_by": "scripts/gold_finalize.py",
        "protocol": "docs/ANNOTATION_PROTOCOL.md v2 (frozen 2026-10-14), section 4",
        "files": {"labels": {"path": _rel(out), "sha256": _sha(out)},
                  "annotations_raw": {"path": "data/realdocs/gold/annotations_raw.jsonl",
                                      "sha256": _sha(GOLD / "annotations_raw.jsonl")},
                  "key_systems": {"path": _rel(key_path), "sha256": _sha(key_path)},
                  "units": {"path": "data/realdocs/gold/units.jsonl", "sha256": _sha(GOLD / "units.jsonl")}},
        "bootstrap": {"unit": "document (10 clusters, resampled with replacement)", "n_boot": N_BOOT, "seed": SEED,
                      "ci": "percentile 2.5-97.5"},
        "units": len(labels), "documents": len(doc_order), "segments": len(labels) // len(systems),
        "systems": {s: key["systems"][s] for s in systems},
        "major": n_maj(labels), "major_rate": doc_bootstrap(labels, rate),
        "major_preregistered": n_maj(labels, "label_preregistered"),
        "critical": sum(r["severity"] == "critical" for r in labels),
        "final_severity_counts": {s: sum(r["severity"] == s for r in labels) for s in SEV_ORDER},
        "vote_distribution_n_major": {str(k): sum(r["n_major"] == k for r in labels) for k in range(4)},
        "segments_major_in_both_systems": both,
        "segments_major_in_any_system": len(seg_major),
        "per_system": per_system,
        "per_doc": per_doc,
        "lens_agreement": lens_agreement(labels),
        "lens_agreement_by_system": {s: {k: v for k, v in lens_agreement([r for r in labels if r["system"] == s]).items()
                                         if k in ("n_units", "fleiss_kappa_binary", "fleiss_kappa_3class")}
                                     for s in systems},
        "adjudication": adjudication,
        "audit_non_binding": audit,
        "agreement_with_old_labels": old_cmp,
    }
    (GOLD / "gold_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    la = stats["lens_agreement"]
    print(f"{len(labels)} units, {stats['segments']} segments x {len(systems)} systems, {stats['documents']} documents")
    print(f"Major {stats['major']} ({stats['major'] / len(labels):.1%}), critical {stats['critical']}, "
          f"pre-registered labelling Major {stats['major_preregistered']}")
    for s in systems:
        ps = per_system[s]
        ci = ps["major_rate"]["ci95"]
        print(f"  {s}: {ps['major']}/{ps['units']} Major ({ps['major_rate']['point']:.1%}, doc-bootstrap 95% CI "
              f"{ci[0]:.1%}-{ci[1]:.1%}), critical {ps['critical']}")
    for d in doc_order:
        pd = per_doc[d]
        print(f"  {d}: {pd['major']}/{pd['units']} " + ", ".join(f"{s} {pd['by_system'][s]['major']}" for s in systems))
    print(f"Fleiss kappa binary {la['fleiss_kappa_binary']['point']:.3f} {la['fleiss_kappa_binary']['ci95']}, "
          f"3-class {la['fleiss_kappa_3class']['point']:.3f}")
    print(f"adjudicated {adjudication['n']} ({adjudication['share']:.2%}), overturned {adjudication['overturned_majority']}")
    print(f"old labels: overlap {old_cmp['overlap_units']} OPUS-MT units, MT mismatches {len(mt_mismatch)}, "
          f"confusion {old_cmp['confusion']}, kappa {old_cmp['cohen_kappa']}")
    print(f"wrote {_rel(out)} (sha256 {_sha(out)[:12]}) and {_rel(GOLD / 'gold_stats.json')}")


if __name__ == "__main__":
    main()
