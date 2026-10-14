"""Merge the three lens annotations of the gold panel and build the adjudication queue (protocol v2, section 4).

usage:
  uv run --frozen python -I scripts/gold_merge.py --import data/realdocs/gold/annotations/panel_return.json
  uv run --frozen python -I scripts/gold_merge.py            # merge from the per-lens files only

--import splits a workflow return [{lens, batch, units: [...]}, ...] into the verbatim per-lens files
data/realdocs/gold/annotations/<lens>_batch_XX.jsonl (one record per line, "lens" added, nothing else changed).
An existing file with different content is never overwritten.

The merge reads only the batch files (verified against batches/MANIFEST.json) and the per-lens files. It never
reads key_systems.json, MT metadata, automatic scores or earlier real-document labels.

Outputs (data/realdocs/gold/):
  annotations_raw.jsonl   one line per uid: unit position, the three lens records, the vote and the queue reasons
  to_adjudicate.jsonl     one line per queued uid: the blind unit fields + the three annotations + the reasons
  merge_summary.json      counts, missing/duplicate/unknown uids, QA flags, agreement (protocol section 4.5)

Aggregation (docs/ANNOTATION_PROTOCOL.md section 4, frozen v2): a lens votes Major on major|critical; a unit is
Major when >= 2 of 3 lenses vote Major. Queue reasons, all recorded, the first one in this order is
adjudication_reason:
  critical                   any lens says critical (pre-registered)
  minority_major_conf>=0.7   exactly one Major vote, with confidence >= 0.7 (pre-registered)
  minority_major_low_conf    exactly one Major vote, confidence < 0.7 (amendment A1)
A unit with a missing lens annotation gets majority_label null and reason missing_annotation; it must be
annotated before finalising.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "data" / "realdocs" / "gold"
ADIR = GOLD / "annotations"
LENSES = ["mqm", "posteditor", "reader"]
SEVERITIES = ["none", "minor", "major", "critical"]
MAJOR = {"major", "critical"}
CATEGORIES = {"accuracy/mistranslation", "accuracy/omission", "accuracy/addition", "accuracy/untranslated",
              "terminology", "fluency", "style", "locale", "none"}
MEANING_CATS = {"accuracy/mistranslation", "accuracy/omission", "accuracy/addition", "accuracy/untranslated",
                "terminology"}
CONF_THRESHOLD = 0.7
REASON_ORDER = ["missing_annotation", "critical", "minority_major_conf>=0.7", "minority_major_low_conf"]
PREREGISTERED = {"critical", "minority_major_conf>=0.7"}
UNIT_FIELDS = ["uid", "doc_id", "seg_index", "context_src", "src", "mt", "ref_es"]
RECORD_FIELDS = ["severity", "category", "span_mt", "explanation", "confidence"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def import_return(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    ADIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    for block in data:
        lens, batch = block["lens"], Path(block["batch"]).stem
        if lens not in LENSES:
            sys.exit(f"unknown lens {lens!r}")
        if (lens, batch) in seen:
            sys.exit(f"{lens} {batch} appears twice in {path}")
        seen.add((lens, batch))
        text = "".join(json.dumps({"lens": lens, **r}, ensure_ascii=False) + "\n" for r in block["units"])
        out = ADIR / f"{lens}_{batch}.jsonl"
        if out.exists() and out.read_text(encoding="utf-8") != text:
            sys.exit(f"{_rel(out)} exists with different content; refusing to overwrite")
        out.write_text(text, encoding="utf-8")
    print(f"imported {len(seen)} lens-batch files from {_rel(path)} (sha256 {_sha(path)[:12]})")


def fleiss_kappa(table: list[list[int]]) -> float | None:
    """table[i][j] = raters putting item i in category j; same rater count per item."""
    if not table:
        return None
    n = sum(table[0])
    N = len(table)
    k = len(table[0])
    p_j = [sum(row[j] for row in table) / (N * n) for j in range(k)]
    P_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in table]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    return None if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def cohen_kappa(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def cls3(sev: str) -> str:
    return "major+" if sev in MAJOR else sev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp", type=Path, help="workflow return JSON to split into per-lens files")
    args = ap.parse_args()
    if args.imp:
        import_return(args.imp if args.imp.is_absolute() else REPO / args.imp)

    manifest = json.loads((GOLD / "batches" / "MANIFEST.json").read_text(encoding="utf-8"))
    units, unit_batch, batch_files = {}, {}, {}
    for b in manifest["batches"]:
        p = REPO / b["file"]
        if _sha(p) != b["sha256"]:
            sys.exit(f"{b['file']} does not match MANIFEST.json")
        batch_files[p.stem] = b["sha256"]
        for line in p.open(encoding="utf-8"):
            u = json.loads(line)
            if set(u) != set(UNIT_FIELDS):
                sys.exit(f"{b['file']} {u.get('uid')}: unexpected fields {sorted(set(u) ^ set(UNIT_FIELDS))}")
            if u["uid"] in units:
                sys.exit(f"duplicate uid {u['uid']} in batches")
            units[u["uid"]] = u
            unit_batch[u["uid"]] = p.stem
    if len(units) != manifest["n_units"]:
        sys.exit(f"{len(units)} units in batches, MANIFEST says {manifest['n_units']}")

    ann = {lens: {} for lens in LENSES}
    issues = {"duplicate": [], "unknown_uid": [], "wrong_batch": [], "bad_severity": [], "bad_confidence": [],
              "bad_category": [], "missing_category_on_error": [], "missing_explanation": [],
              "explanation_over_25_words": [], "span_not_in_mt": [], "span_on_none": [],
              "major_with_form_category": [], "reader_major_without_read_as": []}
    ann_files = {}
    missing_files = []
    for lens in LENSES:
        for bstem in batch_files:
            p = ADIR / f"{lens}_{bstem}.jsonl"
            if not p.exists():
                missing_files.append(_rel(p))
                continue
            ann_files[_rel(p)] = _sha(p)
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                uid = r.get("uid")
                where = f"{lens} {uid}"
                if r.get("lens", lens) != lens:
                    sys.exit(f"{_rel(p)}: record for lens {r.get('lens')}")
                if uid not in units:
                    issues["unknown_uid"].append(where)
                    continue
                if unit_batch[uid] != bstem:
                    issues["wrong_batch"].append(f"{where} in {bstem}, unit is in {unit_batch[uid]}")
                if uid in ann[lens]:
                    issues["duplicate"].append(where)
                    continue
                sev = str(r.get("severity", "")).strip().lower()
                if sev not in SEVERITIES:
                    issues["bad_severity"].append(f"{where}: {r.get('severity')!r}")
                    continue
                conf = r.get("confidence")
                if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
                    issues["bad_confidence"].append(f"{where}: {conf!r}")
                    conf = None
                cat = (r.get("category") or "").strip()
                if not cat:
                    if sev == "none":
                        cat = "none"
                    else:
                        issues["missing_category_on_error"].append(where)
                elif cat not in CATEGORIES:
                    issues["bad_category"].append(f"{where}: {cat!r}")
                span = r.get("span_mt") or ""
                expl = (r.get("explanation") or "").strip()
                if not expl:
                    issues["missing_explanation"].append(f"{where} ({sev})")
                elif len(expl.split()) > 25:
                    issues["explanation_over_25_words"].append(f"{where} ({len(expl.split())})")
                span_in_mt = None
                if span:
                    span_in_mt = span in units[uid]["mt"] or span.strip('"“”\' ') in units[uid]["mt"]
                    if not span_in_mt:
                        issues["span_not_in_mt"].append(f"{where}: {span!r}")
                    if sev == "none":
                        issues["span_on_none"].append(where)
                if sev in MAJOR and cat and cat not in MEANING_CATS:
                    issues["major_with_form_category"].append(f"{where}: {sev}/{cat}")
                if lens == "reader" and sev in MAJOR and not expl.lower().startswith("read as"):
                    issues["reader_major_without_read_as"].append(where)
                ann[lens][uid] = {"severity": sev, "category": cat, "span_mt": span, "explanation": expl,
                                  "confidence": conf, "span_in_mt": span_in_mt}

    missing = {lens: sorted(set(units) - set(ann[lens])) for lens in LENSES}
    raw_rows, queue = [], []
    for uid in sorted(units):
        u = units[uid]
        recs = {lens: ann[lens].get(uid) for lens in LENSES}
        present = [lens for lens in LENSES if recs[lens]]
        votes = {lens: recs[lens]["severity"] in MAJOR for lens in present}
        n_major = sum(votes.values())
        reasons = []
        if len(present) < len(LENSES):
            reasons.append("missing_annotation")
        if any(recs[lens]["severity"] == "critical" for lens in present):
            reasons.append("critical")
        if len(present) == len(LENSES) and n_major == 1:
            lone = next(lens for lens in present if votes[lens])
            c = recs[lone]["confidence"]
            reasons.append("minority_major_conf>=0.7" if c is not None and c >= CONF_THRESHOLD
                           else "minority_major_low_conf")
        reasons.sort(key=REASON_ORDER.index)
        majority = None if len(present) < len(LENSES) else ("Major" if n_major >= 2 else "OK")
        row = {"uid": uid, "doc_id": u["doc_id"], "seg_index": u["seg_index"], "batch": unit_batch[uid],
               "annotations": {lens: ({k: recs[lens][k] for k in RECORD_FIELDS} if recs[lens] else None)
                               for lens in LENSES},
               "missing_lenses": [lens for lens in LENSES if not recs[lens]],
               "votes_major": votes, "n_major": n_major, "majority_label": majority,
               "adjudication_reason": reasons[0] if reasons else None, "adjudication_reasons": reasons,
               "preregistered_queue": bool(PREREGISTERED & set(reasons)) or "missing_annotation" in reasons}
        raw_rows.append(row)
        if reasons:
            queue.append({**{k: u[k] for k in UNIT_FIELDS}, "annotations": row["annotations"],
                          "missing_lenses": row["missing_lenses"], "adjudication_reason": row["adjudication_reason"],
                          "adjudication_reasons": reasons})

    with (GOLD / "annotations_raw.jsonl").open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (GOLD / "to_adjudicate.jsonl").open("w", encoding="utf-8") as f:
        for row in queue:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    complete = [r for r in raw_rows if not r["missing_lenses"]]
    sev = {lens: [r["annotations"][lens]["severity"] for r in complete] for lens in LENSES}
    bin_ = {lens: [s in MAJOR for s in sev[lens]] for lens in LENSES}
    c3 = ["none", "minor", "major+"]
    agreement = {
        "n_units": len(complete),
        "fleiss_kappa_binary": fleiss_kappa([[sum(not bin_[l][i] for l in LENSES), sum(bin_[l][i] for l in LENSES)]
                                             for i in range(len(complete))]),
        "fleiss_kappa_3class": fleiss_kappa([[sum(cls3(sev[l][i]) == c for l in LENSES) for c in c3]
                                             for i in range(len(complete))]),
        "cohen_kappa_binary": {f"{a}-{b}": cohen_kappa(bin_[a], bin_[b]) for a, b in combinations(LENSES, 2)},
        "cohen_kappa_3class": {f"{a}-{b}": cohen_kappa([cls3(s) for s in sev[a]], [cls3(s) for s in sev[b]])
                               for a, b in combinations(LENSES, 2)},
    }
    by_doc = {}
    for r in complete:
        d = by_doc.setdefault(r["doc_id"], {"units": 0, "majority_major": 0, "queued": 0})
        d["units"] += 1
        d["majority_major"] += r["majority_label"] == "Major"
        d["queued"] += bool(r["adjudication_reasons"])
    n = len(raw_rows)
    summary = {
        "protocol": "docs/ANNOTATION_PROTOCOL.md v2 (frozen 2026-10-14), section 4",
        "built_by": "scripts/gold_merge.py",
        "inputs": {"manifest": "data/realdocs/gold/batches/MANIFEST.json", "batch_sha256_verified": batch_files,
                   "annotation_files_sha256": ann_files, "missing_annotation_files": missing_files},
        "n_units": n,
        "annotations_per_lens": {lens: len(ann[lens]) for lens in LENSES},
        "missing_uids": missing,
        "missing_uids_any": sorted(set().union(*missing.values())),
        "severity_counts": {lens: dict(Counter(a["severity"] for a in ann[lens].values())) for lens in LENSES},
        "major_votes_per_lens": {lens: sum(a["severity"] in MAJOR for a in ann[lens].values()) for lens in LENSES},
        "vote_distribution_n_major": {str(k): sum(r["n_major"] == k for r in complete) for k in range(4)},
        "majority_major": sum(r["majority_label"] == "Major" for r in raw_rows),
        "majority_ok": sum(r["majority_label"] == "OK" for r in raw_rows),
        "queue": {
            "n": len(queue), "share": len(queue) / n,
            "by_primary_reason": dict(Counter(q["adjudication_reason"] for q in queue)),
            "by_any_reason": dict(Counter(x for q in queue for x in q["adjudication_reasons"])),
            "preregistered_only_n": sum(r["preregistered_queue"] for r in raw_rows),
            "preregistered_only_share": sum(r["preregistered_queue"] for r in raw_rows) / n,
            "two_one_majority_major_not_queued": sum(r["n_major"] == 2 and not r["adjudication_reasons"]
                                                     for r in raw_rows),
        },
        "by_doc": by_doc,
        "agreement": agreement,
        "qa_flags": {k: {"n": len(v), "items": v} for k, v in issues.items()},
    }
    (GOLD / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                            encoding="utf-8")
    print(f"{n} units; annotations per lens {summary['annotations_per_lens']}; missing {len(summary['missing_uids_any'])}")
    print(f"vote distribution (n Major of 3): {summary['vote_distribution_n_major']}; majority Major "
          f"{summary['majority_major']}")
    print(f"queue {len(queue)} ({len(queue) / n:.1%}): {summary['queue']['by_primary_reason']}; "
          f"pre-registered only {summary['queue']['preregistered_only_n']}")
    print("QA flags: " + ", ".join(f"{k}={len(v)}" for k, v in issues.items() if v))
    print(f"wrote {_rel(GOLD / 'annotations_raw.jsonl')}, {_rel(GOLD / 'to_adjudicate.jsonl')}, "
          f"{_rel(GOLD / 'merge_summary.json')}")


if __name__ == "__main__":
    main()
