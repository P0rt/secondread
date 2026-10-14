"""Build the blind annotation units of the gold corpus and split them into batch files.

usage: uv run --frozen python -I scripts/gold_units.py

One unit per (document, segment, MT system) for every document in data/realdocs/gold/DOC_LIST.json and
every system in SYSTEMS, read from data/realdocs/gold/mt/<system>/<doc_id>.jsonl:
  {uid, doc_id, seg_index, context_src, src, mt, ref_es}
seg_index is the 0-based row of the segment in the document's aligned.jsonl; context_src holds the previous
two source segments of the same document (fewer at the start), the same context the LLM MT step saw.
Nothing is excluded: identical outputs of the two systems still give two units.

Blinding: the unit carries no system name, file name or system-specific field. Units are ordered by document
(DOC_LIST order) and shuffled within each document with one random.Random(0) used in that order; uids
g0001, g0002, ... are given in the final order, so they reveal neither system nor segment order.
The uid -> system mapping goes to data/realdocs/gold/key_systems.json, which annotators never see.
Batches: data/realdocs/gold/batches/batch_XX.jsonl, consecutive runs of the unit order (documents stay
grouped; a document may continue into the next batch), sizes as equal as possible around 110.
Old labels, adjudications and system scores are never read.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "data" / "realdocs" / "gold"
SYSTEMS = ["marian", "deepseek-v3.1"]
SEED = 0
TARGET_BATCH = 110
FIELDS = ["uid", "doc_id", "seg_index", "context_src", "src", "mt", "ref_es"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    lst = json.loads((GOLD / "DOC_LIST.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    units, key = [], {}
    per_doc = {}
    for d in lst["documents"]:
        aligned = REPO / d["aligned"]
        if _sha(aligned) != d["aligned_sha256"]:
            sys.exit(f"{aligned} does not match DOC_LIST.json")
        rows = [json.loads(line) for line in aligned.open(encoding="utf-8")]
        doc_units = []
        for system in SYSTEMS:
            mt_rows = [json.loads(line) for line in (GOLD / "mt" / system / f"{d['doc_id']}.jsonl").open(encoding="utf-8")]
            if [(m["id"], m["src"], m["ref_es"]) for m in mt_rows] != [(r["id"], r["src"], r["ref_es"]) for r in rows]:
                sys.exit(f"mt/{system}/{d['doc_id']}.jsonl does not match aligned.jsonl")
            for i, (r, m) in enumerate(zip(rows, mt_rows)):
                if not m["mt"].strip():
                    sys.exit(f"empty MT: {system} {r['id']}")
                doc_units.append(({"doc_id": d["doc_id"], "seg_index": i,
                                   "context_src": [x["src"] for x in rows[max(0, i - 2):i]],
                                   "src": r["src"], "mt": m["mt"], "ref_es": r["ref_es"]}, system, r["id"]))
        rng.shuffle(doc_units)
        per_doc[d["doc_id"]] = len(doc_units)
        units.extend(doc_units)
    out_units = []
    for n, (u, system, seg_id) in enumerate(units, 1):
        uid = f"g{n:04d}"
        out_units.append({"uid": uid, **u})
        key[uid] = {"system": system, "segment_id": seg_id}
    n_batches = max(1, round(len(out_units) / TARGET_BATCH))
    size = len(out_units) / n_batches
    bdir = GOLD / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for old in bdir.glob("batch_*.jsonl"):
        old.unlink()
    batches = []
    for b in range(n_batches):
        chunk = out_units[math.floor(b * size):math.floor((b + 1) * size)]
        p = bdir / f"batch_{b + 1:02d}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for u in chunk:
                f.write(json.dumps({k: u[k] for k in FIELDS}, ensure_ascii=False) + "\n")
        batches.append({"file": str(p.relative_to(REPO)), "units": len(chunk), "first_uid": chunk[0]["uid"],
                        "last_uid": chunk[-1]["uid"], "docs": sorted({u["doc_id"] for u in chunk},
                                                                      key=list(per_doc).index), "sha256": _sha(p)})
        for u in chunk:
            key[u["uid"]]["batch"] = b + 1
    with (GOLD / "units.jsonl").open("w", encoding="utf-8") as f:
        for u in out_units:
            f.write(json.dumps({k: u[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    (GOLD / "key_systems.json").write_text(json.dumps({
        "warning": "ANNOTATOR-BLIND KEY: maps each unit to the MT system that produced it. Never show to annotators.",
        "systems": {s: json.loads((GOLD / "mt" / "MT_META.json").read_text())["systems"][s]["model_id"] for s in SYSTEMS},
        "units": key}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest = {
        "units_file": "data/realdocs/gold/units.jsonl", "units_sha256": _sha(GOLD / "units.jsonl"),
        "fields": FIELDS, "n_units": len(out_units), "n_docs": len(per_doc), "n_systems": len(SYSTEMS),
        "units_per_doc": per_doc, "order": "documents in DOC_LIST order; within a document shuffled with "
        f"random.Random({SEED}) (one generator, used document by document); uids assigned in final order",
        "context_src": "previous two source segments of the same document in aligned order",
        "exclusions": "none", "batches": batches,
        "blind": "units carry no system identity; the mapping is in data/realdocs/gold/key_systems.json only",
        "built_by": "scripts/gold_units.py",
    }
    (GOLD / "batches" / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                                   encoding="utf-8")
    print(f"{len(out_units)} units, {len(per_doc)} docs, {n_batches} batches "
          f"({', '.join(str(b['units']) for b in batches)})")


if __name__ == "__main__":
    main()
