"""Per-document real-document numbers at several fixed flag lines (docs/REALDOCS.md tables).

usage: PYTHONPATH=. uv run --frozen python scripts/realdocs_eval.py [--scored data/realdocs/scored.jsonl] \
           [--adapter v1] [--out runs/realdocs_review/eval.json]

Evaluation only. The real documents choose nothing: the lines come from secondread/thresholds.json
(picked on WMT dev) and are fixed before this script reads a label.
  <adapter> f1      the adapter's F1 line for en-es (v1: 0.7019, the line REALDOCS.md was first written at)
  <adapter> review  the adapter's review line for en-es (v1: 0.4984, the app's default since the review pass)
Each line is applied to SecondRead's fused risk in the scored file (scripts/realdocs_score.py output: the
backend pipeline, model plus calibrated rule weights). The riskiest-10% count does not depend on a line.

Labels, as scripts/make_realdocs_demo.py reads them:
  * the three chosen OPUS-MT documents (IRS, FDA, EU 261): every segment adjudicated against the official
    Spanish (data/realdocs/adjudication_final.jsonl + adjudication_fill.jsonl); Major = major_error;
  * the WMT24 documents: human MQM gold_label (and, for comparison, our adjudication where it exists);
  * official_es (the publishers' own Spanish, 5 documents): a control; every flag counts as a false flag;
  * NIDDK and Ready.gov OPUS-MT, and all NLLB-200 output: only the review pool is adjudicated, so flags
    outside the pool are counted as "unlabelled", never as right or wrong.
Gemma 4 31B direct at 0.971 (its en-ru dev F1 line) is carried along unchanged for reference.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "realdocs"
LP = "en-es"
CHOSEN = ["irs-failure-to-file-penalty", "fda-medicine-disposal", "eu-air-passenger-rights"]
OTHER_MARIAN = ["niddk-hypoglycemia", "ready-hurricanes"]
WMT = ["wmt24_the_other_side_stormfall", "wmt24_forever_snow"]
ALL5 = CHOSEN + OTHER_MARIAN
GEMMA_HI = 0.971


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def adjudications() -> dict[tuple[str, str], dict]:
    adj = {}
    for p in (REAL / "adjudication_final.jsonl", REAL / "adjudication_fill.jsonl"):
        for r in load_jsonl(p):
            k = (r["id"], r["system"])
            assert k not in adj, f"{k} adjudicated twice"
            adj[k] = r
    return adj


def doc_stats(rows: list[dict], lines: dict[str, float], label_of) -> dict:
    """label_of(row) -> 'Major' | 'OK' | None (None = not labelled)."""
    n = len(rows)
    by_risk = sorted(rows, key=lambda r: -r["secondread"]["risk"])  # stable, as make_realdocs_demo.stats
    top = max(1, round(0.1 * n))
    maj = [r for r in rows if label_of(r) == "Major"]
    e = {"segments": n, "major": len(maj), "labelled": sum(label_of(r) is not None for r in rows),
         "top10pct_n": top, "major_in_top10pct": sum(label_of(r) == "Major" for r in by_risk[:top]),
         "majors": [{"id": r["id"], "risk": r["secondread"]["risk"],
                     "gemma": r["gemma"]["risk"] if r.get("gemma") else None} for r in maj],
         "gemma_flagged": sum(bool(r.get("gemma")) and r["gemma"]["risk"] >= GEMMA_HI for r in rows),
         "gemma_caught": sum(bool(r.get("gemma")) and r["gemma"]["risk"] >= GEMMA_HI for r in maj),
         "gemma_scored": sum(bool(r.get("gemma")) for r in rows), "lines": {}}
    for name, t in lines.items():
        flag = [r for r in rows if r["secondread"]["risk"] >= t]
        e["lines"][name] = {
            "threshold": t, "flagged": len(flag), "flag_rate": len(flag) / n,
            "caught": sum(label_of(r) == "Major" for r in flag),
            "false": sum(label_of(r) == "OK" for r in flag),
            "unlabelled": sum(label_of(r) is None for r in flag),
            "flags": [{"id": r["id"], "risk": r["secondread"]["risk"], "label": label_of(r),
                       "verdict": (r.get("adj") or {}).get("verdict")} for r in flag],
        }
    return e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scored", default=str(REAL / "scored.jsonl"))
    ap.add_argument("--adapter", default="v1")
    ap.add_argument("--out", default="runs/realdocs_review/eval.json")
    a = ap.parse_args()
    thr = json.loads((ROOT / "secondread/thresholds.json").read_text())["adapters"][a.adapter]
    lines = {p: thr[p]["thresholds"][LP] for p in ("f1", "review") if p in thr}

    scored = load_jsonl(Path(a.scored))
    adj = adjudications()
    for r in scored:
        r["adj"] = adj.get((r["id"], r["system"]))

    def rows_of(doc, system):
        rs = [r for r in scored if r["doc_id"] == doc and r["system"] == system]
        assert rs, (doc, system)
        return rs

    def adj_label(r):
        if r["adj"] is None:
            return None
        return "Major" if r["adj"]["verdict"] == "major_error" else "OK"

    out: dict = {"scored": str(Path(a.scored)), "adapter": a.adapter, "lines": lines, "docs": {}}
    for doc in CHOSEN:
        rs = rows_of(doc, "marian")
        assert all(r["adj"] for r in rs), f"{doc}: not every segment adjudicated"
        out["docs"][f"{doc}|marian"] = doc_stats(rs, lines, adj_label)
    for doc in OTHER_MARIAN:
        out["docs"][f"{doc}|marian"] = doc_stats(rows_of(doc, "marian"), lines, adj_label)
    for doc in ALL5:
        out["docs"][f"{doc}|nllb"] = doc_stats(rows_of(doc, "nllb"), lines, adj_label)
        # control: the publisher's own Spanish; every flag is a false flag
        out["docs"][f"{doc}|official_es"] = doc_stats(rows_of(doc, "official_es"), lines, lambda r: "OK")
    for doc in WMT:
        rs = rows_of(doc, "MSLC")
        out["docs"][f"{doc}|MSLC|mqm"] = doc_stats(rs, lines, lambda r: r["gold_label"])
        ours = [r for r in rs if r["adj"]]
        if ours:
            out["docs"][f"{doc}|MSLC|ours"] = doc_stats(ours, lines, adj_label)

    def total(keys):
        es = [out["docs"][k] for k in keys]
        t = {"segments": sum(e["segments"] for e in es), "major": sum(e["major"] for e in es),
             "major_in_top10pct": sum(e["major_in_top10pct"] for e in es),
             "top10pct_n": sum(e["top10pct_n"] for e in es),
             "gemma_flagged": sum(e["gemma_flagged"] for e in es), "gemma_caught": sum(e["gemma_caught"] for e in es),
             "lines": {}}
        for name in lines:
            ls = [e["lines"][name] for e in es]
            t["lines"][name] = {k: sum(x[k] for x in ls) for k in ("flagged", "caught", "false", "unlabelled")}
            t["lines"][name]["flag_rate"] = t["lines"][name]["flagged"] / t["segments"]
        return t

    out["totals"] = {
        "chosen3_marian": total([f"{d}|marian" for d in CHOSEN]),
        "chosen3_official_es": total([f"{d}|official_es" for d in CHOSEN]),
        "all5_official_es": total([f"{d}|official_es" for d in ALL5]),
        "all5_nllb": total([f"{d}|nllb" for d in ALL5]),
        "other2_marian": total([f"{d}|marian" for d in OTHER_MARIAN]),
    }

    # ------------------------------------------------------------ print
    names = list(lines)
    hdr = " | ".join(f"{n} {lines[n]}: flagged / caught / false / unlabelled / rate" for n in names)
    print(f"adapter {a.adapter}, lines {lines}, scored {a.scored}\n")
    print(f"| key | segs | Major | top10% | {hdr} | Gemma>=0.971 flagged (caught) |")
    for k, e in list(out["docs"].items()) + [(f"TOTAL {k}", v) for k, v in out["totals"].items()]:
        cells = []
        for n in names:
            x = e["lines"][n]
            cells.append(f"{x['flagged']} / {x['caught']} / {x['false']} / {x['unlabelled']} / {x['flag_rate']:.1%}")
        print(f"| {k} | {e['segments']} | {e['major']} | {e['major_in_top10pct']} of {e['top10pct_n']} | "
              + " | ".join(cells) + f" | {e['gemma_flagged']} ({e['gemma_caught']}) |")
    print()
    for k, e in out["docs"].items():
        if k.endswith("|marian") and k.split("|")[0] in CHOSEN or k.endswith("|official_es") or "MSLC" in k:
            mj = ", ".join(f"{m['id'].split('-')[-1]}:{m['risk']:.3f}" for m in sorted(e["majors"], key=lambda m: -m["risk"]))
            print(f"{k}: majors (risk) {mj}")
            for n in names:
                fl = [f"{f['id'].split('-')[-1]}:{f['risk']:.3f}:{f['label']}/{f['verdict']}" for f in e["lines"][n]["flags"]
                      if f["label"] != "Major"]
                print(f"   {n}: non-Major flags {fl}")
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    for e in out["docs"].values():  # keep the json small: no ids of OK segments outside the flag lists
        e["majors"] = sorted(e["majors"], key=lambda m: -m["risk"])
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
