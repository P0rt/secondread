"""Score a Major/OK detector on the real-document gold set (docs/REALDOCS_GOLD.md, "How to score a system").

usage:
  # 1. blind inputs for the system under test (no labels, no MT-system name, no reference by default)
  uv run --frozen python -I scripts/gold_score.py --export-inputs /tmp/gold_inputs.jsonl [--with-ref]
  # 2. score one or more prediction files (one JSON line per uid: {"uid": "g0001", "risk": 0.12})
  uv run --frozen python -I scripts/gold_score.py --pred A=preds_a.jsonl --threshold A=0.4984 \
        [--pred B=preds_b.jsonl --threshold B=0.5] [--labelling primary|preregistered] [--out report.json]
  # 3. check the scorer itself on synthetic predictions (oracle, constant, random); scores no real system
  uv run --frozen python -I scripts/gold_score.py --selftest

Rules, all enforced here:
  * every one of the 1440 gold units must be predicted exactly once; nothing is dropped or added;
  * "risk" is a finite number, higher = more likely Major; a binary system writes 1.0 / 0.0 and uses --threshold 0.5;
  * a threshold must be fixed before scoring (e.g. secondread/thresholds.json, picked on WMT dev); it is never
    chosen on this set, so F1 is only reported at a given threshold;
  * 95% CIs come from a document-level cluster bootstrap (10 documents resampled with replacement; the two MT
    outputs of a segment always stay together); one weight matrix per stratum is shared by all systems, so the
    differences between systems are paired.
Metric definitions are those of secondread/metrics.py (AUROC with ties as 1/2, average precision, recall@10% with
fractional ties, precision/recall/F1 at risk >= threshold).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from secondread.metrics import (auroc, average_precision, bootstrap_weights, percentile_ci, prf_at,  # noqa: E402
                                recall_at)

GOLD = REPO / "data" / "realdocs" / "gold"
LABELS = GOLD / "labels.jsonl"
UNITS = GOLD / "units.jsonl"
INPUT_FIELDS = ["uid", "doc_id", "seg_index", "context_src", "src", "mt"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def export_inputs(path: Path, with_ref: bool) -> None:
    fields = INPUT_FIELDS + (["ref_es"] if with_ref else [])
    rows = sorted(_jsonl(UNITS), key=lambda u: u["uid"])
    with path.open("w", encoding="utf-8") as f:
        for u in rows:
            f.write(json.dumps({k: u[k] for k in fields}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} units to {path} (fields {fields})")


def load_preds(path: Path, gold_uids: set[str]) -> dict[str, float]:
    preds: dict[str, float] = {}
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        uid, risk = r.get("uid"), r.get("risk")
        if uid in preds:
            sys.exit(f"{path}:{i}: uid {uid} predicted twice")
        if uid not in gold_uids:
            sys.exit(f"{path}:{i}: uid {uid!r} is not a gold unit")
        if isinstance(risk, bool) or not isinstance(risk, (int, float)) or not math.isfinite(risk):
            sys.exit(f"{path}:{i}: risk must be a finite number, got {risk!r}")
        preds[uid] = float(risk)
    missing = gold_uids - set(preds)
    if missing:
        sys.exit(f"{path}: {len(missing)} gold units have no prediction, e.g. {sorted(missing)[:5]}")
    return preds


def metric_fns(threshold: float | None, frac: float) -> dict:
    fns = {"auroc": auroc, "pr_auc": average_precision,
           f"recall@{frac:.0%}": lambda y, s, w=None: recall_at(y, s, frac, w)}
    if threshold is not None:
        def _fpr(y, s, w=None):
            w = np.ones(len(y)) if w is None else w
            neg = w * (1 - y)
            return float((neg * (s >= threshold)).sum() / neg.sum()) if neg.sum() else float("nan")
        for k in ("precision", "recall", "f1", "flag_rate"):
            fns[k] = (lambda key: (lambda y, s, w=None: prf_at(y, s, threshold, w)[key]))(k)
        fns["false_flag_rate_on_OK"] = _fpr
    return fns


def score(rows: list[dict], systems: dict[str, dict], thresholds: dict[str, float | None], frac: float,
          n_boot: int, seed: int) -> dict:
    """rows: gold rows of one stratum; systems: name -> {uid: risk}."""
    y = np.fromiter((r["label"] == "Major" for r in rows), dtype=np.int64, count=len(rows))
    W = bootstrap_weights([r["doc_id"] for r in rows], n_boot, seed)
    out = {"n": len(rows), "n_docs": len({r["doc_id"] for r in rows}), "major": int(y.sum()),
           "major_rate": float(y.mean()) if len(y) else None, "systems": {}, "paired": {}}
    per_rep = {}
    for name, preds in systems.items():
        s = np.fromiter((preds[r["uid"]] for r in rows), dtype=np.float64, count=len(rows))
        res = {}
        for m, f in metric_fns(thresholds.get(name), frac).items():
            point = f(y, s)
            reps = np.array([f(y, s, w=w) for w in W])
            per_rep[(name, m)] = reps
            lo, hi = percentile_ci(reps)
            res[m] = {"point": None if math.isnan(point) else point,
                      "ci95": None if math.isnan(lo) else [lo, hi],
                      "n_boot_defined": int((~np.isnan(reps)).sum())}
        t = thresholds.get(name)
        if t is not None:
            flag = s >= t
            res["counts"] = {"threshold": t, "tp": int((flag & (y == 1)).sum()), "fp": int((flag & (y == 0)).sum()),
                             "fn": int((~flag & (y == 1)).sum()), "tn": int((~flag & (y == 0)).sum())}
        out["systems"][name] = res
    names = list(systems)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = {}
            for m in out["systems"][a]:
                if m == "counts" or (b, m) not in per_rep:
                    continue
                d = per_rep[(a, m)] - per_rep[(b, m)]
                pa, pb = out["systems"][a][m]["point"], out["systems"][b][m]["point"]
                lo, hi = percentile_ci(d)
                ok = d[~np.isnan(d)]
                diff[m] = {"point": None if pa is None or pb is None else pa - pb,
                           "ci95": None if math.isnan(lo) else [lo, hi],
                           "share_a_better": float((ok > 0).mean()) if len(ok) else None}
            out["paired"][f"{a} - {b}"] = diff
    return out


def run(gold: list[dict], systems: dict[str, dict], thresholds: dict, labelling: str, frac: float, n_boot: int,
        seed: int) -> dict:
    field = "label" if labelling == "primary" else "label_preregistered"
    rows = [{"uid": g["uid"], "doc_id": g["doc_id"], "system": g["system"], "label": g[field]} for g in gold]
    report = {"gold": {"labels": str(LABELS.relative_to(REPO)), "labelling": labelling, "field": field},
              "bootstrap": {"unit": "document", "n_boot": n_boot, "seed": seed, "ci": "percentile 2.5-97.5"},
              "frac": frac, "thresholds": thresholds, "strata": {}}
    report["strata"]["all"] = score(rows, systems, thresholds, frac, n_boot, seed)
    for mt in sorted({r["system"] for r in rows}):
        report["strata"][f"mt={mt}"] = score([r for r in rows if r["system"] == mt], systems, thresholds, frac,
                                             n_boot, seed)
    per_doc = {}
    for d in sorted({r["doc_id"] for r in rows}):
        rs = [r for r in rows if r["doc_id"] == d]
        entry = {"n": len(rs), "major": sum(r["label"] == "Major" for r in rs), "systems": {}}
        for name, preds in systems.items():
            t = thresholds.get(name)
            if t is not None:
                entry["systems"][name] = {
                    "flags": sum(preds[r["uid"]] >= t for r in rs),
                    "tp": sum(preds[r["uid"]] >= t and r["label"] == "Major" for r in rs)}
        per_doc[d] = entry
    report["per_doc"] = per_doc
    return report


def parse_named(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--{what} wants NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        if k in out:
            sys.exit(f"--{what} {k} given twice")
        out[k] = v
    return out


def selftest(gold: list[dict]) -> None:
    rng = random.Random(0)
    systems = {"oracle": {g["uid"]: 1.0 if g["label"] == "Major" else 0.0 for g in gold},
               "constant": {g["uid"]: 0.5 for g in gold},
               "random": {g["uid"]: rng.random() for g in gold}}
    th = {"oracle": 0.5, "constant": 0.5, "random": 0.9}
    rep = run(gold, systems, th, "primary", 0.10, 200, 0)
    a = rep["strata"]["all"]["systems"]
    assert a["oracle"]["auroc"]["point"] == 1.0 and a["oracle"]["f1"]["point"] == 1.0
    assert a["oracle"]["recall@10%"]["point"] == 1.0  # 84 Majors fit in the top 144 units
    assert a["constant"]["auroc"]["point"] == 0.5 and a["constant"]["flag_rate"]["point"] == 1.0
    assert abs(a["constant"]["recall@10%"]["point"] - 0.10) < 0.01  # fractional tie break
    assert 0.35 < a["random"]["auroc"]["point"] < 0.65
    n_major = rep["strata"]["all"]["major"]
    c = a["oracle"]["counts"]
    assert c["tp"] == n_major and c["fp"] == 0 and c["fn"] == 0
    assert rep["strata"]["all"]["paired"]["oracle - random"]["auroc"]["share_a_better"] == 1.0
    print(f"selftest passed: {len(gold)} units, {n_major} Major; oracle AUROC 1.0, constant AUROC 0.5, "
          f"random AUROC {a['random']['auroc']['point']:.3f} {a['random']['auroc']['ci95']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-inputs", type=Path)
    ap.add_argument("--with-ref", action="store_true", help="include ref_es in the exported inputs")
    ap.add_argument("--pred", action="append", help="NAME=path to a JSONL file of {uid, risk}")
    ap.add_argument("--threshold", action="append", help="NAME=flag threshold fixed before scoring (risk >= t)")
    ap.add_argument("--labelling", choices=["primary", "preregistered"], default="primary")
    ap.add_argument("--frac", type=float, default=0.10, help="review budget for recall@k (default 0.10)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.export_inputs:
        export_inputs(args.export_inputs, args.with_ref)
        return
    gold = _jsonl(LABELS)
    if args.selftest:
        selftest(gold)
        return
    preds_paths = parse_named(args.pred, "pred")
    if not preds_paths:
        ap.error("give --pred NAME=path (or --export-inputs, or --selftest)")
    thresholds = {k: float(v) for k, v in parse_named(args.threshold, "threshold").items()}
    if set(thresholds) - set(preds_paths):
        sys.exit(f"--threshold for unknown system(s) {sorted(set(thresholds) - set(preds_paths))}")
    uids = {g["uid"] for g in gold}
    systems = {name: load_preds(Path(p), uids) for name, p in preds_paths.items()}
    report = run(gold, systems, {k: thresholds.get(k) for k in systems}, args.labelling, args.frac, args.n_boot,
                 args.seed)
    report["gold"]["labels_sha256"] = _sha(LABELS)
    report["pred_files"] = {k: {"path": p, "sha256": _sha(Path(p))} for k, p in preds_paths.items()}
    text = json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    for stratum, st in report["strata"].items():
        print(f"[{stratum}] n={st['n']} docs={st['n_docs']} Major={st['major']}")
        for name, res in st["systems"].items():
            parts = []
            for m in ("auroc", "pr_auc", f"recall@{args.frac:.0%}", "precision", "recall", "f1"):
                if m in res and res[m]["point"] is not None:
                    ci = res[m]["ci95"]
                    parts.append(f"{m} {res[m]['point']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
            print(f"  {name}: " + "; ".join(parts))
    if not args.out:
        print(text)


if __name__ == "__main__":
    main()
