"""Error analysis of the frozen gold-set predictions (after scripts/gold_score.py has scored them).

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python -I scripts/gold_analyze.py
Reads runs/gold_eval/preds_*.jsonl (frozen, PREDICTIONS_MANIFEST.json), data/realdocs/gold/labels.jsonl and
runs/gold_eval/report_primary.json; writes runs/gold_eval/analysis.json and prints the tables of the
"Scores on the gold set" section of docs/REALDOCS_GOLD.md. Nothing is re-scored or tuned here: every
threshold is the one fixed before scoring, and per-document numbers are point values (a document-cluster
bootstrap has no CI inside one document).
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from secondread.metrics import auroc, average_precision, recall_at  # noqa: E402

D = ROOT / "runs" / "gold_eval"
GOLD = ROOT / "data" / "realdocs" / "gold"
# (system name in the report, prediction file, threshold fixed before scoring or None)
SYSTEMS = [("v1_app_review", "preds_v1_app.jsonl", 0.4984), ("v1_app_f1", "preds_v1_app.jsonl", 0.7019),
           ("v1_model", "preds_v1_model.jsonl", None), ("base_app_review", "preds_base_app.jsonl", 0.6183),
           ("gemma_review", "preds_gemma_direct.jsonl", 0.1212), ("gemma_f1", "preds_gemma_direct.jsonl", 0.9714)]
RANKERS = {"v1_app": "preds_v1_app.jsonl", "v1_model": "preds_v1_model.jsonl", "base_app": "preds_base_app.jsonl",
           "gemma": "preds_gemma_direct.jsonl"}
LINES = [(n, f, t) for n, f, t in SYSTEMS if t is not None]


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def _np(o):
    """numpy scalars in json.dumps."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__)


def check_manifest() -> None:
    man = json.loads((D / "PREDICTIONS_MANIFEST.json").read_text())
    for name, e in man["files"].items():
        if hashlib.sha256((D / name).read_bytes()).hexdigest() != e["sha256"]:
            sys.exit(f"{name} changed after it was frozen")


def point_metrics(y: np.ndarray, s: np.ndarray, t: float | None) -> dict:
    out = {"auroc": auroc(y, s) if 0 < y.sum() < len(y) else None,
           "pr_auc": average_precision(y, s) if y.sum() else None,
           "recall@10%": recall_at(y, s, 0.10) if y.sum() else None}
    if t is not None:
        f = s >= t
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        neg = int((y == 0).sum())
        out.update(flags=int(f.sum()), tp=tp, fp=fp, recall=tp / y.sum() if y.sum() else None,
                   precision=tp / f.sum() if f.sum() else None, flag_rate=float(f.mean()),
                   false_flag_rate_on_OK=fp / neg if neg else None)
    return out


def main() -> None:
    check_manifest()
    gold = sorted(jl(GOLD / "labels.jsonl"), key=lambda g: g["uid"])
    uid = [g["uid"] for g in gold]
    y = np.array([g["label"] == "Major" for g in gold], dtype=np.int64)
    risk = {}
    for name, f in RANKERS.items():
        p = {r["uid"]: r["risk"] for r in jl(D / f)}
        risk[name] = np.array([p[u] for u in uid], dtype=np.float64)
    sys_risk = {n: risk[{"preds_v1_app.jsonl": "v1_app", "preds_v1_model.jsonl": "v1_model",
                         "preds_base_app.jsonl": "base_app", "preds_gemma_direct.jsonl": "gemma"}[f]]
                for n, f, _ in SYSTEMS}
    thr = {n: t for n, _, t in SYSTEMS}
    out: dict = {"labels_sha256": hashlib.sha256((GOLD / "labels.jsonl").read_bytes()).hexdigest(),
                 "thresholds": thr}

    # ---- the riskiest 10% of all 1,440 units (144) per ranker: membership, for the category table
    k = round(len(uid) * 0.10)
    top = {}
    for name, s in risk.items():
        order = np.argsort(-s, kind="stable")
        cut = s[order[k - 1]]
        if (s == cut).sum() > 1:
            print(f"note: {name} has a tie at the top-10% cut ({(s == cut).sum()} units)")
        top[name] = set(np.array(uid)[order[:k]])
    out["top10_cut"] = {n: float(np.sort(s)[::-1][k - 1]) for n, s in risk.items()}

    # ---- per document (point values)
    per_doc = {}
    docs = sorted({g["doc_id"] for g in gold})
    for d in docs:
        idx = np.array([g["doc_id"] == d for g in gold])
        e = {"n": int(idx.sum()), "major": int(y[idx].sum()),
             "major_marian": int(sum(g["label"] == "Major" and g["system"] == "marian" for g in gold if g["doc_id"] == d)),
             "major_deepseek": int(sum(g["label"] == "Major" and g["system"] == "deepseek-v3.1"
                                       for g in gold if g["doc_id"] == d)), "systems": {}}
        for n, _, t in SYSTEMS:
            e["systems"][n] = point_metrics(y[idx], sys_risk[n][idx], t)
        per_doc[d] = e
    out["per_doc"] = per_doc

    # ---- Majors by category / severity / MT system: caught at each line and in the riskiest 10%
    def caught_table(key) -> dict:
        tab: dict = collections.OrderedDict()
        for i, g in enumerate(gold):
            if g["label"] != "Major":
                continue
            c = key(g)
            row = tab.setdefault(c, {"n": 0, **{n: 0 for n, _, _ in LINES}, **{f"top10_{r}": 0 for r in RANKERS}})
            row["n"] += 1
            for n, _, t in LINES:
                row[n] += int(sys_risk[n][i] >= t)
            for r in RANKERS:
                row[f"top10_{r}"] += int(g["uid"] in top[r])
        return dict(sorted(tab.items(), key=lambda kv: -kv[1]["n"]))
    out["by_category"] = caught_table(lambda g: g["category"])
    out["by_severity"] = caught_table(lambda g: g["severity"])
    out["by_category_marian"] = caught_table(lambda g: (g["system"], g["category"]))
    out["by_category_marian"] = {f"{a}|{b}": v for (a, b), v in out["by_category_marian"].items()}

    # ---- unit lists: v1 flags, critical units, DeepSeek Majors, v1's misses
    def unit(i: int) -> dict:
        g = gold[i]
        return {"uid": g["uid"], "doc_id": g["doc_id"], "seg_index": g["seg_index"], "system": g["system"],
                "label": g["label"], "severity": g["severity"], "category": g["category"], "span_mt": g["span_mt"],
                "n_major": g.get("n_major"), "risk": {r: round(float(risk[r][i]), 4) for r in RANKERS}}
    v1_flag = [unit(i) for i in np.argsort(-risk["v1_app"]) if risk["v1_app"][i] >= thr["v1_app_review"]]
    out["v1_review_flags"] = v1_flag
    out["critical"] = [unit(i) for i, g in enumerate(gold) if g["severity"] == "critical"]
    ds = np.array([g["system"] == "deepseek-v3.1" for g in gold])
    ds_rank = {}
    for r in RANKERS:
        s = risk[r]
        order = [i for i in np.argsort(-s, kind="stable") if ds[i]]
        ds_rank[r] = {gold[i]["uid"]: pos + 1 for pos, i in enumerate(order)}
    out["deepseek_majors"] = [unit(i) | {"rank_in_deepseek_720": {r: ds_rank[r][g["uid"]] for r in RANKERS}}
                              for i, g in enumerate(gold) if ds[i] and g["label"] == "Major"]
    maj = y == 1
    out["v1_risk_on_majors"] = {
        "quantiles": {q: round(float(np.quantile(risk["v1_app"][maj], q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "bands": {b: int(((risk["v1_app"] >= lo) & (risk["v1_app"] < hi) & maj).sum())
                  for b, (lo, hi) in {"<0.1": (0, 0.1), "0.1-0.3": (0.1, 0.3), "0.3-0.4984": (0.3, 0.4984),
                                      "0.4984-0.7019": (0.4984, 0.7019), ">=0.7019": (0.7019, 1.01)}.items()}}
    out["v1_risk_on_ok"] = {q: round(float(np.quantile(risk["v1_app"][~maj], q)), 4) for q in (0.5, 0.9, 0.99)}
    out["gemma_caught_v1_missed_review"] = int(((risk["gemma"] >= 0.1212) & (risk["v1_app"] < 0.4984) & maj).sum())
    out["v1_caught_gemma_missed_review"] = int(((risk["gemma"] < 0.1212) & (risk["v1_app"] >= 0.4984) & maj).sum())

    # ---- rule findings (the app score minus the model)
    app = {r["uid"]: r for r in jl(D / "preds_v1_app.jsonl")}
    hits = [i for i, u in enumerate(uid) if app[u]["rule_codes"]]
    out["rule_findings"] = {
        "units": len(hits), "major": int(y[hits].sum()),
        "codes": dict(collections.Counter(c for i in hits for c in set(app[uid[i]]["rule_codes"]))),
        "flag_changed_review": int(sum((risk["v1_app"][i] >= 0.4984) != (risk["v1_model"][i] >= 0.4984) for i in hits)),
        "flag_changed_f1": int(sum((risk["v1_app"][i] >= 0.7019) != (risk["v1_model"][i] >= 0.7019) for i in hits))}

    # ---- false flags by the gold severity of the OK unit, and AUROC by pair type (descriptive)
    out["false_flags_by_severity"] = {
        n: dict(collections.Counter(f"{g['system']}|{g['severity']}" for i, g in enumerate(gold)
                                    if g["label"] != "Major" and sys_risk[n][i] >= t)) for n, _, t in LINES}
    sysarr = np.array([g["system"] for g in gold])

    def pair_auc(pos: np.ndarray, neg: np.ndarray, s: np.ndarray) -> float:
        a, b = s[pos][:, None], s[neg][None, :]
        return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))
    mm, om, od = maj & (sysarr == "marian"), ~maj & (sysarr == "marian"), ~maj & (sysarr != "marian")
    out["auroc_by_pair_type"] = {r: {"opus_major_vs_opus_ok": pair_auc(mm, om, risk[r]),
                                     "opus_major_vs_deepseek_ok": pair_auc(mm, od, risk[r])} for r in RANKERS}

    # ---- official-Spanish control (ref_es scored as the MT, one row per source segment)
    ctrl = {}
    for name, f in (("v1_app", "preds_control_ref_es_v1_app.jsonl"), ("gemma", "preds_control_ref_es_gemma.jsonl")):
        rows = jl(D / f)
        lines = {"v1_app": {"review": 0.4984, "f1": 0.7019}, "gemma": {"review": 0.1212, "f1": 0.9714}}[name]
        e = {"n": len(rows), "lines": {}}
        for ln, t in lines.items():
            fl = [r for r in rows if r["risk"] >= t]
            e["lines"][ln] = {"threshold": t, "flags": len(fl), "flag_rate": len(fl) / len(rows),
                              "per_doc": dict(collections.Counter(r["id"].split("|")[0] for r in fl))}
            if name == "v1_app":
                e["lines"][ln]["flagged"] = [{"id": r["id"], "risk": round(r["risk"], 4)} for r in fl]
        e["rule_hit_segments"] = sum(bool(r.get("rule_codes")) for r in rows) if name == "v1_app" else None
        ctrl[name] = e
    out["control_ref_es"] = ctrl

    # ---- primary vs pre-registered labelling
    rp = json.loads((D / "report_primary.json").read_text())
    rr = json.loads((D / "report_preregistered.json").read_text())
    out["preregistered_identical"] = rp["strata"] == rr["strata"] and rp["per_doc"] == rr["per_doc"]

    (D / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_np) + "\n",
                                     encoding="utf-8")
    print(f"wrote {D / 'analysis.json'}; preregistered identical to primary: {out['preregistered_identical']}")

    # ---------------------------------------------------------------- tables (printed)
    f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
    print("\nPer document, ranking (point values):")
    print("| Document | Units | Major (OPUS / DS) | v1 app AUROC / PR-AUC / r@10% | base AUROC / PR-AUC / r@10% "
          "| Gemma AUROC / PR-AUC / r@10% |")
    for d, e in per_doc.items():
        s = e["systems"]
        cell = {n: " / ".join(f3(s[n][m]) for m in ("auroc", "pr_auc", "recall@10%"))
                for n in ("v1_app_review", "base_app_review", "gemma_review")}
        print(f"| {d} | {e['n']} | {e['major']} ({e['major_marian']} / {e['major_deepseek']}) | {cell['v1_app_review']} "
              f"| {cell['base_app_review']} | {cell['gemma_review']} |")
    print("\nPer document, lines (flags, caught = TP, recall, precision, flag rate, false-flag rate on OK):")
    for d, e in per_doc.items():
        s = e["systems"]
        parts = []
        for n, _, _ in LINES:
            m = s[n]
            parts.append(f"{n}: {m['flags']} flags, {m['tp']}/{e['major']} caught, P {f3(m['precision'])}, "
                         f"flag {m['flag_rate']:.1%}, FFR {m['false_flag_rate_on_OK']:.1%}")
        print(f"- {d}: " + "; ".join(parts))
    for title, tab in (("category", out["by_category"]), ("severity", out["by_severity"]),
                       ("system|category", out["by_category_marian"])):
        print(f"\nMajors by {title}: n, caught at " + ", ".join(n for n, _, _ in LINES) + ", top-10% of 1,440 "
              + ", ".join(RANKERS))
        for c, row in tab.items():
            print(f"| {c} | {row['n']} | " + " | ".join(str(row[n]) for n, _, _ in LINES) + " | "
                  + " | ".join(str(row[f'top10_{r}']) for r in RANKERS) + " |")
    print("\nv1 review-line flags:")
    for u in v1_flag:
        print(f"  {u['uid']} {u['doc_id']} {u['system']} {u['label']} {u['severity']} {u['category']} "
              f"v1 {u['risk']['v1_app']} gemma {u['risk']['gemma']:.3f} base {u['risk']['base_app']}")
    print("\ncritical:", json.dumps(out["critical"], ensure_ascii=False, default=_np))
    print("\nDeepSeek Majors:")
    for u in out["deepseek_majors"]:
        print(f"  {u['uid']} {u['doc_id']} {u['category']} risks {u['risk']} ranks {u['rank_in_deepseek_720']}")
    print("\nv1 risk on Majors:", out["v1_risk_on_majors"], "on OK:", out["v1_risk_on_ok"])
    print("top-10% cut:", out["top10_cut"])
    print("Gemma-only catches at review lines:", out["gemma_caught_v1_missed_review"], "v1-only:",
          out["v1_caught_gemma_missed_review"])
    print("rule findings:", out["rule_findings"])
    print("false flags by severity:", out["false_flags_by_severity"])
    print("AUROC by pair type:", json.dumps(out["auroc_by_pair_type"]))
    print("control:", json.dumps(ctrl, ensure_ascii=False, default=_np))


if __name__ == "__main__":
    main()
