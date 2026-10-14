"""Ship-phase test run: the final candidate on both test splits at both operating points, vs Gemma 4 31B direct.

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python scripts/final_test.py [--systems v1] \
           [--out runs/ship_test/test.json]

The v3 dev selection (docs/RESULTS.md, "v3 (wrong-sense synthetic) — dev selection") kept v1, so v1 is
the only final candidate and the default. This script scores nothing itself: it reads the existing
score files after checking that they cover exactly the rows (ids and gold labels) of the current split
files, and it never picks anything on test.

Inputs
  results/local_<sys>_{enru,enes}_test.jsonl   probe_local.py output of the adapter (model risk)
  results/gemma_direct_{enru,enes}_test.jsonl  Gemma 4 31B direct (scripts/baseline_gemma.py)
  results/gemma_direct_enru_dev.jsonl          Gemma's only dev run: its thresholds come from here
  secondread/thresholds.json                   the adapter's `review` and `f1` thresholds (dev only)

Systems per pair
  <sys> app    the score the backend ranks and flags by: model log-odds plus the train-calibrated rule
               weights (scripts/pick_thresholds.fuse = secondread.agent.aggregate)
  <sys> model  the adapter alone (continuity with the main table in docs/RESULTS.md)
  gemma        Gemma direct risk, unfused. Thresholds by the same two rules on Gemma's en-ru dev scores:
               f1 = metrics.best_f1_threshold (0.971, as in the main table), review = the highest
               threshold with dev recall >= 0.60 (pick_thresholds.recall_threshold). There is no Gemma
               en-es dev run, so both en-ru dev values are used for en-es too.

Metrics: AUROC, PR-AUC, recall@10% (threshold free), and at each operating point recall, precision,
flag rate and F1 at the fixed threshold. 95% CIs: 1,000 document-bootstrap replicates, seed 0, over the
rows sorted by id, grouped by (lp, doc): the same replicates as scripts/report.py, so the model-only
ranking numbers reproduce the main table. One weight matrix per pair is shared by all systems, so every
difference is paired.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from secondread import metrics as M  # noqa: E402
from pick_thresholds import fuse, recall_threshold  # noqa: E402

LPS = {"en-ru": "enru", "en-es": "enes"}
POINTS = ("review", "f1")
N_BOOT, SEED, TARGET = 1000, 0, 0.60
RANK = {"auroc": M.auroc, "pr_auc": M.average_precision, "recall_at_10": lambda y, s, w=None: M.recall_at(y, s, 0.10, w)}
AT = ("recall", "precision", "flag_rate", "f1")


def load_checked(path: Path, split: dict[str, dict]) -> list[dict]:
    rows = M.load_scored(str(path))
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == len(split) and set(ids) == set(split), f"{path}: not the rows of the split"
    assert all(r["label"] == split[r["id"]]["label"] for r in rows), f"{path}: gold labels differ from the split"
    return sorted(rows, key=lambda r: r["id"])


def ci(vals) -> list[float]:
    lo, hi = M.percentile_ci(v for v in vals if np.isfinite(v))
    return [lo, hi]


def evaluate(y, s, thr: dict[str, float], W) -> dict:
    out = {k: {"value": f(y, s), "ci": ci(f(y, s, w=w) for w in W)} for k, f in RANK.items()}
    for p, t in thr.items():
        prf = M.prf_at(y, s, t)
        boot = [M.prf_at(y, s, t, w) for w in W]
        flag = s >= t
        out[p] = {"threshold": t, "flagged": int(flag.sum()), "tp": int((flag & (y == 1)).sum()),
                  **{k: {"value": prf[k], "ci": ci(b[k] for b in boot)} for k in AT}}
    return out


def delta(ya, sa, ta, yb, sb, tb, W) -> dict:
    """A - B on the shared replicates; threshold metrics at each system's own threshold."""
    out = {}
    for k, f in RANK.items():
        out[k] = {"value": f(ya, sa) - f(yb, sb), "ci": ci(f(ya, sa, w=w) - f(yb, sb, w=w) for w in W)}
    for p in POINTS:
        if p in ta and p in tb:
            pa, pb = M.prf_at(ya, sa, ta[p]), M.prf_at(yb, sb, tb[p])
            ba = [M.prf_at(ya, sa, ta[p], w) for w in W]
            bb = [M.prf_at(yb, sb, tb[p], w) for w in W]
            out[p] = {k: {"value": pa[k] - pb[k], "ci": ci(x[k] - z[k] for x, z in zip(ba, bb))} for k in AT}
    return out


def f3(x: dict, signed: bool = False, pct: bool = False) -> str:
    v, (lo, hi) = x["value"], x["ci"]
    if pct:
        f = (lambda z: f"{z * 100:+.1f}") if signed else (lambda z: f"{z * 100:.1f}%")
        return f"{f(v)} [{f(lo)}, {f(hi)}]" + (" pt" if signed else "")
    f = (lambda z: f"{z:+.3f}") if signed else (lambda z: f"{z:.3f}")
    return f"{f(v)} [{f(lo)}, {f(hi)}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", nargs="+", default=["v1"], help="final candidates (adapters in thresholds.json)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="runs/ship_test/test.json")
    a = ap.parse_args()
    R = ROOT / a.results
    thr_file = json.loads((ROOT / "secondread/thresholds.json").read_text())

    # Gemma thresholds: dev only (en-ru dev; there is no Gemma en-es dev run)
    gdev_split = {r["id"]: r for r in M.load_scored(str(ROOT / "data/splits/en-ru.dev.jsonl"))}
    gy, gs = M.to_arrays(load_checked(R / "gemma_direct_enru_dev.jsonl", gdev_split))
    g_thr = {"review": recall_threshold(gy, gs, TARGET)[0], "f1": M.best_f1_threshold(gy, gs)}

    out: dict = {"n_boot": N_BOOT, "seed": SEED, "gemma_thresholds": g_thr, "pairs": {}}
    for lp, tag in LPS.items():
        split = {r["id"]: r for r in M.load_scored(str(ROOT / f"data/splits/{lp}.test.jsonl"))}
        gem = load_checked(R / f"gemma_direct_{tag}_test.jsonl", split)
        W = M.bootstrap_weights(M.group_keys(gem), N_BOOT, SEED)
        y0, gsc = M.to_arrays(gem)
        blk = out["pairs"][lp] = {"n": len(gem), "n_docs": len(set(M.group_keys(gem))), "majors": int(y0.sum()),
                                  "systems": {}, "deltas": {}}
        blk["systems"]["gemma"] = evaluate(y0, gsc, g_thr, W)
        arrays = {"gemma": (y0, gsc, g_thr)}
        for name in a.systems:
            e = thr_file["adapters"][name]
            thr = {p: e[p]["thresholds"][lp] for p in POINTS if p in e}
            fused = fuse(split, load_checked(R / f"local_{name}_{tag}_test.jsonl", split))
            assert [r["id"] for r in fused] == [r["id"] for r in gem]
            ya, sa = M.to_arrays(fused)
            ym, sm = M.to_arrays([{**r, "risk": r["model_risk"]} for r in fused])
            assert (ya == y0).all() and (ym == y0).all()
            blk["systems"][f"{name} app"] = evaluate(ya, sa, thr, W)
            blk["systems"][f"{name} app"]["rule_hit_rows"] = int(sum(r["rule_hit"] for r in fused))
            blk["systems"][f"{name} model"] = evaluate(ym, sm, {}, W)
            arrays[f"{name} app"], arrays[f"{name} model"] = (ya, sa, thr), (ym, sm, {})
            blk["deltas"][f"{name} app - gemma"] = delta(ya, sa, thr, y0, gsc, g_thr, W)
            blk["deltas"][f"{name} model - gemma"] = delta(ym, sm, {}, y0, gsc, g_thr, W)
            blk["deltas"][f"{name} app - {name} model"] = delta(ya, sa, {}, ym, sm, {}, W)
            # what the review line buys against the F1 line, same scores, paired
            if "review" in thr and "f1" in thr:
                pr, pf = M.prf_at(ya, sa, thr["review"]), M.prf_at(ya, sa, thr["f1"])
                br = [M.prf_at(ya, sa, thr["review"], w) for w in W]
                bf = [M.prf_at(ya, sa, thr["f1"], w) for w in W]
                blk["deltas"][f"{name} app: review - f1"] = {
                    k: {"value": pr[k] - pf[k], "ci": ci(x[k] - z[k] for x, z in zip(br, bf))} for k in AT}

    # ------------------------------------------------------------------ markdown
    lines = [f"Gemma thresholds (en-ru dev): review {g_thr['review']:.4f}, f1 {g_thr['f1']:.4f}", ""]
    for lp, blk in out["pairs"].items():
        lines += [f"### {lp} test: {blk['n']} rows, {blk['n_docs']} docs, {blk['majors']} Major", "",
                  "| System | AUROC | PR-AUC | recall@10% |", "|---|---|---|---|"]
        for name, e in blk["systems"].items():
            lines.append(f"| {name} | {f3(e['auroc'])} | {f3(e['pr_auc'])} | {f3(e['recall_at_10'])} |")
        lines += ["", "| System | Point | Threshold | Flagged (Major) | Recall | Precision | Flag rate | F1 |",
                  "|---|---|---:|---:|---|---|---|---|"]
        for name, e in blk["systems"].items():
            for p in POINTS:
                if p in e:
                    x = e[p]
                    lines.append(f"| {name} | {p} | {x['threshold']:.4f} | {x['flagged']} ({x['tp']}) | {f3(x['recall'])} | "
                                 f"{f3(x['precision'])} | {f3(x['flag_rate'], pct=True)} | {f3(x['f1'])} |")
        lines += ["", "| A − B | ΔAUROC | ΔPR-AUC | Δrecall@10% | review: Δrecall / Δprecision / Δflag rate / ΔF1 | "
                      "f1: Δrecall / Δprecision / Δflag rate / ΔF1 |", "|---|---|---|---|---|---|"]
        for name, d in blk["deltas"].items():
            if "auroc" not in d:
                continue
            cells = [f3(d[k], signed=True) for k in RANK]
            for p in POINTS:
                cells.append(" / ".join(f3(d[p][k], signed=True, pct=k == "flag_rate") for k in AT) if p in d else "n/a")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        for name, d in blk["deltas"].items():
            if "auroc" in d:
                continue
            lines.append("")
            lines.append(f"{name} (paired): " + ", ".join(
                f"Δ{k} {f3(d[k], signed=True, pct=k == 'flag_rate')}" for k in AT))
        lines.append("")
    print("\n".join(lines))
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
