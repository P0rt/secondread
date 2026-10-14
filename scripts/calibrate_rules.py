"""Measure the rule checks on the train splits; write secondread/rule_calibration.json.

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/calibrate_rules.py
           [--method logodds_rules] [--out secondread/rule_calibration.json]
           [--candidates-dir runs/rulecal --splits dev [test]]

Everything that sets a number in the output comes from `data/splits/{lp}.train.jsonl` (and, for
the candidates that need model scores, the local LoRA v1 scores of the same rows,
`results/local_v1_{enru,enes}_train.jsonl`). Dev and test are only read with --candidates-dir.

1. **Rule statistics.** `number_check` and `negation_check` run on every train unit, exactly as
   in the server. Per pair and issue code: hits, Major among them, distinct source sentences
   and documents, the raw P(Major | hit) with a 95% document-bootstrap CI (1,000 replicates),
   and P(Major | no hit).
2. **Weights.** A weighted logistic regression per pair, fitted by Newton's method (no
   randomness). Each code is one 0/1 feature with a N(0, 1) prior on its coefficient, so a
   code seen a few times stays near 0. Clustering: one source sentence is translated by up to
   16 MT systems, and a finding often fires on most of them. So each hit of a code is weighted
   by (distinct source sentences of the code) / (hits of the code): the hits keep their
   row-level Major rate but count as only as many observations as there are distinct source
   sentences behind them. Rows without a hit have weight 1; they mainly fix the intercept. A
   negative coefficient is clamped to 0 (a finding never lowers the risk). The "*" entry is the same fit on both pairs pooled,
   with one intercept per pair; it serves pairs that have no entry of their own.
3. **Candidates** (`--method` picks the one written as `pairs`; all are kept under
   `candidates` for the record):
   - `logodds_rules`: features = codes only. The weight is the log odds ratio of Major with vs
     without the finding, i.e. the finding's likelihood ratio, applied on top of the model's
     log-odds. Needs no model scores. It ignores that the model sees the same evidence, so it
     tends to over-count. **Shipped**: chosen on dev by a rule fixed beforehand (RESULTS.md,
     "Rule calibration").
   - `logodds_offset`: the same features plus logit(v1 risk) as a fixed offset. The weight is
     the correction the model needs on rows where the finding fires. In-sample: v1 was trained
     on these rows, so its train scores are sharper than on unseen rows.
   - `logreg`: intercept, slope on logit(v1 risk) and the codes, all fitted (the slope and
     intercept rescale the model risk too).
   - `noisy_or_train`: the old noisy-OR with each confidence set to the train P(Major | hit)
     (smoothed, from the `logodds_rules` fit).
   The run-time form (`secondread.rulecal`) is logit(risk) = logit(model) + the strongest weight
   of each check; `logreg` is only compared, not shipped.
4. `--candidates-dir`: writes each candidate's combined scores on the requested splits (from
   `results/local_v1_*_{split}.jsonl`), plus `model` (v1 alone) and `handset` (the noisy-OR with
   the hand-set confidences), as scored jsonl for `scripts/report.py`.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from secondread import checks, rulecal
from secondread import metrics as M

ROOT = Path(__file__).resolve().parents[1]
LPS = {"en-ru": "enru", "en-es": "enes"}
PRIOR_SD = 1.0
N_BOOT = 1000
METHODS = ("logodds_rules", "logodds_offset", "logreg", "noisy_or_train")
SHIPPED = "logodds_rules"  # chosen on dev (RESULTS.md, "Rule calibration")


# ---------------------------------------------------------------- data
def run_checks(u: dict) -> list[checks.CheckResult]:
    return [checks.number_check(u["src"], u["mt"], u["lp"]), checks.negation_check(u["src"], u["mt"], u["lp"])]


def load_split(lp: str, part: str, scores: str | None = None) -> list[dict]:
    """Split rows with their rule results (and model risk, from a scored file, if given)."""
    rows = M.load_scored(ROOT / f"data/splits/{lp}.{part}.jsonl")
    risk = {r["id"]: r["risk"] for r in M.load_scored(ROOT / scores)} if scores else None
    out = []
    for u in rows:
        res = run_checks(u)
        out.append({"id": u["id"], "lp": lp, "doc": u["doc"], "src": u["src"], "label": u["label"],
                    "y": int(u["label"] == M.POSITIVE), "rules": res,
                    "codes": sorted({i.code for c in res for i in c.issues}),
                    "model": None if risk is None else risk[u["id"]]})
    if risk is not None:
        assert len(risk) == len(rows), f"{scores}: {len(risk)} scores for {len(rows)} rows"
    return out


def row_weights(rows: list[dict]) -> np.ndarray:
    """Design-effect weights: 1 for a row without a hit; a hit of code c weighs
    (distinct source sentences of c) / (hits of c), so that the hits of c together count as many
    observations as there are distinct source sentences behind them, while the estimate itself
    stays the row-level rate. A row with several codes takes the smallest factor."""
    hits = collections.Counter(c for r in rows for c in r["codes"])
    srcs = {c: len({r["src"] for r in rows if c in r["codes"]}) for c in hits}
    return np.array([min((srcs[c] / hits[c] for c in r["codes"]), default=1.0) for r in rows])


# ---------------------------------------------------------------- fit
def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, prior_prec: np.ndarray,
                 offset: np.ndarray | None = None) -> np.ndarray:
    """MAP weighted logistic regression, Gaussian prior N(0, 1/prior_prec) per coefficient
    (prior_prec 0 = flat). Newton-Raphson from 0; deterministic."""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(X @ beta + off)))
        g = X.T @ (w * (y - p)) - prior_prec * beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(prior_prec)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    raise RuntimeError("logistic fit did not converge")


def design(rows: list[dict], codes: list[str], lps: list[str], model: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Columns: one intercept per pair, [slope on logit(model)], one per code. Returns X, prior_prec, names."""
    cols, prec, names = [], [], []
    for lp in lps:
        cols.append([float(r["lp"] == lp) for r in rows]); prec.append(0.0); names.append(f"intercept:{lp}")
    if model == "slope":
        cols.append([rulecal.logit(r["model"]) for r in rows]); prec.append(0.0); names.append("slope")
    for c in codes:
        cols.append([float(c in r["codes"]) for r in rows]); prec.append(1.0 / PRIOR_SD ** 2); names.append(c)
    return np.array(cols).T, np.array(prec), names


def fit(rows: list[dict], codes: list[str], kind: str) -> dict[str, float]:
    """kind: 'rules' (codes only), 'offset' (codes + logit(model) as offset), 'slope' (logreg)."""
    lps = sorted({r["lp"] for r in rows})
    X, prec, names = design(rows, codes, lps, kind)
    y = np.array([r["y"] for r in rows], dtype=float)
    off = np.array([rulecal.logit(r["model"]) for r in rows]) if kind == "offset" else None
    return dict(zip(names, (float(b) for b in fit_logistic(X, y, row_weights(rows), prec, off)), strict=True))


# ---------------------------------------------------------------- stats
def code_stats(rows: list[dict], code: str, boot: np.ndarray) -> dict:
    hit = np.array([code in r["codes"] for r in rows])
    y = np.array([r["y"] for r in rows])
    n, k = int(hit.sum()), int((hit * y).sum())
    num, den = boot @ (hit * y), boot @ hit
    reps = num[den > 0] / den[den > 0]
    lo, hi = M.percentile_ci(reps)
    return {"hits": n, "major": k, "sources": len({r["src"] for r, h in zip(rows, hit) if h}),
            "docs": len({r["doc"] for r, h in zip(rows, hit) if h}),
            "p_major_raw": round(k / n, 4) if n else None, "p_major_raw_ci": [round(lo, 4), round(hi, 4)]}


def pair_stats(rows: list[dict]) -> dict:
    y = np.array([r["y"] for r in rows])
    hit = np.array([bool(r["codes"]) for r in rows])
    out = {"n": len(rows), "docs": len({r["doc"] for r in rows}), "sources": len({r["src"] for r in rows}),
           "major_rate": round(float(y.mean()), 4), "rows_with_hit": int(hit.sum()),
           "p_major_hit_any": round(float(y[hit].mean()), 4), "p_major_no_hit": round(float(y[~hit].mean()), 4)}
    if rows[0]["model"] is not None:  # in-sample v1 diagnostics
        s = np.array([r["model"] for r in rows])
        out["v1_train_auroc"] = round(M.auroc(y, s), 4)
        out["v1_mean_risk_hit"] = round(float(s[hit].mean()), 4)
        out["v1_mean_risk_no_hit"] = round(float(s[~hit].mean()), 4)
    return out


# ---------------------------------------------------------------- candidates
def combined(method: str, params: dict, r: dict) -> float:
    """Candidate scores of one row (rows carry `rules` and `model`)."""
    lp, m = r["lp"], r["model"]
    if method == "model":
        return m
    if method == "handset":
        return rulecal.combine_noisy_or(m, r["rules"], lambda i: i.confidence)
    def entry(code: str) -> dict:  # this pair's entry, else the pooled one (as secondread.rulecal)
        return params["pairs"].get(lp, {}).get(code) or params["pairs"]["*"].get(code) or {}
    if method == "noisy_or_train":
        return rulecal.combine_noisy_or(m, r["rules"], lambda i: entry(i.code).get("p_major", 0.0))
    if method == "logreg":
        e = params["logreg"][lp]
        bump = sum(rulecal.strongest(r["rules"], lambda i: max(0.0, e["codes"].get(i.code, 0.0))))
        return rulecal.sigmoid(e["intercept"] + e["slope"] * rulecal.logit(m) + bump)
    return rulecal.combine_logodds(m, r["rules"], lambda i: max(0.0, entry(i.code).get("weight", 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", default=SHIPPED, choices=[m for m in METHODS if m != "logreg"])
    ap.add_argument("--out", default="secondread/rule_calibration.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--candidates-dir", help="write each candidate's combined scores here (scored jsonl)")
    ap.add_argument("--splits", nargs="*", default=["dev"], choices=["dev", "test"])
    a = ap.parse_args()

    train = {}
    for lp, tag in LPS.items():
        f = Path(a.results) / f"local_v1_{tag}_train.jsonl"
        train[lp] = load_split(lp, "train", str(f) if (ROOT / f).exists() else None)
    have_model = all(rows[0]["model"] is not None for rows in train.values())
    if not have_model:
        print("no v1 train scores: only the model-free candidates are fitted")
    pooled = [r for rows in train.values() for r in rows]

    fits: dict[str, dict[str, dict[str, float]]] = collections.defaultdict(dict)
    for key, rows in [*train.items(), ("*", pooled)]:
        codes = sorted({c for r in rows for c in r["codes"]})
        fits["rules"][key] = fit(rows, codes, "rules")
        if have_model:
            fits["offset"][key] = fit(rows, codes, "offset")
            if key != "*":
                fits["slope"][key] = fit(rows, codes, "slope")

    rng_boot = {key: M.bootstrap_weights([r["doc"] for r in rows], N_BOOT, seed=0)
                for key, rows in [*train.items(), ("*", pooled)]}
    pairs_stats, cands = {}, {m: {"pairs": {}} for m in METHODS}
    for key, rows in [*train.items(), ("*", pooled)]:
        st = pair_stats(rows) if key != "*" else {"n": len(rows), "note": "both pairs pooled, one intercept per pair"}
        st["codes"] = {}
        rf, of = fits["rules"][key], fits["offset"].get(key)
        icpt = {k: v for k, v in rf.items() if k.startswith("intercept:")}
        for c in sorted({c for r in rows for c in r["codes"]}):
            cs = code_stats(rows, c, rng_boot[key])
            # smoothed P(Major | hit): the rules-only fit at this pair's intercept (pooled: mean intercept)
            b0 = icpt.get(f"intercept:{key}", float(np.mean(list(icpt.values()))))
            cs["p_major"] = round(rulecal.sigmoid(b0 + rf[c]), 4)
            cs["coef_rules"] = round(rf[c], 4)
            if of is not None:
                cs["coef_offset"] = round(of[c], 4)
            st["codes"][c] = cs
            for m in ("logodds_rules", "noisy_or_train"):
                cands[m]["pairs"].setdefault(key, {})[c] = {"weight": round(max(0.0, rf[c]), 4), "p_major": cs["p_major"]}
            if of is not None:
                cands["logodds_offset"]["pairs"].setdefault(key, {})[c] = {"weight": round(max(0.0, of[c]), 4),
                                                                           "p_major": cs["p_major"]}
        pairs_stats[key] = st
    if have_model:
        cands["logreg"] = {"pairs": cands["logodds_rules"]["pairs"], "logreg": {
            lp: {"intercept": f[f"intercept:{lp}"], "slope": f["slope"],
                 "codes": {c: v for c, v in f.items() if not c.startswith(("intercept", "slope"))}}
            for lp, f in fits["slope"].items()}}

    # ---- report
    for key, st in pairs_stats.items():
        print(f"== {key}: " + ", ".join(f"{k} {v}" for k, v in st.items() if k != "codes"))
        for c, cs in st["codes"].items():
            print(f"   {c:17s} hits {cs['hits']:4d} major {cs['major']:3d} src {cs['sources']:3d} docs {cs['docs']:3d} "
                  f"P raw {cs['p_major_raw']:.3f} [{cs['p_major_raw_ci'][0]:.3f}, {cs['p_major_raw_ci'][1]:.3f}] "
                  f"smoothed {cs['p_major']:.3f} | coef rules {cs['coef_rules']:+.3f}"
                  + (f" offset {cs['coef_offset']:+.3f}" if "coef_offset" in cs else ""))
    for lp, f in fits["slope"].items():
        print(f"   logreg {lp}: " + ", ".join(f"{k} {v:+.3f}" for k, v in f.items()))

    if a.method not in cands or not cands[a.method]["pairs"]:
        raise SystemExit(f"method {a.method} needs the v1 train scores")
    weights_of = {m: {key: {c: e["weight"] for c, e in codes.items()} for key, codes in c_["pairs"].items()}
                  for m, c_ in cands.items() if m in ("logodds_rules", "logodds_offset")}
    out = {
        "method_id": "logodds",
        "fitted_as": a.method,
        "method": "logit(risk) = logit(model risk) + for each rule check, the largest weight among its findings "
                  "(secondread/rulecal.py). Weights come from a weighted logistic regression on the train split only "
                  f"(prior N(0, {PRIOR_SD:g}) per code; each hit of a code weighs its distinct source sentences / its "
                  "hits, since one source is translated by up to 16 MT systems), clamped at 0 so a finding never "
                  "lowers the risk. p_major is the smoothed train P(Major | finding), used alone when no model "
                  "score is available; scripts/calibrate_rules.py",
        "generated": time.strftime("%Y-%m-%d"),
        "data": {lp: f"data/splits/{lp}.train.jsonl" for lp in LPS}
                | ({"model_scores_for_candidates": [f"{a.results}/local_v1_{t}_train.jsonl" for t in LPS.values()]}
                   if have_model else {}),
        "pairs": {key: {"codes": codes} for key, codes in cands[a.method]["pairs"].items()},
        "candidates": weights_of,
        "train_stats": pairs_stats,
    }
    path = ROOT / a.out
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", a.out, f"({a.method})")

    if a.candidates_dir:
        d = ROOT / a.candidates_dir
        d.mkdir(parents=True, exist_ok=True)
        names = ["model", "handset", *[m for m in METHODS if cands[m]["pairs"]]]
        for part in a.splits:
            for lp, tag in LPS.items():
                rows = load_split(lp, part, f"{a.results}/local_v1_{tag}_{part}.jsonl")
                for m in names:
                    with open(d / f"{m}_{tag}_{part}.jsonl", "w") as f:
                        f.writelines(json.dumps({"id": r["id"], "label": r["label"], "risk": combined(m, cands.get(m), r),
                                                 "doc": r["doc"], "lp": lp, "rule_hit": bool(r["codes"])}) + "\n"
                                     for r in rows)
                print(f"wrote {len(names)} candidate files for {lp} {part} -> {d.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
