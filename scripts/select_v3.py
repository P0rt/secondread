"""v3 (wrong-sense synthetic) dev selection: v1 against the candidates v3a and v3b.

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python scripts/select_v3.py [--out runs/v3_select/select.json]

Reads only dev splits, the held-out synthetic sense set and that set's clean parents:
  results/local_{v1,v3a,v3b}_{enru,enes}_dev.jsonl        full dev splits (probe_local.py)
  results/local_{v1,v3a,v3b}_synth_sense_test.jsonl       data/synth/sense.test.jsonl (420 rows)
  results/local_{v1,v3a,v3b}_synth_sense_parents.jsonl    data/synth/sense.test.parents.jsonl: the 380
                                                          distinct clean parents (all gold OK) of those rows
It opens no data/splits/*.test.jsonl file and no real-document file.

SELECTION RULE (fixed before any v3a/v3b dev score was computed, 2026-10-14 22:54 CEST):
  1. On each pair's full dev split, compute the MODEL-ONLY AUROC and PR-AUC of v1, v3a and v3b
     (point estimates on all rows).
  2. A candidate is eligible only if, on BOTH pairs, its PR-AUC >= v1's PR-AUC - 0.01 AND its
     AUROC >= v1's AUROC - 0.01.
  3. Among eligible candidates, pick the one with the highest mean of (dPR-AUC en-ru,
     dPR-AUC en-es) against v1.
  4. If no candidate is eligible, or the best mean dPR-AUC is not > 0, keep v1.
  The fused (app) AUROC/PR-AUC are reported next to it and the script says whether the
  decision would differ under them; they do not decide.

Everything else is descriptive and plays no part in the choice:
  * paired dev deltas vs v1 with 95% CIs (1,000 document-bootstrap replicates shared by the
    systems of a pair, seed 0), model-only and fused;
  * each system's REVIEW thresholds by the pre-registered threshold rule (highest threshold
    with dev recall of Major >= 0.60 on the app's fused score, rounded down to 4 decimals;
    `scripts/pick_thresholds.py` functions, same seeds), with dev precision/flag rate;
  * real mistranslations on dev: Majors with a major mistranslation/wrong_term span, the share
    found in the riskiest 10% and at the review line;
  * the sense set: per type, recall at each system's own review threshold (false-flag rate for
    the meaning-preserving `synonym_ok`), the false-flag rate on the clean parents, and AUROC
    of the edited rows against their own clean parents (pair win rate in brackets), all on the
    app's fused score.
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
from pick_thresholds import fuse, pick_review  # noqa: E402  (same fusion and rule as the app)

SYSTEMS = ("v1", "v3a", "v3b")
CANDIDATES = ("v3a", "v3b")
LPS = {"en-ru": "enru", "en-es": "enes"}
MARGIN = 0.01
TARGET = 0.60
N_BOOT = 1000
MISTRANSLATION = {"mistranslation", "wrong_term"}


def load(path: Path) -> list[dict]:
    return M.load_scored(str(path))


def boot_delta(f, ya, sa, yb, sb, W) -> tuple[float, list[float]]:
    d = f(ya, sa) - f(yb, sb)
    reps = [f(ya, sa, w=w) - f(yb, sb, w=w) for w in W]
    lo, hi = M.percentile_ci(reps)
    return d, [lo, hi]


def ci(f, y, s, W) -> list[float]:
    lo, hi = M.percentile_ci(f(y, s, w=w) for w in W)
    return [lo, hi]


METRICS = {"auroc": M.auroc, "pr_auc": M.average_precision,
           "recall_at_10": lambda y, s, w=None: M.recall_at(y, s, 0.10, w)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="runs/v3_select/select.json")
    a = ap.parse_args()
    R = ROOT / a.results

    # ------------------------------------------------------------- dev
    dev_split = {lp: {r["id"]: r for r in load(ROOT / f"data/splits/{lp}.dev.jsonl")} for lp in LPS}
    fused, model = {}, {}
    for name in SYSTEMS:
        for lp, tag in LPS.items():
            rows = load(R / f"local_{name}_{tag}_dev.jsonl")
            assert {r["id"] for r in rows} == set(dev_split[lp]), f"{name} {lp}: not the full dev split"
            rows.sort(key=lambda r: r["id"])
            fused[name, lp] = fuse(dev_split[lp], rows)
            model[name, lp] = [{**r, "risk": r["model_risk"]} for r in fused[name, lp]]

    out: dict = {"rule": __doc__.split("SELECTION RULE")[1].split("Everything else")[0].strip(),
                 "dev": {}, "review": {}, "mistranslation_dev": {}, "sense": {}}
    for i, lp in enumerate(LPS):
        W = M.bootstrap_weights(M.group_keys(fused["v1", lp]), N_BOOT, 0)
        blk = out["dev"][lp] = {"n": len(fused["v1", lp]), "n_docs": len(set(M.group_keys(fused["v1", lp]))),
                                "majors": int(sum(r["label"] == "Major" for r in fused["v1", lp])), "systems": {}}
        ys = {}
        for kind, src in (("model", model), ("fused", fused)):
            for name in SYSTEMS:
                y, s = M.to_arrays(src[name, lp])
                ys[kind, name] = (y, s)
        for name in SYSTEMS:
            e = blk["systems"][name] = {}
            for kind in ("model", "fused"):
                y, s = ys[kind, name]
                e[kind] = {k: f(y, s) for k, f in METRICS.items()}
                e[kind]["ci"] = {k: ci(f, y, s, W) for k, f in METRICS.items()}
                if name != "v1":
                    yb, sb = ys[kind, "v1"]
                    e[kind]["delta_vs_v1"] = {k: boot_delta(f, y, s, yb, sb, W) for k, f in METRICS.items()}

        # review thresholds (pre-registered threshold rule, same seeds as pick_thresholds.py)
    for name in SYSTEMS:
        out["review"][name] = pick_review({lp: fused[name, lp] for lp in LPS}, TARGET, N_BOOT, 0)

    # check: v1's review thresholds equal the ones the app ships
    shipped = json.loads((ROOT / "secondread/thresholds.json").read_text())["adapters"]["v1"]["review"]["thresholds"]
    assert out["review"]["v1"]["thresholds"] == shipped, (out["review"]["v1"]["thresholds"], shipped)

    # ------------------------------------------------------------- the selection rule
    sel: dict = {"margin": MARGIN, "candidates": {}}
    for kind in ("model", "fused"):
        best, best_mean = None, None
        for c in CANDIDATES:
            v = {lp: out["dev"][lp]["systems"] for lp in LPS}
            d_pr = {lp: v[lp][c][kind]["pr_auc"] - v[lp]["v1"][kind]["pr_auc"] for lp in LPS}
            d_au = {lp: v[lp][c][kind]["auroc"] - v[lp]["v1"][kind]["auroc"] for lp in LPS}
            eligible = all(d_pr[lp] >= -MARGIN for lp in LPS) and all(d_au[lp] >= -MARGIN for lp in LPS)
            mean = float(np.mean(list(d_pr.values())))
            sel["candidates"].setdefault(c, {})[kind] = {"d_pr_auc": d_pr, "d_auroc": d_au, "eligible": eligible,
                                                         "mean_d_pr_auc": mean}
            if eligible and (best_mean is None or mean > best_mean):
                best, best_mean = c, mean
        decision = best if best is not None and best_mean > 0 else "v1"
        sel["decision" if kind == "model" else "decision_if_fused"] = decision
    out["selection"] = sel

    # ------------------------------------------------------------- real mistranslations (dev, descriptive)
    for lp in LPS:
        mis = set()
        for uid, u in dev_split[lp].items():
            if u["label"] == "Major" and any(sp.get("severity") in ("major", "critical") and sp.get("family") == "meaning"
                                             and sp.get("category") in MISTRANSLATION for sp in u.get("spans", [])):
                mis.add(uid)
        blk = out["mistranslation_dev"][lp] = {"n_majors_mistranslation": len(mis)}
        for name in SYSTEMS:
            rows = fused[name, lp]
            _, s = M.to_arrays(rows)
            thr = out["review"][name]["thresholds"][lp]
            other = {r["id"] for r in rows if r["label"] == "Major"} - mis
            y_mis = np.array([r["id"] in mis for r in rows], dtype=np.int64)
            y_oth = np.array([r["id"] in other for r in rows], dtype=np.int64)
            # share of each subset in the riskiest 10% of ALL dev rows (ties fractional, as recall_at)
            blk[name] = {"mis_in_top10": M.recall_at(y_mis, s), "mis_at_review": float(s[y_mis == 1].__ge__(thr).mean()),
                         "other_in_top10": M.recall_at(y_oth, s), "other_at_review": float(s[y_oth == 1].__ge__(thr).mean()),
                         "n_other": len(other)}

    # ------------------------------------------------------------- sense set (descriptive)
    sense = {r["id"]: r for r in load(ROOT / "data/synth/sense.test.jsonl")}
    parents = {r["id"]: r for r in load(ROOT / "data/synth/sense.test.parents.jsonl")}
    assert all(p["label"] == "OK" for p in parents.values())
    for name in SYSTEMS:
        srow = fuse(sense, load(R / f"local_{name}_synth_sense_test.jsonl"))
        prow = {r["id"]: r for r in fuse(parents, load(R / f"local_{name}_synth_sense_parents.jsonl"))}
        assert len(srow) == len(sense) and set(prow) == set(parents), name
        res = out["sense"][name] = {}
        for lp in LPS:
            thr = out["review"][name]["thresholds"][lp]
            rows_lp = [r for r in srow if r["lp"] == lp]
            types = sorted({sense[r["id"]]["synth_type"] for r in rows_lp})
            e = res[lp] = {"threshold": thr, "types": {}}
            for t in types + ["all_major"]:
                rt = [r for r in rows_lp if (sense[r["id"]]["synth_type"] == t if t != "all_major" else r["label"] == "Major")]
                pids = sorted({sense[r["id"]]["synth_parent"] for r in rt})
                pos = np.array([r["risk"] for r in rt])
                neg = np.array([prow[p]["risk"] for p in pids])
                yy = np.r_[np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
                au = M.auroc(yy, np.r_[pos, neg])
                # model-only version of the same AUROC, for reference
                posm = np.array([r["model_risk"] for r in rt])
                negm = np.array([prow[p]["model_risk"] for p in pids])
                au_m = M.auroc(yy, np.r_[posm, negm])
                win = float(np.mean([1.0 if r["risk"] > prow[sense[r["id"]]["synth_parent"]]["risk"] else
                                     0.5 if r["risk"] == prow[sense[r["id"]]["synth_parent"]]["risk"] else 0.0 for r in rt]))
                e["types"][t] = {"n": len(rt), "label": rt[0]["label"] if t != "all_major" else "Major",
                                 "flag_rate": float(np.mean(pos >= thr)), "auroc_vs_parents": au,
                                 "auroc_vs_parents_model_only": au_m, "pair_win_rate": win, "n_parents": len(pids)}
                if t == "all_major":
                    # document bootstrap (documents of the parents) for the pooled AUROC vs parents
                    docs = [r["doc"] for r in rt] + [prow[p]["doc"] for p in pids]
                    Wd = M.bootstrap_weights([f"{lp}|{d}" for d in docs], N_BOOT, 0)
                    e["types"][t]["auroc_vs_parents_ci"] = list(M.percentile_ci(M.auroc(yy, np.r_[pos, neg], w=w) for w in Wd))
            ppl = sorted({sense[r["id"]]["synth_parent"] for r in rows_lp})
            e["parents"] = {"n": len(ppl), "false_flag_rate": float(np.mean([prow[p]["risk"] >= thr for p in ppl]))}

    # ------------------------------------------------------------- print
    f3 = lambda x: f"{x:.3f}"
    fci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    fd = lambda d: f"{d[0]:+.3f} [{d[1][0]:+.3f}, {d[1][1]:+.3f}]"
    for lp in LPS:
        b = out["dev"][lp]
        print(f"\n## dev {lp}: n={b['n']} docs={b['n_docs']} Major={b['majors']}")
        for kind in ("model", "fused"):
            print(f"  [{kind}]")
            for name in SYSTEMS:
                e = b["systems"][name][kind]
                line = (f"   {name:4} AUROC {f3(e['auroc'])} {fci(e['ci']['auroc'])}  PR-AUC {f3(e['pr_auc'])} {fci(e['ci']['pr_auc'])}"
                        f"  R@10 {f3(e['recall_at_10'])} {fci(e['ci']['recall_at_10'])}")
                if "delta_vs_v1" in e:
                    dd = e["delta_vs_v1"]
                    line += f"\n        vs v1: dAUROC {fd(dd['auroc'])} dPR-AUC {fd(dd['pr_auc'])} dR@10 {fd(dd['recall_at_10'])}"
                print(line)
    print("\n## selection (model-only, point estimates)")
    for c, v in sel["candidates"].items():
        m = v["model"]
        print(f"  {c}: dPR-AUC en-ru {m['d_pr_auc']['en-ru']:+.4f} en-es {m['d_pr_auc']['en-es']:+.4f} | "
              f"dAUROC en-ru {m['d_auroc']['en-ru']:+.4f} en-es {m['d_auroc']['en-es']:+.4f} | eligible {m['eligible']} | "
              f"mean dPR-AUC {m['mean_d_pr_auc']:+.4f}   (fused: eligible {v['fused']['eligible']}, mean {v['fused']['mean_d_pr_auc']:+.4f})")
    print(f"  DECISION: {sel['decision']}   (under the fused reading: {sel['decision_if_fused']})")
    print("\n## review thresholds (fused, dev recall >= 0.60)")
    for name in SYSTEMS:
        for lp in LPS:
            st = out["review"][name]["stats"][lp]["dev"]
            print(f"  {name:4} {lp}: thr {out['review'][name]['thresholds'][lp]:.4f} recall {st['recall']:.3f} {fci(st['recall_ci'])} "
                  f"P {st['precision']:.3f} {fci(st['precision_ci'])} flags {st['flag_rate']:.1%} {fci(st['flag_rate_ci'])} "
                  f"({st['flagged']}/{st['n']}, {st['majors_flagged']}/{st['majors']}) thr re-picked {fci(st['threshold_bootstrap_ci'])}")
        if out["review"][name]["pooled"]:
            print(f"  {name} POOLED reported: {out['review'][name]['pooled']['threshold']}")
    print("\n## real mistranslations on dev (fused)")
    for lp in LPS:
        b = out["mistranslation_dev"][lp]
        for name in SYSTEMS:
            e = b[name]
            print(f"  {lp} {name:4}: mistranslation Majors (n={b['n_majors_mistranslation']}) top10 {e['mis_in_top10']:.3f} "
                  f"review {e['mis_at_review']:.3f} | other Majors (n={e['n_other']}) top10 {e['other_in_top10']:.3f} review {e['other_at_review']:.3f}")
    print("\n## sense set (fused; flag rate at own review threshold | AUROC vs parents (win))")
    for lp in LPS:
        for name in SYSTEMS:
            e = out["sense"][name][lp]
            cells = [f"{t}: {v['flag_rate']:.2f} | {v['auroc_vs_parents']:.2f} ({v['pair_win_rate']:.2f}) n={v['n']}"
                     for t, v in e["types"].items()]
            print(f"  {lp} {name:4} thr {e['threshold']:.4f}: " + "; ".join(cells)
                  + f"; parents ff {e['parents']['false_flag_rate']:.2f} (n={e['parents']['n']})"
                  + f"; all_major AUROC CI {fci(e['types']['all_major']['auroc_vs_parents_ci'])}")

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("\nwrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
