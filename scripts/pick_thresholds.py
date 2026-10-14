"""Pick the app's flag thresholds per adapter and language pair; write secondread/thresholds.json.

usage: PYTHONPATH=. uv run python scripts/pick_thresholds.py                       # "f1" operating point
       PYTHONPATH=. uv run python scripts/pick_thresholds.py --target-recall 0.60  # "review" operating point
       common options: [--out secondread/thresholds.json] [--results results] [--adapters base v1 v2]
                       [--fused-dir runs/ship]

The server flags a segment when its fused risk is at or above the threshold of the active
operating point, for the loaded adapter and the segment's language pair. The fused risk is the
model risk with the train-calibrated log-odds weights of the two rule checks added
(`secondread.agent.aggregate`, `secondread/rule_calibration.json`). So every threshold is picked
on the fused risk, not on the model risk alone. For every unit of the FULL dev split of a pair,
the script runs `number_check` and `negation_check` exactly as the server does and fuses them
with that adapter's local dev score (`probe_local.py` output in results/). Then:

  f1      (default) the F1-optimal threshold (`metrics.best_f1_threshold`), rounded to 4
          decimals. Test files are read only to report what the threshold does there
          (--skip-test leaves them out); they never choose it.
  review  (--target-recall R) the HIGHEST threshold whose dev recall of Major is at least R
          (`metrics.max_threshold_at_recall`), rounded DOWN to 4 decimals so that the stored
          value still has dev recall >= R. Stored with dev precision, recall and flag rate and
          their 95% CIs from a document-level bootstrap at that fixed threshold (--n-boot,
          --seed). This mode opens dev files only, never a test file.
          When a pair's recall CI is wider than +-0.15 (hi - lo > 0.30), the threshold that the
          same rule gives on all pairs' dev splits pooled is reported too, for information
          (`"pooled"`, `"used": false`). The per-pair threshold is the one written and used.

Each run rewrites only its own operating point (the "f1" or "review" block of every adapter it
scores) and keeps everything else in the file, so the two modes can be re-run independently.
A file in the one-operating-point format (adapter-level "thresholds"/"stats", 2026-10-14) is
read as the "f1" point. The backend uses `default_operating_point` unless
SECONDREAD_OPERATING_POINT or the request says otherwise (secondread/agent.py).
`--fused-dir` also writes the fused scores as scored jsonl files for `scripts/report.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from secondread import checks
from secondread import metrics as M
from secondread.agent import ModelScore, aggregate

ROOT = Path(__file__).resolve().parents[1]
LPS = {"en-ru": "enru", "en-es": "enes"}
POINTS = ("review", "f1")      # operating points, in the order they appear in the file
CI_HALF_WIDTH = 0.15           # review: a recall CI wider than +-this triggers the pooled report

# name -> GGUF (None = the base model: no adapter, or lora_scale 0) and a one-line description
SYSTEMS = {
    "base": (None, "Qwen3.5-4B Q4_K_M without an adapter (also any adapter at lora_scale 0)"),
    "v1": ("models/secondread-lora-v1.gguf",
           "LoRA v1: rank 16 MLP-only, 1 epoch on 14,791 human-labelled WMT units "
           "(data/sft_v1.jsonl), Tinker run 479fd56b-2a66-539a-ae24-5b020ba97bbb:train:0"),
    "v2": ("models/secondread-lora-v2.gguf",
           "LoRA v2: as v1 plus 1,109 rule and 3,986 Gemma 4 synthetic rows (data/sft_v2.jsonl), "
           "Tinker run 4087b7c0-b94b-5bbc-87a9-10ac847ad928:train:0"),
    "v3a": ("models/secondread-lora-v3a.gguf",
            "LoRA v3a (candidate): as v1 plus 1,467 Gemma 4 wrong-sense/false-friend/antonym/synonym rows "
            "(data/synth/sense.v3a.train.jsonl, at most 12% of each pair; data/sft_v3a.jsonl), "
            "Tinker run abc2567b-8635-5c1b-9e87-20f536e95a06:train:0"),
    "v3b": ("models/secondread-lora-v3b.gguf",
            "LoRA v3b (candidate): as v1 plus 1,613 wrong-sense rows (at most 12% per pair) and 1,426 v2 "
            "synthetic rows (at most 8% per pair) (data/synth/mix_v3b/; data/sft_v3b.jsonl), "
            "Tinker run 7df264ae-f959-55df-aa64-e32f372a36e7:train:0"),
}

FUSED = ("the fused risk (the model risk plus the train-calibrated log-odds weights of the "
         "number/negation rule checks in secondread/rule_calibration.json, as secondread.agent.aggregate)")
F1_METHOD = (f"F1-optimal threshold on {FUSED} over the full dev split of each language pair, rounded to "
             "4 decimals; scripts/pick_thresholds.py")


def review_method(target: float, n_boot: int, seed: int) -> str:
    return (f"Highest threshold whose dev recall of Major is at least {target:.2f}, on {FUSED} over the full "
            "dev split of each language pair, rounded down to 4 decimals; dev precision, recall and flag rate "
            f"at it with 95% document-bootstrap CIs ({n_boot} replicates, seed {seed}); if a pair's recall CI is "
            f"wider than +-{CI_HALF_WIDTH:.2f}, the same rule on all pairs' dev pooled is reported too but not "
            f"used; reads no test file; scripts/pick_thresholds.py --target-recall {target:g}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fuse(split: dict[str, dict], scored: list[dict]) -> list[dict]:
    """Scored rows -> the same rows with the server's fused risk (model + rule checks)."""
    out = []
    for r in scored:
        u = split[r["id"]]
        rules = [checks.number_check(u["src"], u["mt"], u["lp"]), checks.negation_check(u["src"], u["mt"], u["lp"])]
        risk, _, _, _ = aggregate(ModelScore(risk=r["risk"]), rules, 1.0, "model", u["lp"])
        out.append({"id": r["id"], "label": r["label"], "risk": risk, "model_risk": r["risk"],
                    "rule_hit": any(c.issues for c in rules), "doc": u["doc"], "lp": u["lp"]})
    return out


def stats(rows: list[dict], thr: float) -> dict:
    y, s = M.to_arrays(rows)
    prf = M.prf_at(y, s, thr)
    return {"n": len(rows), "major_rate": round(float(y.mean()), 4),
            "precision": round(prf["precision"], 4), "recall": round(prf["recall"], 4),
            "f1": round(prf["f1"], 4), "flag_rate": round(prf["flag_rate"], 4),
            "auroc_fused": round(M.auroc(y, s), 4), "pr_auc_fused": round(M.average_precision(y, s), 4),
            "recall_at_10_fused": round(M.recall_at(y, s), 4)}


# ------------------------------------------------------------------ review point
def floor4(x: float) -> float:
    return round(math.floor(x * 1e4 + 1e-9) / 1e4, 4)


def recall_threshold(y: np.ndarray, s: np.ndarray, target: float) -> tuple[float, float]:
    """(stored threshold, exact one): the rule's threshold rounded down to 4 decimals, lowered
    further if float rounding ever left the stored value with recall below the target."""
    exact = M.max_threshold_at_recall(y, s, target)
    thr = floor4(exact)
    while M.prf_at(y, s, thr)["recall"] < target - 1e-9:
        thr = round(thr - 1e-4, 4)
    return thr, exact


def _ci(values) -> list[float]:
    lo, hi = M.percentile_ci(v for v in values if math.isfinite(v))
    return [round(lo, 4), round(hi, 4)]


def review_stats(rows: list[dict], thr: float, W: np.ndarray, target: float) -> dict:
    """Dev numbers at a fixed threshold, with document-bootstrap CIs (rows of W = replicates)."""
    y, s = M.to_arrays(rows)
    prf = M.prf_at(y, s, thr)
    flag = s >= thr
    boot = [M.prf_at(y, s, thr, w) for w in W]
    rec_ci = _ci(b["recall"] for b in boot)
    return {
        "n": len(rows), "n_docs": len(set(M.group_keys(rows))), "majors": int(y.sum()),
        "major_rate": round(float(y.mean()), 4), "flagged": int(flag.sum()), "majors_flagged": int((flag & (y == 1)).sum()),
        "recall": round(prf["recall"], 4), "recall_ci": rec_ci,
        "recall_ci_half_width": round((rec_ci[1] - rec_ci[0]) / 2, 4),
        "recall_ci_wider_than_limit": bool(rec_ci[1] - rec_ci[0] > 2 * CI_HALF_WIDTH + 1e-12),
        "precision": round(prf["precision"], 4), "precision_ci": _ci(b["precision"] for b in boot),
        "flag_rate": round(prf["flag_rate"], 4), "flag_rate_ci": _ci(b["flag_rate"] for b in boot),
        "f1": round(prf["f1"], 4),
        # how far the rule's threshold itself moves when it is re-picked on each replicate
        "threshold_bootstrap_ci": _ci(M.max_threshold_at_recall(y, s, target, w) for w in W),
        "auroc_fused": round(M.auroc(y, s), 4), "pr_auc_fused": round(M.average_precision(y, s), 4),
        "recall_at_10_fused": round(M.recall_at(y, s), 4),
    }


def pick_review(dev: dict[str, list[dict]], target: float, n_boot: int, seed: int) -> dict:
    block: dict = {"thresholds": {}, "stats": {}}
    weights = {}
    for i, (lp, rows) in enumerate(dev.items()):
        y, s = M.to_arrays(rows)
        thr, exact = recall_threshold(y, s, target)
        weights[lp] = M.bootstrap_weights(M.group_keys(rows), n_boot, seed + i)
        st = review_stats(rows, thr, weights[lp], target)
        ym, sm = M.to_arrays([{**r, "risk": r["model_risk"]} for r in rows])
        block["thresholds"][lp] = thr
        block["stats"][lp] = {"exact_threshold": round(exact, 6),
                              "model_only_threshold": recall_threshold(ym, sm, target)[0],
                              "dev": st,
                              "dev_rule_hit_rate": round(sum(r["rule_hit"] for r in rows) / len(rows), 4)}
    wide = [lp for lp in dev if block["stats"][lp]["dev"]["recall_ci_wider_than_limit"]]
    block["pooled"] = None
    if wide:  # pre-registered: report the pooled threshold, still use the per-pair ones
        rows = [r for lp in dev for r in dev[lp]]
        W = np.hstack([weights[lp] for lp in dev])  # stratified: each pair keeps its own documents
        y, s = M.to_arrays(rows)
        thr, exact = recall_threshold(y, s, target)
        block["pooled"] = {
            "used": False, "pairs": list(dev), "exact_threshold": round(exact, 6), "threshold": thr,
            "reported_because": f"dev recall CI wider than +-{CI_HALF_WIDTH:.2f} for {', '.join(wide)}",
            "bootstrap": "stratified by pair: each replicate resamples each pair's own dev documents",
            "dev": review_stats(rows, thr, W, target),
            "dev_per_pair": {lp: review_stats(dev[lp], thr, weights[lp], target) for lp in dev},
        }
    return block


# ------------------------------------------------------------------ the file
def load_existing(path: Path) -> dict:
    """The current file, with the one-operating-point format read as the "f1" point."""
    data = json.loads(path.read_text()) if path.exists() else {}
    ops = data.setdefault("operating_points", {})
    if "method" in data and "f1" not in ops:  # format of 2026-10-14: a single F1 point
        ops["f1"] = {"method": data["method"], "generated": data.get("generated"),
                     "rule_calibration": data.get("rule_calibration")}
    for k in ("method", "generated", "rule_calibration"):
        data.pop(k, None)
    for e in (data.get("adapters") or {}).values():
        if "thresholds" in e and "f1" not in e:
            e["f1"] = {"thresholds": e.pop("thresholds"), "stats": e.pop("stats", {})}
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="secondread/thresholds.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--adapters", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--target-recall", type=float,
                    help="pick the 'review' operating point: the highest threshold with dev recall >= this")
    ap.add_argument("--n-boot", type=int, default=1000, help="review: document-bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0, help="review: bootstrap seed (pair i uses seed + i)")
    ap.add_argument("--skip-test", action="store_true", help="f1: do not read the test splits for information")
    ap.add_argument("--fused-dir", help="also write fused scored jsonl files here")
    a = ap.parse_args()
    point = "review" if a.target_recall is not None else "f1"
    if point == "review" and not 0.0 < a.target_recall <= 1.0:
        ap.error("--target-recall must be in (0, 1]")
    parts = ("dev",) if point == "review" or a.skip_test else ("dev", "test")

    path = ROOT / a.out
    old = load_existing(path)
    splits = {(lp, part): {r["id"]: r for r in M.load_scored(ROOT / f"data/splits/{lp}.{part}.jsonl")}
              for lp in LPS for part in parts}
    rulecal = {"file": "secondread/rule_calibration.json",
               "sha256": sha256(ROOT / "secondread/rule_calibration.json")}
    ops = dict(old.get("operating_points") or {})
    if point == "review":
        ops["review"] = {"method": review_method(a.target_recall, a.n_boot, a.seed), "target_recall": a.target_recall,
                         "n_boot": a.n_boot, "seed": a.seed, "ci_half_width_limit": CI_HALF_WIDTH,
                         "generated": time.strftime("%Y-%m-%d"), "rule_calibration": rulecal}
    else:
        ops["f1"] = {"method": F1_METHOD, "generated": time.strftime("%Y-%m-%d"), "rule_calibration": rulecal,
                     "test_stats": "none (--skip-test)" if a.skip_test else "for information only"}
    out = {
        "default_operating_point": old.get("default_operating_point") or ("review" if "review" in ops else "f1"),
        "fallback": old.get("fallback", 0.5),
        "operating_points": {p: ops[p] for p in POINTS if p in ops},
        "adapters": {},
    }

    for name, (gguf, desc) in SYSTEMS.items():
        prev = (old.get("adapters") or {}).get(name) or {}
        if gguf and (ROOT / gguf).exists():
            sha = sha256(ROOT / gguf)
        else:  # e.g. a fresh clone without that GGUF: keep the hash recorded before
            sha = prev.get("sha256")
            if gguf and name in a.adapters:
                print(f"WARNING: {gguf} not found; keeping the recorded sha256 {sha}")
        entry: dict = {"sha256": sha, "file": gguf, "description": desc}
        entry.update({p: prev[p] for p in POINTS if p in prev})
        files = {(lp, part): ROOT / a.results / f"local_{name}_{tag}_{part}.jsonl" for lp, tag in LPS.items() for part in parts}
        if name not in a.adapters or not all(f.exists() for f in files.values()):
            if name in a.adapters:
                print(f"WARNING: {name}: no full set of score files in {a.results}/; its {point} block is kept as it was")
            out["adapters"][name] = {k: entry[k] for k in ("sha256", "file", "description", *POINTS) if k in entry}
            continue
        fused = {key: fuse(splits[key], M.load_scored(f)) for key, f in files.items()}
        for lp in LPS:
            assert len(fused[lp, "dev"]) == len(splits[lp, "dev"]), f"{name} {lp}: dev file is not the full split"

        if point == "review":
            block = pick_review({lp: fused[lp, "dev"] for lp in LPS}, a.target_recall, a.n_boot, a.seed)
            for lp in LPS:
                st = block["stats"][lp]
                d = st["dev"]
                print(f"{name:4} {lp} review: threshold {block['thresholds'][lp]:.4f} (exact {st['exact_threshold']:.6f}, "
                      f"model-only {st['model_only_threshold']:.4f}) | dev recall {d['recall']:.3f} "
                      f"[{d['recall_ci'][0]:.3f}, {d['recall_ci'][1]:.3f}] (+-{d['recall_ci_half_width']:.3f}) "
                      f"P {d['precision']:.3f} [{d['precision_ci'][0]:.3f}, {d['precision_ci'][1]:.3f}] "
                      f"flags {d['flag_rate']:.1%} [{d['flag_rate_ci'][0]:.1%}, {d['flag_rate_ci'][1]:.1%}] "
                      f"({d['flagged']}/{d['n']}, {d['majors_flagged']}/{d['majors']} Major) "
                      f"| threshold re-picked per replicate [{d['threshold_bootstrap_ci'][0]:.4f}, "
                      f"{d['threshold_bootstrap_ci'][1]:.4f}]")
            if block["pooled"]:
                pl = block["pooled"]
                d = pl["dev"]
                print(f"{name:4} pooled (not used; {pl['reported_because']}): threshold {pl['threshold']:.4f} | "
                      f"dev recall {d['recall']:.3f} [{d['recall_ci'][0]:.3f}, {d['recall_ci'][1]:.3f}] "
                      f"P {d['precision']:.3f} flags {d['flag_rate']:.1%}")
                for lp, d in pl["dev_per_pair"].items():
                    print(f"       at the pooled threshold, {lp}: recall {d['recall']:.3f} "
                          f"[{d['recall_ci'][0]:.3f}, {d['recall_ci'][1]:.3f}] P {d['precision']:.3f} "
                          f"flags {d['flag_rate']:.1%}")
        else:
            block = {"thresholds": {}, "stats": {}}
            for lp in LPS:
                dev = fused[lp, "dev"]
                y, s = M.to_arrays(dev)
                thr = round(M.best_f1_threshold(y, s), 4)
                ym, sm = M.to_arrays([{**r, "risk": r["model_risk"]} for r in dev])
                st = {"model_only_dev_threshold": round(M.best_f1_threshold(ym, sm), 4), "dev": stats(dev, thr)}
                if "test" in parts:
                    st["test"] = stats(fused[lp, "test"], thr)
                st["dev_rule_hit_rate"] = round(sum(r["rule_hit"] for r in dev) / len(dev), 4)
                block["thresholds"][lp], block["stats"][lp] = thr, st
                line = (f"{name:4} {lp} f1: threshold {thr:.4f} (model-only {st['model_only_dev_threshold']:.4f}) | "
                        f"dev F1 {st['dev']['f1']:.3f} R {st['dev']['recall']:.3f} flags {st['dev']['flag_rate']:.1%}")
                if "test" in st:
                    line += (f" | test F1 {st['test']['f1']:.3f} P {st['test']['precision']:.3f} "
                             f"R {st['test']['recall']:.3f} flags {st['test']['flag_rate']:.1%}")
                print(line)
        entry[point] = block
        out["adapters"][name] = {k: entry[k] for k in ("sha256", "file", "description", *POINTS) if k in entry}

        if a.fused_dir:
            d = ROOT / a.fused_dir
            d.mkdir(parents=True, exist_ok=True)
            for (lp, part), rows in fused.items():
                with open(d / f"fused_{name}_{LPS[lp]}_{part}.jsonl", "w") as f:
                    for r in rows:
                        f.write(json.dumps({k: r[k] for k in ("id", "label", "risk", "doc", "lp")}) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
