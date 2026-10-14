"""Pick the app's flag thresholds per adapter and language pair; write secondread/thresholds.json.

usage: PYTHONPATH=. uv run python scripts/pick_thresholds.py [--out secondread/thresholds.json]
           [--fused-dir runs/ship]

The server flags a segment when its fused risk is at or above the threshold for the loaded
adapter and the segment's language pair. The fused risk is the model risk with the
train-calibrated log-odds weights of the two rule checks added (`secondread.agent.aggregate`,
`secondread/rule_calibration.json`). So the threshold is picked on fused risk, not on the
model risk alone:

  1. for every unit of the full dev split, run `number_check` and `negation_check`, exactly as
     the server does;
  2. fuse them with that adapter's local dev score (`probe_local.py` output in results/);
  3. keep the F1-optimal threshold (`metrics.best_f1_threshold`), rounded to 4 decimals.

The dev and test statistics stored next to each threshold are computed at the rounded value.
Test files are only read to report what the threshold does there; they never choose it.
`--fused-dir` also writes the fused scores as scored jsonl files for `scripts/report.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from secondread import checks
from secondread import metrics as M
from secondread.agent import ModelScore, aggregate

ROOT = Path(__file__).resolve().parents[1]
LPS = {"en-ru": "enru", "en-es": "enes"}

# name -> GGUF (None = the base model: no adapter, or lora_scale 0) and a one-line description
SYSTEMS = {
    "base": (None, "Qwen3.5-4B Q4_K_M without an adapter (also any adapter at lora_scale 0)"),
    "v1": ("models/secondread-lora-v1.gguf",
           "LoRA v1: rank 16 MLP-only, 1 epoch on 14,791 human-labelled WMT units "
           "(data/sft_v1.jsonl), Tinker run 479fd56b-2a66-539a-ae24-5b020ba97bbb:train:0"),
    "v2": ("models/secondread-lora-v2.gguf",
           "LoRA v2: as v1 plus 1,109 rule and 3,986 Gemma 4 synthetic rows (data/sft_v2.jsonl), "
           "Tinker run 4087b7c0-b94b-5bbc-87a9-10ac847ad928:train:0"),
}


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="secondread/thresholds.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--fused-dir", help="also write fused scored jsonl files here")
    a = ap.parse_args()

    splits = {(lp, part): {r["id"]: r for r in M.load_scored(ROOT / f"data/splits/{lp}.{part}.jsonl")}
              for lp in LPS for part in ("dev", "test")}
    out = {
        "method": "F1-optimal threshold on the fused risk (the model risk plus the train-calibrated "
                  "log-odds weights of the number/negation rule checks in secondread/rule_calibration.json, "
                  "as secondread.agent.aggregate) over the full dev split of each language pair, rounded to "
                  "4 decimals; scripts/pick_thresholds.py",
        "generated": time.strftime("%Y-%m-%d"),
        "fallback": 0.5,
        "rule_calibration": {"file": "secondread/rule_calibration.json",
                             "sha256": sha256(ROOT / "secondread/rule_calibration.json")},
        "adapters": {},
    }
    old = json.loads((ROOT / a.out).read_text()) if (ROOT / a.out).exists() else {"adapters": {}}
    for name, (gguf, desc) in SYSTEMS.items():
        if gguf and (ROOT / gguf).exists():
            sha = sha256(ROOT / gguf)
        else:  # e.g. a fresh clone without that GGUF: keep the hash recorded before
            sha = (old["adapters"].get(name) or {}).get("sha256")
            if gguf:
                print(f"WARNING: {gguf} not found; keeping the recorded sha256 {sha}")
        entry: dict = {"sha256": sha, "file": gguf, "description": desc, "thresholds": {}, "stats": {}}
        for lp, tag in LPS.items():
            dev = fuse(splits[lp, "dev"], M.load_scored(ROOT / a.results / f"local_{name}_{tag}_dev.jsonl"))
            test = fuse(splits[lp, "test"], M.load_scored(ROOT / a.results / f"local_{name}_{tag}_test.jsonl"))
            assert len(dev) == len(splits[lp, "dev"]), f"{name} {lp}: dev file is not the full split"
            y, s = M.to_arrays(dev)
            thr = round(M.best_f1_threshold(y, s), 4)
            ym, sm = M.to_arrays([{**r, "risk": r["model_risk"]} for r in dev])
            entry["thresholds"][lp] = thr
            entry["stats"][lp] = {"model_only_dev_threshold": round(M.best_f1_threshold(ym, sm), 4),
                                  "dev": stats(dev, thr), "test": stats(test, thr),
                                  "dev_rule_hit_rate": round(sum(r["rule_hit"] for r in dev) / len(dev), 4)}
            print(f"{name:4} {lp}: threshold {thr:.4f} (model-only {entry['stats'][lp]['model_only_dev_threshold']:.4f}) | "
                  f"dev F1 {entry['stats'][lp]['dev']['f1']:.3f} flags {entry['stats'][lp]['dev']['flag_rate']:.1%} | "
                  f"test F1 {entry['stats'][lp]['test']['f1']:.3f} P {entry['stats'][lp]['test']['precision']:.3f} "
                  f"R {entry['stats'][lp]['test']['recall']:.3f} flags {entry['stats'][lp]['test']['flag_rate']:.1%}")
            if a.fused_dir:
                d = ROOT / a.fused_dir
                d.mkdir(parents=True, exist_ok=True)
                for part, rows in (("dev", dev), ("test", test)):
                    with open(d / f"fused_{name}_{tag}_{part}.jsonl", "w") as f:
                        for r in rows:
                            f.write(json.dumps({k: r[k] for k in ("id", "label", "risk", "doc", "lp")}) + "\n")
        out["adapters"][name] = entry
    path = ROOT / a.out
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
