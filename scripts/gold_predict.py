"""Blind predictions for the real-document gold set (docs/REALDOCS_GOLD.md, section 6).

Reads ONLY the blind input file written by `scripts/gold_score.py --export-inputs` (uid, doc_id, seg_index,
context_src, src, mt). It never opens labels.jsonl, annotations*, adjudication*, key_systems.json,
gold_stats.json or the earlier labels. Nothing here is tuned: models, prompts, rule weights and thresholds are
the shipped ones.

usage:
  py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
  py scripts/gold_score.py --export-inputs runs/gold_eval/gold_inputs.jsonl
  py scripts/gold_predict.py v1    [--llama-url http://127.0.0.1:9931]   # v1 app score + v1 model-only
  py scripts/gold_predict.py base  --llama-url http://127.0.0.1:9980     # LoRA scale 0 = base model, app score
  py scripts/gold_predict.py control --inputs runs/gold_eval/gold_inputs_with_ref.jsonl   # ref_es scored as MT
  set -a; . ./.env; set +a
  py scripts/gold_predict.py gemma --cap 0.15 [--dry-run]                # Gemma 4 31B direct, cached
  py scripts/gold_predict.py gemma-control --inputs runs/gold_eval/gold_inputs_with_ref.jsonl --cap 0.13
  py scripts/gold_predict.py freeze    # sha256 of every prediction file -> runs/gold_eval/PREDICTIONS_MANIFEST.json

Systems (each writes runs/gold_eval/preds_<name>.jsonl, one line per uid, {uid, risk, ...}):
  v1_app     secondread.agent.ReviewAgent.review, the backend pipeline in-process, against llama-server with LoRA
             v1 at scale 1, lp en-es, operating point "review". risk = the fused score unrounded (logit(model
             risk) + the train-calibrated rule weights, secondread.rulecal), which is what the backend compares
             with the threshold; `risk_backend` is the rounded value the API returns, `flagged_backend` its flag.
  v1_model   the same run's model risk p(Major)/(p(Major)+p(OK)) alone, no rules.
  base_app   the same pipeline with lora_scale 0 (the base model; base thresholds), against its own server.
  gemma      gemma-4-31B-it on DigitalOcean, the SecondRead classifier prompt (scripts/baseline_gemma.py, direct
             method: temperature 0, 3 tokens, top-10 logprobs), risk = p(Major)/(p(Major)+p(OK)). Cache:
             data/cache/gemma_baseline.jsonl, keyed "direct|gold:<uid>"; a reply already cached for a
             byte-identical request body under another key (e.g. the same src/mt sent for docs/REALDOCS.md) is
             reused instead of paying for it again. --cap is the new spend allowed for this run.
  control    v1 app score on the publisher's official Spanish (ref_es) in place of the MT, one row per source
             segment (720), id "<doc_id>|<seg_index>". A false-alarm control: these are human translations.
  gemma-control  the same control for Gemma direct (cache key "direct|gold-ref:<doc_id>|<seg_index>").
No system gets ref_es as a reference, and none uses context_src (all are segment-level).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "gold_eval"
LP = "en-es"
FORBIDDEN = ("labels.jsonl", "key_systems.json", "gold_stats.json", "adjudication", "annotations")


def read_inputs(path: Path) -> list[dict]:
    if any(f in path.name for f in FORBIDDEN):
        sys.exit(f"{path} is not a blind input file")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if any("label" in r or "system" in r for r in rows):
        sys.exit(f"{path} carries labels or system names: export it with gold_score.py --export-inputs")
    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids)), "duplicate uid in inputs"
    return rows


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} lines, sha256 {sha(path)[:12]})", flush=True)


def check_preds(rows: list[dict], uids: list[str]) -> None:
    got = [r["uid"] for r in rows]
    assert len(got) == len(set(got)) == len(uids) and set(got) == set(uids), "uid set mismatch"
    bad = [r["uid"] for r in rows if not isinstance(r["risk"], (int, float)) or not math.isfinite(r["risk"])]
    assert not bad, f"non-finite risk for {bad[:5]}"


# ------------------------------------------------------------------ backend (v1, base, control) ----

async def run_backend(items: list[dict], llama_url: str, lora_scale: float | None) -> tuple[list, dict, dict]:
    from secondread.agent import ReviewAgent
    agent = ReviewAgent(llama_url, operating_point="review")
    try:
        agent.warm_up()
        health = await agent.health()
        res = await agent.review(items, lp=LP, lora_scale=lora_scale)
    finally:
        await agent.aclose()
    return res.segments, res.summary, health


def backend_rows(items: list[dict], segs: list, id_key: str) -> tuple[list[dict], list[dict], dict]:
    """(app rows, model-only rows, consistency stats). The fused risk is recomputed unrounded with the
    backend's own function on the same check results; it must round to what the backend returned."""
    from secondread import checks
    from secondread.agent import RULES
    app, model, st = [], [], collections.Counter()
    for it, s in zip(items, segs, strict=True):
        assert s.id == it["id"]
        results = [checks.number_check(it["src"], it["mt"], LP), checks.negation_check(it["src"], it["mt"], LP)]
        m_risk = (s.model or {}).get("risk")
        fused = RULES.risk(m_risk, results, LP)
        if round(fused, 4) != s.risk:
            st["fused_mismatch"] += 1
        st["model_error"] += m_risk is None
        hits = [c for r in results for c in r.codes]
        st["rule_hit_units"] += bool(hits)
        weights = {k: v.get("weight") for k, v in s.signals.items() if isinstance(v, dict) and v.get("issues")}
        app.append({id_key: it["id"], "risk": fused, "risk_backend": s.risk, "flagged_backend": s.flagged,
                    "model_risk": m_risk, "rule_codes": hits, "rule_weights": weights,
                    "model_error": (s.model or {}).get("error"), "labels_in_top": (s.model or {}).get("labels_in_top")})
        model.append({id_key: it["id"], "risk": m_risk, "p_major": (s.model or {}).get("p_major"),
                      "p_ok": (s.model or {}).get("p_ok")})
    return app, model, dict(st)


def cmd_backend(a, name: str) -> None:
    rows = read_inputs(a.inputs)
    if name == "control":
        items = [{"id": c["uid"], "src": c["src"], "mt": c["mt"]} for c in control_items(rows)]
        id_key = "id"
    else:
        items = [{"id": r["uid"], "src": r["src"], "mt": r["mt"]} for r in rows]
        id_key = "uid"
    lora_scale = 0.0 if name == "base" else None
    t0 = time.perf_counter()
    segs, summary, health = asyncio.run(run_backend(items, a.llama_url, lora_scale))
    wall = time.perf_counter() - t0
    app, model, st = backend_rows(items, segs, id_key)
    expect_key = "base" if name == "base" else "v1"
    meta = {"system": name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "inputs": str(a.inputs),
            "inputs_sha256": sha(a.inputs), "llama_url": a.llama_url, "lora_scale": lora_scale,
            "health": {k: health.get(k) for k in ("ok", "model", "base_model", "lora_adapters", "adapter_version",
                                                   "operating_point", "operating_point_source", "thresholds",
                                                   "threshold_source", "operating_points")},
            "summary": {k: summary.get(k) for k in ("segments", "flagged", "thresholds", "threshold_source",
                                                     "operating_point", "model", "lora_active", "adapter_version",
                                                     "model_errors", "timing_ms")},
            "wall_s": round(wall, 1), "consistency": st,
            "thresholds_json_sha256": sha(ROOT / "secondread" / "thresholds.json"),
            "rule_calibration_sha256": sha(ROOT / "secondread" / "rule_calibration.json"),
            "expected_calibration_key": expect_key}
    print(json.dumps(meta["summary"], ensure_ascii=False), flush=True)
    print(f"consistency: {st}; wall {wall:.0f} s", flush=True)
    if summary["threshold_source"] != "dev_review":
        sys.exit(f"unexpected threshold source {summary['threshold_source']}")
    if name in ("v1", "control") and (summary.get("adapter_version") != "v1" or not summary.get("lora_active")):
        sys.exit("the server at --llama-url does not score with LoRA v1 at scale 1")
    if name == "base" and summary.get("lora_active"):
        sys.exit("base run has an active LoRA")
    if st.get("fused_mismatch"):
        sys.exit(f"recomputed fused risk differs from the backend on {st['fused_mismatch']} rows")
    if name == "control":
        write_jsonl(OUT / "preds_control_ref_es_v1_app.jsonl", app)
    else:
        if st.get("model_error"):
            sys.exit(f"{st['model_error']} model errors; refusing to write a model-only file with gaps")
        uids = [r["uid"] for r in rows]
        check_preds(app, uids)
        check_preds(model, uids)
        write_jsonl(OUT / f"preds_{name}_app.jsonl", app)
        if name == "v1":
            write_jsonl(OUT / "preds_v1_model.jsonl", model)
    (OUT / f"meta_{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------ gemma ----

def load_gemma():
    spec = importlib.util.spec_from_file_location("baseline_gemma", ROOT / "scripts" / "baseline_gemma.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def control_items(rows: list[dict]) -> list[dict]:
    """One item per source segment with the official Spanish (ref_es) as the MT; id "<doc_id>|<seg_index>"."""
    if not rows or "ref_es" not in rows[0]:
        sys.exit("the control needs the --with-ref export (runs/gold_eval/gold_inputs_with_ref.jsonl)")
    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(f"{r['doc_id']}|{r['seg_index']}", r)
    return [{"uid": k, "src": r["src"], "mt": r["ref_es"]} for k, r in sorted(seen.items())]


def cmd_gemma(a, control: bool = False) -> None:
    G = load_gemma()
    rows = read_inputs(a.inputs)
    if control:
        rows = control_items(rows)
    args = SimpleNamespace(direct_max_tokens=3, top_logprobs=10, concurrency=8, rpm=110, base_url=G.DO_BASE_URL,
                           progress=250)
    cache = G.Cache(G.CACHE_PATH)
    by_hash: dict[str, dict] = {}
    for e in cache.data.values():
        if e.get("method") == "direct":
            by_hash.setdefault(e["req_hash"], e)
    spent_before = cache.total_cost
    plan, jobs = [], []
    for r in rows:
        row = {"id": f"gold-ref:{r['uid']}" if control else f"gold:{r['uid']}", "src": r["src"], "mt": r["mt"],
               "lp": LP}
        body = G.build_request("direct", row, args, None)
        rh = G.request_hash(body)
        if cache.get(f"direct|{row['id']}", rh):
            src = "cache_key"
        elif rh in by_hash:
            src = "cache_same_request"
        else:
            src = "call"
            jobs.append((row, body, rh))
        plan.append((r["uid"], row, rh, src))
    worst = sum(G.worst_cost(b) for _, b, _ in jobs)
    print(f"gemma direct: {len(rows)} units; {collections.Counter(p[3] for p in plan)}; "
          f"worst case ${worst:.4f} for {len(jobs)} calls; cap for this run ${a.cap:.2f}; "
          f"track spend before ${spent_before:.4f}", flush=True)
    if a.dry_run:
        if jobs:
            print(json.dumps(jobs[0][1], ensure_ascii=False, indent=1)[:800])
        return
    if worst > a.cap:
        sys.exit(f"worst case ${worst:.4f} > cap ${a.cap:.2f}")
    stats: collections.Counter = collections.Counter()
    if jobs:
        args.cap = spent_before + a.cap  # run_calls budgets args.cap - cache spend: the new spend only
        stats = asyncio.run(G.run_calls(jobs, "direct", cache, args))
        print(f"called {stats.get('called', 0)}, failed {stats.get('failed', 0)}, tokens in "
              f"{stats.get('tokens_in', 0):,} out {stats.get('tokens_out', 0):,}, "
              f"${stats.get('cost_micro', 0) / 1e6:.4f}, {stats.get('wall_s')} s", flush=True)
    out, missing = [], []
    for uid, row, rh, src in plan:
        e = cache.get(f"direct|{row['id']}", rh) or by_hash.get(rh)
        if e is None or e["req_hash"] != rh:
            missing.append(uid)
            continue
        sc = G.score_direct(e)
        out.append({"uid": uid, "risk": sc["risk"], "p_major": sc.get("p_major"), "p_ok": sc.get("p_ok"),
                    "status": sc.get("status"), "top": sc.get("top"), "model": e.get("model"),
                    "source": src, "cache_key": e["key"]})
    if missing:
        sys.exit(f"{len(missing)} units have no Gemma reply, e.g. {missing[:5]}; re-run to fill them")
    check_preds(out, [r["uid"] for r in rows])
    if control:
        out = [{"id": o.pop("uid"), **o} for o in out]
    write_jsonl(OUT / ("preds_control_ref_es_gemma.jsonl" if control else "preds_gemma_direct.jsonl"), out)
    new = stats.get("cost_micro", 0) / 1e6
    meta = {"system": "gemma_direct", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": G.DO_MODEL,
            "endpoint": G.DO_BASE_URL, "method": "direct (scripts/baseline_gemma.py), SecondRead classifier prompt",
            "inputs": str(a.inputs), "inputs_sha256": sha(a.inputs),
            "sources": dict(collections.Counter(p[3] for p in plan)),
            "status": dict(collections.Counter(o["status"] for o in out)),
            "called": stats.get("called", 0), "failed": stats.get("failed", 0),
            "tokens_in": stats.get("tokens_in", 0), "tokens_out": stats.get("tokens_out", 0),
            "new_spend_usd_list_price": round(new, 5), "cap_usd": a.cap, "worst_case_usd": round(worst, 5),
            "track_spend_before_usd": round(spent_before, 5),
            "thresholds_note": "review 0.1212 and f1 0.9714416886 come from Gemma's en-ru dev run "
                               "(scripts/final_test.py, docs/RESULTS.md); there is no Gemma en-es dev run"}
    (OUT / ("meta_gemma_control.json" if control else "meta_gemma.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("sources", "status", "called", "failed", "new_spend_usd_list_price")}),
          flush=True)


def cmd_freeze() -> None:
    """Record the sha256 of every prediction file and of the inputs before anything reads the labels."""
    man = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "note": "prediction files frozen before any scoring against labels.jsonl",
           "files": {f.name: {"sha256": sha(f), "lines": sum(1 for _ in f.open(encoding="utf-8"))}
                     for f in sorted(OUT.glob("preds_*.jsonl"))},
           "inputs": {str(p): sha(ROOT / p) for p in (
               Path("secondread/thresholds.json"), Path("secondread/rule_calibration.json"),
               Path("models/secondread-lora.gguf"), Path("runs/gold_eval/gold_inputs.jsonl"),
               Path("runs/gold_eval/gold_inputs_with_ref.jsonl"))}}
    path = OUT / "PREDICTIONS_MANIFEST.json"
    if path.exists():
        sys.exit(f"{path} exists: the predictions are already frozen")
    path.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", choices=["v1", "base", "control", "gemma", "gemma-control", "freeze"])
    ap.add_argument("--inputs", type=Path, default=OUT / "gold_inputs.jsonl")
    ap.add_argument("--llama-url", default=os.getenv("SECONDREAD_LLAMA_URL", "http://127.0.0.1:9931"))
    ap.add_argument("--cap", type=float, default=0.15, help="gemma: USD of NEW spend allowed in this run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.getenv("SECONDREAD_THRESHOLD") or os.getenv("SECONDREAD_OPERATING_POINT"):
        sys.exit("unset SECONDREAD_THRESHOLD / SECONDREAD_OPERATING_POINT: the shipped thresholds must be used")
    if a.system == "freeze":
        cmd_freeze()
    elif a.system in ("gemma", "gemma-control"):
        cmd_gemma(a, control=a.system == "gemma-control")
    else:
        cmd_backend(a, a.system)


if __name__ == "__main__":
    main()
