"""Score the real-document MT (and the WMT24 demo docs) with SecondRead v1, the rule checks and Gemma 4 direct.

usage: set -a; . ./.env; set +a   # only for the Gemma calls (DO_MODEL_KEY, never printed)
       HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python scripts/realdocs_score.py \
           [--gemma-cap 0.30] [--no-gemma] [--gemma-dry-run]

Inputs
  data/realdocs/<doc>/mt_<system>.jsonl   from scripts/realdocs_mt.py (marian, nllb)
  data/realdocs/<doc>/aligned.jsonl       the official Spanish (ref_es) is scored too, as system
                                          "official_es": a control for false alarms on a human translation
  data/realdocs/wmt24_*/aligned.jsonl     WMT24 en-es MT (system MSLC) with human MQM gold labels

Signals per segment (data/realdocs/scored.jsonl)
  secondread  secondread.agent.review_sync (the backend pipeline, used read-only): model risk
              p(Major)/(p(Major)+p(OK)) from llama-server (v1 LoRA) on the secondread.prompt
              prompt with lp "en-es", fused with the rule checks by the train-calibrated log-odds
              weights, flagged at the v1 en-es threshold of secondread/thresholds.json
  checks      secondread.checks.number_check / negation_check called directly (codes, reasons, spans)
  gemma       Gemma 4 31B (DigitalOcean) with the same classifier prompt, risk from the top-10
              logprobs (scripts/baseline_gemma.py, direct method, its cache data/cache/gemma_baseline.jsonl);
              flagged at 0.971 (the repo's Gemma direct threshold, from en-ru dev) and at 0.5.
              Not run on official_es. The cap applies to NEW spend of this run only.
Summary per (doc, system): data/realdocs/scored_summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from secondread import checks  # noqa: E402

REAL = ROOT / "data" / "realdocs"
LP = "en-es"
DOCS = ["eu-air-passenger-rights", "niddk-hypoglycemia", "ready-hurricanes",
        "irs-failure-to-file-penalty", "fda-medicine-disposal"]
MT_SYSTEMS = ["marian", "nllb"]
WMT = ["wmt24_forever_snow", "wmt24_the_other_side_stormfall"]
GEMMA_THRESHOLD = 0.971  # docs/RESULTS.md: Gemma direct, F1-optimal on en-ru dev (no en-es dev run)
PUBLIC = {"marian": True, "nllb": False, "official_es": True, "MSLC": True}


def load_gemma_module():
    spec = importlib.util.spec_from_file_location("baseline_gemma", ROOT / "scripts" / "baseline_gemma.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect() -> list[dict]:
    rows = []
    for doc in DOCS:
        for system in MT_SYSTEMS:
            for l in open(REAL / doc / f"mt_{system}.jsonl", encoding="utf-8"):
                r = json.loads(l)
                rows.append({"doc_id": doc, "system": system, "seg_id": r["id"], "section": r.get("section"),
                             "src": r["src"], "mt": r["mt"], "ref_es": r["ref_es"], "gold_label": None,
                             "gemma_key": f"realdocs:{system}:{r['id']}"})
        for l in open(REAL / doc / "aligned.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rows.append({"doc_id": doc, "system": "official_es", "seg_id": r["id"], "section": r.get("section"),
                         "src": r["src"], "mt": r["ref_es"], "ref_es": r["ref_es"], "gold_label": None,
                         "gemma_key": None})
    for doc in WMT:
        for l in open(REAL / doc / "aligned.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rows.append({"doc_id": doc, "system": r["system"], "seg_id": r["id"], "section": None,
                         "src": r["src"], "mt": r["mt"], "ref_es": r["ref_es"], "gold_label": r["gold_label"],
                         "gemma_key": r["wmt_id"]})  # same id as results/gemma_direct_enes_test.jsonl
    return rows


def run_secondread(rows: list[dict]) -> tuple[list, dict]:
    from secondread.agent import review_sync
    res = review_sync([{"id": f"{r['doc_id']}|{r['system']}|{r['seg_id']}", "src": r["src"], "mt": r["mt"]}
                       for r in rows], lp=LP)
    return res.segments, res.summary


def run_checks(r: dict) -> dict:
    out = {}
    for fn in (checks.number_check, checks.negation_check):
        c = fn(r["src"], r["mt"], LP)
        out[c.name] = {"codes": c.codes, "skipped": c.skipped, "prior": c.risk,
                       "issues": [{"code": i.code, "reason": i.reason, "spans": [list(s) for s in i.spans]}
                                  for i in c.issues]}
    return out


def run_gemma(rows: list[dict], cap: float, dry_run: bool) -> tuple[dict, dict]:
    G = load_gemma_module()
    args = SimpleNamespace(direct_max_tokens=3, top_logprobs=10, concurrency=8, rpm=110,
                           base_url=G.DO_BASE_URL, progress=100)
    cache = G.Cache(G.CACHE_PATH)
    spent_before = cache.total_cost
    items = [r for r in rows if r["gemma_key"]]
    jobs, hits = [], 0
    for r in items:
        row = {"id": r["gemma_key"], "src": r["src"], "mt": r["mt"], "lp": LP}
        body = G.build_request("direct", row, args, None)
        rh = G.request_hash(body)
        if cache.get(f"direct|{row['id']}", rh):
            hits += 1
        else:
            jobs.append((row, body, rh))
    worst = sum(G.worst_cost(b) for _, b, _ in jobs)
    print(f"gemma direct: {len(items)} rows, {hits} cached, {len(jobs)} to call, worst case ${worst:.4f}, "
          f"cap for this run ${cap:.2f}", flush=True)
    if worst > cap:
        raise SystemExit(f"gemma worst case ${worst:.4f} > cap ${cap:.2f}")
    stats: collections.Counter = collections.Counter()
    if jobs and not dry_run:
        args.cap = spent_before + cap  # run_calls budgets args.cap - cache spend: the new spend only
        stats = asyncio.run(G.run_calls(jobs, "direct", cache, args))
    new_spend = stats["cost_micro"] / 1e6 if stats else 0.0
    out = {}
    for r in items:
        row = {"id": r["gemma_key"], "src": r["src"], "mt": r["mt"], "lp": LP}
        e = cache.get(f"direct|{row['id']}", G.request_hash(G.build_request("direct", row, args, None)))
        if e is not None:
            out[r["gemma_key"]] = G.score_direct(e) | {"model": e.get("model")}
    meta = {"model": G.DO_MODEL, "endpoint": G.DO_BASE_URL, "method": "direct (scripts/baseline_gemma.py)",
            "rows": len(items), "cached_before": hits, "called": stats.get("called", 0),
            "failed": stats.get("failed", 0), "new_spend_usd": round(new_spend, 5), "cap_usd": cap,
            "threshold": GEMMA_THRESHOLD, "threshold_note": "Gemma direct F1-optimal threshold on en-ru dev "
            "(docs/RESULTS.md); there is no Gemma en-es dev run"}
    return out, meta


def summarize(scored: list[dict], thr: float) -> list[dict]:
    groups: dict[tuple, list] = collections.defaultdict(list)
    for s in scored:
        groups[(s["doc_id"], s["system"])].append(s)
    out = []
    for (doc, system), xs in groups.items():
        sr = [x["secondread"] for x in xs]
        risks = [x["risk"] for x in sr if x["risk"] is not None]
        mrisk = [x["model_risk"] for x in sr if x["model_risk"] is not None]
        gem = [x["gemma"]["risk"] for x in xs if x.get("gemma")]
        e = {"doc_id": doc, "system": system, "public_demo": PUBLIC.get(system, False), "segments": len(xs),
             "threshold": thr, "flagged": sum(x["flagged"] for x in sr),
             "flagged_model_only": sum(x["flagged_model_only"] for x in sr),
             "rule_hits": sum(bool(x["checks"]["number_check"]["codes"] or x["checks"]["negation_check"]["codes"])
                              for x in xs),
             "number_hits": sum(bool(x["checks"]["number_check"]["codes"]) for x in xs),
             "negation_hits": sum(bool(x["checks"]["negation_check"]["codes"]) for x in xs),
             "mean_risk": round(statistics.mean(risks), 4) if risks else None,
             "mean_model_risk": round(statistics.mean(mrisk), 4) if mrisk else None,
             "model_errors": sum(x["model_error"] is not None for x in sr),
             "gemma_scored": len(gem),
             "gemma_mean_risk": round(statistics.mean(gem), 4) if gem else None,
             "gemma_flagged_0971": sum(r >= GEMMA_THRESHOLD for r in gem) if gem else None,
             "gemma_flagged_05": sum(r >= 0.5 for r in gem) if gem else None,
             "flagged_ids": [x["id"] for x, s in zip(xs, sr) if s["flagged"]]}
        if any(x["gold_label"] for x in xs):
            y = [x["gold_label"] == "Major" for x in xs]
            f = [s["flagged"] for s in sr]
            tp = sum(a and b for a, b in zip(y, f))
            e["gold"] = {"major": sum(y), "tp": tp, "precision": round(tp / max(1, sum(f)), 4),
                         "recall": round(tp / max(1, sum(y)), 4)}
            if gem:
                gf = [x["gemma"]["risk"] >= GEMMA_THRESHOLD for x in xs]
                gtp = sum(a and b for a, b in zip(y, gf))
                e["gold"]["gemma_0971"] = {"flagged": sum(gf), "tp": gtp,
                                           "precision": round(gtp / max(1, sum(gf)), 4),
                                           "recall": round(gtp / max(1, sum(y)), 4)}
        out.append(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-cap", type=float, default=0.30)
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--gemma-dry-run", action="store_true")
    ap.add_argument("--out", default=str(REAL / "scored.jsonl"))
    ap.add_argument("--summary", default=str(REAL / "scored_summary.json"))
    a = ap.parse_args()

    rows = collect()
    print(f"{len(rows)} segments: {dict(collections.Counter((r['system']) for r in rows))}", flush=True)
    t0 = time.perf_counter()
    segs, summary = run_secondread(rows)
    print(f"secondread: {summary['segments']} segments, {summary['flagged']} flagged, model {summary['model']}, "
          f"adapter {summary['adapter_version']}, thresholds {summary['thresholds']} "
          f"({summary['threshold_source']}), errors {summary['model_errors']}, {time.perf_counter() - t0:.0f} s",
          flush=True)
    thr = summary["thresholds"][LP]
    gemma, gmeta = ({}, {"skipped": True}) if a.no_gemma else run_gemma(rows, a.gemma_cap, a.gemma_dry_run)

    scored = []
    for r, s in zip(rows, segs, strict=True):
        assert s.id == f"{r['doc_id']}|{r['system']}|{r['seg_id']}"
        m = s.model
        model_risk = s.signals.get("lora_score")
        g = gemma.get(r["gemma_key"]) if r["gemma_key"] else None
        scored.append({
            "doc_id": r["doc_id"], "system": r["system"], "id": r["seg_id"], "section": r["section"], "lp": LP,
            "public_demo": PUBLIC.get(r["system"], False),
            "src": r["src"], "mt": r["mt"], "ref_es": r["ref_es"], "gold_label": r["gold_label"],
            "secondread": {"risk": s.risk, "model_risk": round(model_risk, 4) if model_risk is not None else None,
                           "p_major": m.get("p_major"), "p_ok": m.get("p_ok"),
                           "labels_in_top": m.get("labels_in_top"), "model_error": m.get("error"),
                           "flagged": s.flagged,
                           "flagged_model_only": model_risk is not None and model_risk >= thr,
                           "threshold": thr, "issues": s.issues, "reasons": s.reasons,
                           "rule_weights": {k: v.get("weight") for k, v in s.signals.items()
                                            if isinstance(v, dict) and "weight" in v}},
            "checks": run_checks(r),
            "gemma": ({"risk": round(g["risk"], 4), "p_major": g.get("p_major"), "p_ok": g.get("p_ok"),
                       "status": g.get("status"), "top": g.get("top"),
                       "flagged_0971": g["risk"] >= GEMMA_THRESHOLD, "flagged_05": g["risk"] >= 0.5}
                      if g else None),
        })
    with open(a.out, "w", encoding="utf-8") as f:
        for x in scored:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summ = summarize(scored, thr)
    meta = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "lp": LP,
            "secondread": {k: summary.get(k) for k in ("model", "adapter_version", "thresholds", "threshold_source",
                                                          "lora_active", "model_errors", "timing_ms")},
            "gemma": gmeta, "mt_meta": "data/realdocs/mt_meta.json",
            "systems_note": "marian = Helsinki-NLP/opus-mt-en-es (Apache-2.0, public demo); nllb = "
                            "facebook/nllb-200-distilled-600M (CC-BY-NC-4.0, research comparison only, not in the "
                            "public demo); official_es = the publisher's own Spanish (control); MSLC = WMT24 system "
                            "with human MQM labels",
            "per_doc_system": summ}
    Path(a.summary).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(scored)} rows) and {a.summary}")
    for e in summ:
        print(f"  {e['doc_id']:<32} {e['system']:<12} n={e['segments']:<4} flagged={e['flagged']:<3} "
              f"model-only={e['flagged_model_only']:<3} rules={e['rule_hits']:<3} mean_risk={e['mean_risk']} "
              f"gemma>=0.971={e['gemma_flagged_0971']} {e.get('gold', '')}")


if __name__ == "__main__":
    main()
