"""Build ui/public/demo-results.json: a small precomputed /api/review response for the static demo.

usage:
  # placeholder (now): simple rule checks + base-model score from a llama-server WITHOUT the LoRA
  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/make_demo_results.py --port 9931
  # real: send the same segments to the backend and store its answer (placeholder stays true
  # until the backend reports the LoRA as active)
  PYTHONPATH=. uv run python scripts/make_demo_results.py --api http://127.0.0.1:8787

The output follows the response shape in ui/API.md, plus two demo-only extras the UI understands:
  meta  - provenance, attribution and a `placeholder` flag shown as a banner
  gold  - the human MQM label for each segment (WMT22, Unbabel raters), shown under the segment

The segments are 10 consecutive units of one public WMT22 en-ru test document (one MT system,
one rater), in document order. Only a 10-segment slice is stored, with attribution.
"""
import argparse
import json
import math
import re
import time
import urllib.request
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="data/splits/en-ru.test.jsonl")
ap.add_argument("--doc", default="t1_hkc5no1")  # Reddit thread about a first restaurant job
ap.add_argument("--system", default="comet_bestmbr")  # its #10 drops a negated clause
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--port", type=int, default=9931, help="llama-server for the placeholder model score")
ap.add_argument("--api", help="SecondRead backend base URL; if set, use its real /api/review output")
ap.add_argument("--out", default="ui/public/demo-results.json")
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.split, encoding="utf-8")]
unit = lambda r: int(r["id"].split("|")[2])  # segment index inside the WMT test set
rows = sorted((r for r in rows if r["doc"] == a.doc and r["system"] == a.system), key=unit)[: a.n]
assert len(rows) == a.n, f"only {len(rows)} rows for {a.doc}/{a.system}"
assert unit(rows[-1]) - unit(rows[0]) == a.n - 1, "segments are not consecutive"
lp = rows[0]["lp"]


def gold(r):
    """Human annotation, kept separate from the model output; demo-only field."""
    return {"label": r["label"], "source": "WMT22 MQM",
            "spans": [{k: s[k] for k in ("side", "start", "end", "text", "category", "severity")}
                      for s in r["spans"]
                      if s["severity"].lower() in ("major", "critical") and s["family"] == "meaning"]}


meta = {
    "placeholder": True,
    "title": "Restaurant job thread (WMT22 en-ru, social domain)",
    "note": "Model risk here comes from the base Qwen3.5-4B (Q4_K_M) without the fine-tuned "
            "LoRA, and the number and negation checks are simple regex stand-ins.",
    "attribution": "Source text and MT output: WMT22 General MT test set, en-ru, system "
                   f"{a.system}. Human MQM labels: Unbabel raters, released by Google in "
                   "google/wmt-mqm-human-evaluation (Apache-2.0). WMT data is provided for "
                   "research purposes.",
    "lp": lp,
    "doc": a.doc,
    "system": a.system,
    "generated": time.strftime("%Y-%m-%d"),
}

if a.api:
    # Real path: whatever the backend says is the demo. Only the gold labels are added.
    body = {"lp": lp, "segments": [{"id": str(i + 1), "src": r["src"], "mt": r["mt"]} for i, r in enumerate(rows)]}
    req = urllib.request.Request(f"{a.api.rstrip('/')}/api/review", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    for seg, r in zip(out["segments"], rows):
        seg["gold"] = gold(r)
    lora = out.get("summary", {}).get("lora_active", False)
    meta.update(placeholder=not lora,
                note="Precomputed with the SecondRead backend, fine-tuned LoRA active." if lora else
                     "Precomputed with the SecondRead backend, but without the fine-tuned LoRA "
                     "(base Qwen3.5-4B, Q4_K_M).")
    out["meta"] = meta
else:
    from secondread.prompt import LABEL_IDS, generation_prompt_ids

    # --- placeholder stand-ins for the backend's rule checks -------------------------------
    NUM = re.compile(r"\d+(?:[.,]\d+)*")
    NEG = {
        "en": re.compile(r"\b(?:not|no|never|none|nor|without|nothing|nobody|neither)\b|\w+n[’']t\b", re.I),
        "ru": re.compile(r"(?<![\w-])(?:не|нет|ни|никогда|без|ничего|никто|нельзя)(?![\w-])", re.I),
        "es": re.compile(r"\b(?:no|nunca|jamás|ni|sin|nada|nadie|ningún|ninguna|tampoco)\b", re.I),
    }
    s_lang, t_lang = lp.split("-")

    def number_check(src, mt):
        norm = lambda x: x.replace(",", "").replace(".", "")  # 1,000 == 1000 (rough; placeholder)
        s = [(m, norm(m.group())) for m in NUM.finditer(src)]
        t = [(m, norm(m.group())) for m in NUM.finditer(mt)]
        sc, tc = Counter(v for _, v in s), Counter(v for _, v in t)
        src_only, mt_only = sc - tc, tc - sc
        spans = [{"side": "src", "start": m.start(), "end": m.end()} for m, v in s if v in src_only] + \
                [{"side": "mt", "start": m.start(), "end": m.end()} for m, v in t if v in mt_only]
        res = {"flagged": bool(src_only or mt_only), "src_only": sorted(src_only.elements()),
               "mt_only": sorted(mt_only.elements())}
        return res, spans

    def negation_check(src, mt):
        s, t = list(NEG[s_lang].finditer(src)), list(NEG[t_lang].finditer(mt))
        flagged = len(s) != len(t)
        side, ms = ("src", s) if len(s) > len(t) else ("mt", t)
        spans = [{"side": side, "start": m.start(), "end": m.end()} for m in ms] if flagged else []
        return {"flagged": flagged, "src_count": len(s), "mt_count": len(t)}, spans

    def model_score(src, mt):
        body = {"prompt": generation_prompt_ids(src, mt, lp), "n_predict": 1, "n_probs": 200,
                "temperature": 0, "cache_prompt": False}
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        res = json.load(urllib.request.urlopen(req, timeout=120))
        top = res["completion_probabilities"][0]
        lps = {c["id"]: c["logprob"] for c in top.get("top_logprobs", [])}
        pm, po = (math.exp(lps.get(LABEL_IDS[k], -30.0)) for k in ("Major", "OK"))
        return pm / (pm + po)

    # Noisy-OR fusion for the placeholder: each rule hit acts like independent evidence with p=0.6.
    RULE_P = 0.6
    MODEL_REASON_AT = 0.5
    generation_prompt_ids("warm", "up", lp)  # load the tokenizer before timing
    segments, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        num, num_spans = number_check(r["src"], r["mt"])
        neg, neg_spans = negation_check(r["src"], r["mt"])
        lora = model_score(r["src"], r["mt"])
        risk = 1 - (1 - lora) * (1 - RULE_P) ** (num["flagged"] + neg["flagged"])
        reasons = []
        if num["flagged"]:
            parts = [f"{', '.join(num['src_only'])} only in source" if num["src_only"] else "",
                     f"{', '.join(num['mt_only'])} only in MT" if num["mt_only"] else ""]
            reasons.append({"code": "number_mismatch", "message": "; ".join(p for p in parts if p),
                            "spans": num_spans})
        if neg["flagged"]:
            reasons.append({"code": "negation_mismatch",
                            "message": f"{neg['src_count']} negation(s) in source, {neg['mt_count']} in MT",
                            "spans": neg_spans})
        if lora >= MODEL_REASON_AT:
            reasons.append({"code": "model_risk", "message": f"Model P(Major) {lora:.0%}", "spans": []})
        segments.append({"id": str(i + 1), "src": r["src"], "mt": r["mt"], "risk": round(risk, 4),
                         "signals": {"number_check": num, "negation_check": neg, "lora_score": round(lora, 4)},
                         "reasons": reasons, "gold": gold(r)})
    out = {"lp": lp, "segments": segments, "timing_ms": round((time.perf_counter() - t0) * 1000)}
    out["meta"] = meta

with open(a.out, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
    f.write("\n")
for s in out["segments"]:
    print(f"{s['id']:>3} risk={s['risk']:.3f} lora={s['signals'].get('lora_score')} "
          f"reasons={[x['code'] if isinstance(x, dict) else x for x in s['reasons']]} gold={s['gold']['label']}")
print("wrote", a.out)
