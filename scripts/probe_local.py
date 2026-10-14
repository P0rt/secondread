"""Score units with a local llama-server and report P(Major), quality and latency.

usage: uv run python scripts/probe_local.py SPLIT.jsonl N [--lora-scale S] [--port 9931] [--out FILE]
Sends renderer-built token ids to /completion and reads the first-token
distribution. Risk = p(Major) / (p(Major) + p(OK)).
"""
import argparse, json, math, random, statistics, time, urllib.request
from secondread.metrics import auroc, pr_auc, recall_at, to_arrays
from secondread.prompt import LABEL_IDS, generation_prompt_ids

ap = argparse.ArgumentParser()
ap.add_argument("split"); ap.add_argument("n", type=int)
ap.add_argument("--lora-scale", type=float, default=None)
ap.add_argument("--port", type=int, default=9931)
ap.add_argument("--n-probs", type=int, default=200)
ap.add_argument("--out")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.split, encoding="utf-8")]
random.Random(0).shuffle(rows)
rows = rows[: a.n]


def call(ids):
    body = {"prompt": ids, "n_predict": 1, "n_probs": a.n_probs, "temperature": 0, "cache_prompt": False}
    if a.lora_scale is not None:
        body["lora"] = [{"id": 0, "scale": a.lora_scale}]
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t = time.perf_counter()
    res = json.load(urllib.request.urlopen(req, timeout=120))
    return res, time.perf_counter() - t


def label_logprobs(res):
    top = res["completion_probabilities"][0]
    cand = top.get("top_logprobs") or top.get("probs") or []
    lp = {}
    for c in cand:
        tid = c.get("id")
        val = c["logprob"] if "logprob" in c else math.log(max(c.get("prob", 0), 1e-30))
        for name, i in LABEL_IDS.items():
            if tid == i:
                lp[name] = val
    return lp


scored, lat, missing = [], [], 0
for r in rows:
    res, dt = call(generation_prompt_ids(r["src"], r["mt"], r["lp"]))
    lat.append(dt)
    lp = label_logprobs(res)
    if len(lp) < 2:
        missing += 1
    pm = math.exp(lp.get("Major", -30.0)); po = math.exp(lp.get("OK", -30.0))
    scored.append({"id": r["id"], "label": r["label"], "risk": pm / (pm + po), "p_major": pm, "p_ok": po,
                   "top": res["completion_probabilities"][0].get("token"),
                   "doc": r.get("doc"), "lp": r.get("lp")})  # doc/lp: grouping for bootstrap CIs


base = sum(x["label"] == "Major" for x in scored) / len(scored)
print(f"n={len(scored)} Major-rate={base:.3f} labels-missing-from-top{a.n_probs}={missing}")
y, s = to_arrays(scored)
print(f"AUROC={auroc(y, s):.3f} PR-AUC={pr_auc(y, s):.3f} (random={base:.3f}) recall@10%={recall_at(y, s):.3f} (random=0.100)")
print(f"top-1 tokens: {dict(__import__('collections').Counter(x['top'] for x in scored).most_common(5))}")
print(f"latency p50={statistics.median(lat)*1000:.0f}ms p95={sorted(lat)[int(0.95*len(lat))-1]*1000:.0f}ms")
if a.out:
    with open(a.out, "w") as f:
        for x in scored:
            f.write(json.dumps(x) + "\n")
