"""Time a running SecondRead server on real segments (G9) and check the fused risk ranks well.

usage: uv run python scripts/bench_review.py data/splits/en-ru.dev.jsonl N [--url http://127.0.0.1:8787]
       [--batch 25] [--lora-scale S]
Posts N random segments (seed 0) to /api/review in batches and prints per-segment latency and
AUROC / PR-AUC of the fused risk vs the model-only score (labels from the split).
"""
import argparse
import json
import random
import statistics
import time
import urllib.request

from secondread.metrics import auroc, average_precision

ap = argparse.ArgumentParser()
ap.add_argument("split")
ap.add_argument("n", type=int)
ap.add_argument("--url", default="http://127.0.0.1:8787")
ap.add_argument("--batch", type=int, default=25)
ap.add_argument("--lora-scale", type=float, default=None)
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.split, encoding="utf-8")]
random.Random(0).shuffle(rows)
rows = rows[: a.n]

wall, per_seg, calls, fused, model, labels = [], [], [], [], [], []
for i in range(0, len(rows), a.batch):
    chunk = rows[i:i + a.batch]
    body = {"segments": [{"id": r["id"], "src": r["src"], "mt": r["mt"], "lp": r["lp"]} for r in chunk],
            "lora_scale": a.lora_scale}
    req = urllib.request.Request(f"{a.url}/api/review", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t = time.perf_counter()
    res = json.load(urllib.request.urlopen(req, timeout=600))
    wall.append(time.perf_counter() - t)
    tm = res["summary"]["timing_ms"]
    per_seg.append(tm["per_segment"])
    calls += [s["model"]["latency_ms"] for s in res["segments"] if s["model"]["error"] is None]
    for r, s in zip(chunk, res["segments"]):
        labels.append(int(r["label"] == "Major"))
        fused.append(s["risk"])
        model.append(s["signals"]["lora_score"] if s["signals"]["lora_score"] is not None else 0.0)

n = len(rows)
print(f"n={n} batches={len(wall)} batch={a.batch} model={res['summary']['model']} lora_active={res['summary']['lora_active']}")
print(f"wall per segment (incl. HTTP): {sum(wall) / n * 1000:.0f} ms;  server per segment: "
      f"mean {statistics.mean(per_seg):.0f} ms;  model call p50 {statistics.median(calls):.0f} ms, "
      f"p95 {sorted(calls)[int(0.95 * len(calls)) - 1]:.0f} ms")
base = sum(labels) / n
print(f"Major rate {base:.3f} | AUROC fused {auroc(labels, fused):.3f} vs model {auroc(labels, model):.3f} | "
      f"PR-AUC fused {average_precision(labels, fused):.3f} vs model {average_precision(labels, model):.3f}")
print(f"flagged at 0.5: fused {sum(x >= 0.5 for x in fused)}, model {sum(x >= 0.5 for x in model)} of {n}")
