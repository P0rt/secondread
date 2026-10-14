"""G1 (Tinker models) and G6 (DigitalOcean Gemma) smoke checks. Never prints key values."""
import asyncio, json, os, time, urllib.request

def load_env(path=".env"):
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v:
                os.environ.setdefault(k, v)

load_env()

async def g1():
    import tinker
    sc = tinker.ServiceClient()
    caps = await sc.get_server_capabilities_async()
    models = [getattr(m, "model_name", str(m)) for m in caps.supported_models]
    print("G1 supported models:", len(models))
    for m in models:
        if "Qwen3" in m or "qwen" in m.lower():
            print("   ", m)
    print("G1 Qwen/Qwen3.5-4B available:", any(m == "Qwen/Qwen3.5-4B" for m in models))

def g6():
    base = os.environ.get("DO_BASE_URL", "https://inference.do-ai.run/v1")
    hdr = {"Authorization": "Bearer " + os.environ["DO_MODEL_KEY"], "Content-Type": "application/json"}
    req = urllib.request.Request(base + "/models", headers=hdr)
    ids = [m["id"] for m in json.load(urllib.request.urlopen(req, timeout=30))["data"]]
    print("G6 models:", len(ids), "| gemma:", [i for i in ids if "gemma" in i.lower()])
    body = {"model": "gemma-4-31B-it", "max_completion_tokens": 16, "temperature": 0, "logprobs": True, "top_logprobs": 5,
            "messages": [{"role": "user", "content": "EN: Do not pay the invoice.\nRU: Оплатите счёт.\nMajor meaning error? Answer Major or OK."}]}
    t = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=hdr), timeout=60)
    res = json.load(r)
    ch = res["choices"][0]
    print(f"G6 chat {r.status} in {time.perf_counter()-t:.2f}s -> {ch['message'].get('content')!r}",
          "| reasoning field:", bool(ch["message"].get("reasoning_content")),
          "| logprobs:", "present" if ch.get("logprobs") else "null", "| usage:", res.get("usage"))
    print("G6 rate-limit headers:", {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")})

try:
    asyncio.run(g1())
except Exception as e:
    print("G1 FAILED:", type(e).__name__, str(e)[:300])
try:
    g6()
except Exception as e:
    print("G6 FAILED:", type(e).__name__, str(e)[:300], getattr(e, "read", lambda: b"")()[:300])
