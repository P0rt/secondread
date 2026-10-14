"""Machine-translate every segment of the frozen gold corpus (data/realdocs/gold/DOC_LIST.json) with two MT systems.

usage:
  OPUS-MT (local):  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen --with sentencepiece==0.2.1 --with sacremoses==0.1.1 \
                        python scripts/gold_mt.py --system marian [--device mps|cpu]
  LLM (Tinker):     set -a; . ./.env; set +a; HF_HUB_OFFLINE=0 PYTHONPATH=. uv run --frozen \
                        python scripts/gold_mt.py --system deepseek-v3.1 [--pilot N] [--cap 0.40] [--dry-run]

Writes data/realdocs/gold/mt/<system>/<doc_id>.jsonl, one row per aligned segment in aligned order:
{id, src, mt, ref_es, section, lp, system}, and the run's settings to data/realdocs/gold/mt/MT_META.json.
Every row of every listed document is translated; nothing is skipped because of its content.

marian    Helsinki-NLP/opus-mt-en-es at the revision pinned in scripts/realdocs_mt.py, with that script's
          own code (split_sentences, load, translate: one sentence per input, beam 4). For the five original
          documents the outputs of scripts/realdocs_mt.py (data/realdocs/<doc>/mt_marian.jsonl, same model,
          same settings) are reused after checking that id, src and ref_es match aligned.jsonl row by row.

deepseek-v3.1
          deepseek-ai/DeepSeek-V3.1 (open weights, MIT licence, 671B-parameter MoE), sampled on Tinker with
          the tinker_cookbook renderer 'deepseekv3_disable_thinking' (non-thinking chat mode). DigitalOcean
          serverless inference was the planned host, but on 2026-10-14 the project's model key was refused
          (HTTP 403) for every open-weight model except Gemma (llama3.3-70b-instruct, llama-4-maverick,
          qwen3.5-397b-a17b, alibaba-qwen3-32b, deepseek-*, glm-*, kimi-*, mistral-*, openai-gpt-oss-*,
          nemotron-*), and its router endpoint picks a different model per request, so no pinned non-Gemma
          open-weight model was reachable there. Not Gemma (the Gemma 4 baseline judge) and not Qwen (the
          SecondRead base model family), so the MT system shares a model family with neither evaluated system.
          CAT-style MT step: one segment per request, the previous two source segments of the same document
          (aligned order) given as context only, plain professional EN->ES instruction (neutral Spanish),
          temperature 0 (greedy), seed 0. Replies are cached in data/realdocs/gold/mt/cache_<system>.jsonl
          keyed by a hash of the request; reruns are free.
          Budget: --cap (USD, default 0.40) covers the whole step including cached spend. Cost is counted from
          the token counts of each request (prompt tokens at --price-in, generated tokens at --price-out, USD per
          1M; defaults are deliberately above the published Tinker rate for this model). Before any call the
          script projects the spend (expected output = 2 x source tokens + 16) and refuses to start if cache
          spend plus the projection exceeds the cap; while running, each request reserves its worst case
          (max_tokens) and nothing new is sent once spend plus reservations reach the cap.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "data" / "realdocs" / "gold"
MT_DIR = GOLD / "mt"
META = MT_DIR / "MT_META.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

LLM = {
    "deepseek-v3.1": {
        "model_id": "deepseek-ai/DeepSeek-V3.1",
        "hf_revision_checked": "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
        "license": "mit",
        "license_evidence": "Hugging Face model card front matter 'license: mit' (API, revision c0781d03, "
                            "checked 2026-10-14); weights not gated",
        "host": "Tinker (Thinking Machines) sampling API, base model weights",
        "renderer": "deepseekv3_disable_thinking",
        "architecture": "DeepSeek-V3.1, 671B-parameter MoE (37B active), non-thinking chat mode",
        "public_demo": True,
        "note": "MIT: outputs may be shown in the public demo with attribution to DeepSeek-V3.1.",
    },
}

SYSTEM_PROMPT = ("You are a professional translator. Translate English into neutral Spanish (español neutro). "
                 "Reply with the translation of the segment only, without notes or explanations.")
NO_CONTEXT = "(none: this is the first segment of the document)"


def user_prompt(context: list[str], src: str) -> str:
    ctx = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(context)) if context else NO_CONTEXT
    return (f"Previous segments, for context only (do not translate them):\n{ctx}\n\n"
            f"Translate this segment:\n{src}")


def load_docs() -> list[dict]:
    lst = json.loads((GOLD / "DOC_LIST.json").read_text(encoding="utf-8"))
    out = []
    for d in lst["documents"]:
        p = REPO / d["aligned"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != d["aligned_sha256"]:
            sys.exit(f"{p} does not match the frozen DOC_LIST.json sha256")
        out.append({**d, "rows": [json.loads(line) for line in p.open(encoding="utf-8")]})
    return out


def write_rows(system: str, doc_id: str, rows: list[dict], mts: list[str]) -> None:
    (MT_DIR / system).mkdir(parents=True, exist_ok=True)
    with (MT_DIR / system / f"{doc_id}.jsonl").open("w", encoding="utf-8") as f:
        for r, mt in zip(rows, mts, strict=True):
            f.write(json.dumps({"id": r["id"], "src": r["src"], "mt": mt, "ref_es": r["ref_es"],
                                "section": r.get("section"), "lp": "en-es", "system": system},
                               ensure_ascii=False) + "\n")


def update_meta(system: str, entry: dict) -> None:
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {"systems": {}}
    meta["systems"][system] = entry
    meta["corpus"] = "data/realdocs/gold/DOC_LIST.json"
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- marian

def run_marian(docs: list[dict], device: str | None) -> None:
    import torch
    import realdocs_mt as rm  # the original script's code, model pin and decoding settings

    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    reused, fresh, n_sent, t0 = [], [], 0, time.perf_counter()
    model = None
    for d in docs:
        rows = d["rows"]
        old = REPO / "data" / "realdocs" / d["doc_id"] / "mt_marian.jsonl"
        if d["origin"] == "realdocs-v1" and old.exists():
            prev = [json.loads(line) for line in old.open(encoding="utf-8")]
            same = len(prev) == len(rows) and all(
                (p["id"], p["src"], p["ref_es"]) == (r["id"], r["src"], r["ref_es"]) for p, r in zip(prev, rows))
            if not same:
                sys.exit(f"{old} does not match {d['aligned']}; refusing to reuse")
            write_rows("marian", d["doc_id"], rows, [p["mt"] for p in prev])
            reused.append(d["doc_id"])
            print(f"marian {d['doc_id']}: {len(rows)} segments reused from {old.relative_to(REPO)}", flush=True)
            continue
        if model is None:
            tok, model, extra = rm.load("marian", dev)
        parts = [rm.split_sentences(r["src"]) for r in rows]
        flat = [s for p in parts for s in p]
        out = rm.translate(flat, tok, model, extra, dev)
        mts, k = [], 0
        for p in parts:
            mts.append(" ".join(out[k:k + len(p)]))
            k += len(p)
        write_rows("marian", d["doc_id"], rows, mts)
        fresh.append(d["doc_id"])
        n_sent += len(flat)
        print(f"marian {d['doc_id']}: {len(rows)} segments, {len(flat)} sentences translated", flush=True)
    import transformers
    old_meta = json.loads((REPO / "data" / "realdocs" / "mt_meta.json").read_text())["systems"]["marian"]
    update_meta("marian", {
        **rm.SYSTEMS["marian"], "decoding": rm.GEN, "batch_size": rm.BATCH,
        "segmentation": "each aligned segment split into sentences (scripts/realdocs_mt.py split_sentences), "
                        "one sentence per MT input, outputs joined with one space",
        "reused_from_realdocs_mt": {doc: f"data/realdocs/{doc}/mt_marian.jsonl" for doc in reused},
        "reused_run": {k: old_meta.get(k) for k in ("device", "dtype", "torch", "transformers", "ts")},
        "translated_here": fresh, "device": dev, "dtype": "float32", "sentences_translated_here": n_sent,
        "wall_s": round(time.perf_counter() - t0, 1), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "machine": platform.machine(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "segments": sum(len(d["rows"]) for d in docs),
        "files": [f"data/realdocs/gold/mt/marian/{d['doc_id']}.jsonl" for d in docs],
    })


# ---------------------------------------------------------------- LLM on Tinker

class Cache:
    def __init__(self, path: Path):
        self.path, self.rows = path, {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                e = json.loads(line)
                self.rows[e["req_hash"]] = e

    def put(self, e: dict) -> None:
        self.rows[e["req_hash"]] = e
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    @property
    def spent(self) -> float:
        return sum(e["cost_usd"] for e in self.rows.values())


def run_llm(docs: list[dict], a) -> None:
    import tinker
    from tinker_cookbook import renderers, tokenizer_utils

    cfg = LLM[a.system]
    tok = tokenizer_utils.get_tokenizer(cfg["model_id"])
    ren = renderers.get_renderer(cfg["renderer"], tok)
    stop = ren.get_stop_sequences()

    jobs = []  # (doc_id, row_index, request dict)
    for d in docs:
        srcs = [r["src"] for r in d["rows"]]
        for i, r in enumerate(d["rows"]):
            ctx = srcs[max(0, i - 2):i]
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(ctx, r["src"])}]
            prompt = ren.build_generation_prompt(msgs)
            ids = prompt.to_ints()
            n_src = len(tok.encode(r["src"], add_special_tokens=False))
            max_tokens = min(1024, 3 * n_src + 64)
            body = {"model": cfg["model_id"], "renderer": cfg["renderer"], "prompt_ids": ids,
                    "temperature": 0.0, "seed": 0, "max_tokens": max_tokens, "stop": stop}
            rh = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            jobs.append({"doc_id": d["doc_id"], "i": i, "id": r["id"], "prompt": prompt, "n_prompt": len(ids),
                         "n_src": n_src, "max_tokens": max_tokens, "req_hash": rh, "messages": msgs})
    if a.pilot:
        jobs = jobs[:a.pilot]
    MT_DIR.mkdir(parents=True, exist_ok=True)
    cache = Cache(MT_DIR / f"cache_{a.system}.jsonl")
    todo = [j for j in jobs if j["req_hash"] not in cache.rows]
    pin, pout = a.price_in / 1e6, a.price_out / 1e6
    proj = sum(j["n_prompt"] * pin + (2 * j["n_src"] + 16) * pout for j in todo)
    worst = sum(j["n_prompt"] * pin + j["max_tokens"] * pout for j in todo)
    print(f"{a.system}: {len(jobs)} requests, {len(jobs) - len(todo)} cached (spent ${cache.spent:.4f}), "
          f"{len(todo)} to send; prompt tokens {sum(j['n_prompt'] for j in todo):,}; projected ${proj:.4f} "
          f"(worst case ${worst:.4f}) at ${a.price_in}/M in, ${a.price_out}/M out; cap ${a.cap}", flush=True)
    if cache.spent + proj > a.cap:
        sys.exit("projected spend exceeds the cap; not starting")
    if a.dry_run:
        print("dry run: no request sent")
        return

    async def go() -> None:
        service = tinker.ServiceClient(user_metadata={"purpose": "secondread_gold_mt", "system": a.system})
        sc = await service.create_sampling_client_async(base_model=cfg["model_id"])
        sem = asyncio.Semaphore(a.concurrency)
        state = {"reserved": 0.0, "spent": cache.spent, "stopped": 0}

        async def one(j: dict) -> None:
            async with sem:
                worst_j = j["n_prompt"] * pin + j["max_tokens"] * pout
                if state["spent"] + state["reserved"] + worst_j > a.cap:
                    state["stopped"] += 1
                    return
                state["reserved"] += worst_j
                t = time.perf_counter()
                for attempt in range(4):
                    try:
                        res = await sc.sample_async(prompt=j["prompt"], num_samples=1, sampling_params=tinker.SamplingParams(
                            max_tokens=j["max_tokens"], temperature=0.0, seed=0, stop=stop))
                        break
                    except Exception as e:  # transient API errors: back off and retry
                        if attempt == 3:
                            state["reserved"] -= worst_j
                            print(f"  FAILED {j['id']}: {type(e).__name__}: {e}", flush=True)
                            return
                        await asyncio.sleep(2 ** attempt * 3)
                seq = res.sequences[0]
                msg, term = ren.parse_response(seq.tokens)
                text = msg["content"] if isinstance(msg["content"], str) else renderers.get_text_content(msg)
                n_out = len(seq.tokens)
                cost = j["n_prompt"] * pin + n_out * pout
                state["reserved"] -= worst_j
                state["spent"] += cost
                cache.put({"req_hash": j["req_hash"], "system": a.system, "doc_id": j["doc_id"], "id": j["id"],
                           "mt_raw": text, "termination": str(term), "stop_reason": str(getattr(seq, "stop_reason", "")),
                           "n_prompt": j["n_prompt"], "n_out": n_out, "cost_usd": cost,
                           "secs": round(time.perf_counter() - t, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

        await asyncio.gather(*(one(j) for j in todo))
        print(f"sent; spend now ${state['spent']:.4f} (list-price estimate); {state['stopped']} not sent (cap)",
              flush=True)

    t0 = time.perf_counter()
    if todo:
        asyncio.run(go())
    missing = [j for j in jobs if j["req_hash"] not in cache.rows]
    if a.pilot or missing:
        for j in jobs[:a.pilot or 0]:
            e = cache.rows.get(j["req_hash"])
            print(f"  {j['id']}: {e['mt_raw'] if e else None!r} ({e and e['n_prompt']}+{e and e['n_out']} tok)")
        if missing:
            print(f"{len(missing)} requests still missing; outputs not written")
        return
    by_doc: dict[str, list[str]] = {}
    for j in jobs:
        by_doc.setdefault(j["doc_id"], []).append(cache.rows[j["req_hash"]]["mt_raw"].strip())
    for d in docs:
        write_rows(a.system, d["doc_id"], d["rows"], by_doc[d["doc_id"]])
    ents = [cache.rows[j["req_hash"]] for j in jobs]
    update_meta(a.system, {
        **cfg, "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": user_prompt(["<previous source segment 1>", "<previous source segment 2>"], "<source segment>"),
        "user_prompt_no_context": NO_CONTEXT,
        "context": "previous two source segments of the same document in aligned order (fewer at the start)",
        "decoding": {"temperature": 0.0, "seed": 0, "max_tokens_rule": "min(1024, 3*len(src_tokens)+64)",
                     "stop": "renderer stop sequences"},
        "postprocessing": "renderer.parse_response content, surrounding whitespace stripped; nothing else",
        "segments": len(jobs), "prompt_tokens": sum(e["n_prompt"] for e in ents),
        "output_tokens": sum(e["n_out"] for e in ents),
        "cost_usd_estimate": round(sum(e["cost_usd"] for e in ents), 4),
        "price_assumed_usd_per_m": {"prefill": a.price_in, "sample": a.price_out},
        "terminations": {t: sum(1 for e in ents if e["termination"] == t) for t in {e["termination"] for e in ents}},
        "tinker": tinker.__version__ if hasattr(tinker, "__version__") else None,
        "python": platform.python_version(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_s_this_run": round(time.perf_counter() - t0, 1),
        "cache": f"data/realdocs/gold/mt/cache_{a.system}.jsonl",
        "files": [f"data/realdocs/gold/mt/{a.system}/{d['doc_id']}.jsonl" for d in docs],
    })
    print(f"wrote {len(docs)} files under {MT_DIR / a.system}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["marian", *LLM])
    ap.add_argument("--device")
    ap.add_argument("--pilot", type=int, default=0, help="only the first N segments of the corpus")
    ap.add_argument("--cap", type=float, default=0.40)
    ap.add_argument("--price-in", type=float, default=2.00, help="USD per 1M prompt tokens (assumed)")
    ap.add_argument("--price-out", type=float, default=5.00, help="USD per 1M generated tokens (assumed)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    docs = load_docs()
    if a.system == "marian":
        run_marian(docs, a.device)
    else:
        run_llm(docs, a)


if __name__ == "__main__":
    main()
