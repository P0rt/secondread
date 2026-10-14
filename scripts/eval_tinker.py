"""Score a split on Tinker with exact label logprobs: risk = p(Major) / (p(Major) + p(OK)).

usage:
  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/eval_tinker.py SPLIT.jsonl \
      (--model-path tinker://.../sampler_weights/final | --base) [--n N] [--out FILE] [--dry-run]

The prompt is secondread.prompt.generation_prompt_ids (the same token ids
probe_local.py sends to llama-server) with one label token appended, so the label
position becomes a *prompt* position and the server can score any token there:

  --method target (default)  one request; target_prompt_logprobs asks for exactly
                             ids 33331 (Major) and 3793 (OK) at that position.
  --method pair              fallback: two compute_logprobs requests (prompt + Major,
                             prompt + OK), each scoring only its last token.

Both are exact; neither depends on the labels making a top-k cut, which matters for
the base-model baseline (docs/prep/02-corrections.md #15). With --topk K > 0 the same
request also returns the top-K tokens at the label position ("top" in the output) and
the logprob of the appended token, which must equal the target logprob (checked).

Output rows match probe_local.py ({"id", "label", "risk", "p_major", "p_ok", "top"})
plus "doc", "lp" (for document-bootstrap CIs in report.py), "lp_major", "lp_ok".
An existing --out file is resumed: ids already in it are skipped.
--n N picks rows exactly like probe_local.py (random.Random(0).shuffle, first N), so
local and Tinker scores of the same N are comparable row by row.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time

from secondread import metrics
from secondread.prompt import BASE_MODEL, LABEL_IDS, generation_prompt_ids, renderer

PREFILL_PRICE_PER_M = 0.33  # USD / 1M prompt tokens, Qwen/Qwen3.5-4B (docs/prep/03 section 3)
SAMPLE_PRICE_PER_M = 1.005  # every request still samples max_tokens=1 token
PROBE = LABEL_IDS["OK"]  # token appended to the prompt; any id works, its own logprob is a free cross-check
IDS = (LABEL_IDS["Major"], LABEL_IDS["OK"])


def select_rows(path: str, n: int | None) -> list[dict]:
    rows = metrics.load_scored(path)  # any jsonl
    if n is not None:  # probe_local.py's selection, for row-by-row comparability
        random.Random(0).shuffle(rows)
        rows = rows[:n]
    return rows


def build_target(prompt_len: int):
    """[len(prompt)+1 - 1, 2] int64 ids, -1 everywhere except the label row.

    Row i scores prompt position i+1; the appended token sits at position prompt_len,
    so its row is prompt_len - 1. Rows before the first named id are not scored, so
    the request costs one position, not the whole prompt."""
    import torch
    import tinker

    ids = torch.full((prompt_len, len(IDS)), -1, dtype=torch.int64)
    ids[prompt_len - 1] = torch.tensor(IDS)
    return tinker.TensorData.from_torch_sparse(ids, pad_value=-1)


def result_row(r: dict, lp_major: float, lp_ok: float, top: str | None) -> dict:
    risk = 1.0 / (1.0 + math.exp(lp_ok - lp_major))  # = pM / (pM + pOK), overflow-safe
    out = {"id": r["id"], "label": r["label"], "risk": risk, "p_major": math.exp(lp_major), "p_ok": math.exp(lp_ok),
           "top": top, "doc": r.get("doc"), "lp": r.get("lp"), "lp_major": lp_major, "lp_ok": lp_ok}
    if "synth_type" in r:
        out["synth_type"] = r["synth_type"]
    return out


async def score_target(sc, r: dict, a, tokenizer) -> tuple[dict, float | None]:
    import tinker

    prompt = generation_prompt_ids(r["src"], r["mt"], r["lp"])
    extra = dict(include_prompt_logprobs=True, prompt_logprobs_last_n=1, topk_prompt_logprobs=a.topk) if a.topk else {}
    res = await sc.sample_async(prompt=tinker.ModelInput.from_ints(prompt + [PROBE]), num_samples=1,
                                sampling_params=tinker.SamplingParams(max_tokens=1),
                                target_prompt_logprobs=build_target(len(prompt)), **extra)
    lp_major, lp_ok = (float(x) for x in res.target_prompt_logprobs.to_torch()[len(prompt) - 1])
    top, drift = None, None
    if a.topk:
        drift = abs(res.prompt_logprobs[-1] - lp_ok)  # same quantity, two code paths
        top_k = res.topk_prompt_logprobs[-1] or []
        top = tokenizer.decode([top_k[0][0]]) if top_k else None
    return result_row(r, lp_major, lp_ok, top), drift


async def score_pair(sc, r: dict, a, tokenizer) -> tuple[dict, float | None]:
    import tinker

    prompt = generation_prompt_ids(r["src"], r["mt"], r["lp"])
    lps = await asyncio.gather(*(sc.compute_logprobs_async(tinker.ModelInput.from_ints(prompt + [t]),
                                                           prompt_logprobs_last_n=1) for t in IDS))
    return result_row(r, float(lps[0][-1]), float(lps[1][-1]), None), None


def summarize(path: str, wall: float | None = None) -> None:
    rows = metrics.load_scored(path)
    y, s = metrics.to_arrays(rows)
    base = y.mean()
    print(f"n={len(rows)} Major-rate={base:.3f}")
    print(f"AUROC={metrics.auroc(y, s):.3f} PR-AUC={metrics.pr_auc(y, s):.3f} (random={base:.3f}) "
          f"recall@10%={metrics.recall_at(y, s):.3f} (random=0.100)")
    tops = {}
    for x in rows:
        tops[x.get("top")] = tops.get(x.get("top"), 0) + 1
    print("top-1 tokens:", dict(sorted(tops.items(), key=lambda kv: -kv[1])[:5]))
    if wall:
        print(f"wall {wall:.0f}s")


def cost(prompt_tokens: int, n_requests: int) -> float:
    return prompt_tokens * PREFILL_PRICE_PER_M / 1e6 + n_requests * SAMPLE_PRICE_PER_M / 1e6


def dry_run(rows: list[dict], a) -> None:
    """Build every prompt and the first --dry-run-n target tensors; check their layout."""
    tok = renderer().tokenizer
    n_req = 2 if a.method == "pair" else 1
    total = 0
    for i, r in enumerate(rows):
        prompt = generation_prompt_ids(r["src"], r["mt"], r["lp"])
        # The label must be the first token after the think-disabled assistant header.
        assert tok.decode(prompt).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), r["id"]
        total += (len(prompt) + 1) * n_req
        if i < a.dry_run_n:
            t = build_target(len(prompt))
            dense = t.to_torch(pad_value=-1)
            assert list(dense.shape) == [len(prompt), len(IDS)]  # server wants len(prompt + probe) - 1 rows
            assert (dense != -1).nonzero().tolist() == [[len(prompt) - 1, j] for j in range(len(IDS))]
            assert dense[len(prompt) - 1].tolist() == list(IDS)
            layout = "sparse CSR" if t.sparse_crow_indices is not None else "dense"
            print(f"[{i}] {r['id']} label={r['label']}: prompt {len(prompt)} tok + probe {PROBE}; "
                  f"target {list(dense.shape)} {layout}, ids {list(IDS)} only in row {len(prompt) - 1}; "
                  f"prompt tail {tok.decode(prompt[-6:])!r}")
    print(f"{len(rows)} rows, method={a.method}: {n_req * len(rows)} requests, {total:,} prompt tokens, "
          f"est. ${cost(total, n_req * len(rows)):.3f} (prefill ${PREFILL_PRICE_PER_M}/M + 1 sampled token/request)")
    print("dry run OK: no network calls were made")


async def run(rows: list[dict], a) -> None:
    import tinker

    done: set[str] = set()
    if os.path.exists(a.out) and not a.overwrite:
        done = {x["id"] for x in metrics.load_scored(a.out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already in {a.out}, scoring {len(todo)} "
          f"({'base ' + BASE_MODEL if a.base else a.model_path}, method={a.method})")
    if todo:
        service = tinker.ServiceClient(base_url=a.base_url)
        sc = await (service.create_sampling_client_async(base_model=BASE_MODEL) if a.base
                    else service.create_sampling_client_async(model_path=a.model_path))
        tok = renderer().tokenizer
        score = score_pair if a.method == "pair" else score_target
        sem = asyncio.Semaphore(a.concurrency)
        failures, drifts = [], []

        async def one(r):
            async with sem:
                try:
                    return await score(sc, r, a, tok)
                except Exception as e:  # keep going; a rerun picks up the missing ids
                    failures.append((r["id"], repr(e)[:200]))
                    return None

        t0 = time.perf_counter()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w" if a.overwrite else "a", encoding="utf-8") as f:
            for k, fut in enumerate(asyncio.as_completed([one(r) for r in todo]), 1):
                got = await fut
                if got is not None:
                    row, drift = got
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    if drift is not None:
                        drifts.append(drift)
                if k % 200 == 0:
                    print(f"  {k}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if drifts:
            worst = max(drifts)
            print(f"cross-check |prompt_logprob - target_logprob| max={worst:.2e}"
                  + ("  WARNING: target logprobs disagree with prompt logprobs" if worst > 1e-3 else ""))
        if failures:
            print(f"{len(failures)} rows failed (rerun to retry), e.g. {failures[:2]}", file=sys.stderr)
        summarize(a.out, time.perf_counter() - t0)
    else:
        summarize(a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact-logprob scoring on Tinker.")
    ap.add_argument("split")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--model-path", help="tinker://.../sampler_weights/<name> from train_tinker.py")
    src.add_argument("--base", action="store_true", help=f"score the untuned {BASE_MODEL} (baseline)")
    ap.add_argument("--n", type=int, default=None, help="random subset, same rows as probe_local.py N")
    ap.add_argument("--out", help="output jsonl (resumed if it exists)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--method", choices=["target", "pair"], default="target")
    ap.add_argument("--topk", type=int, default=5, help="top-k at the label position for 'top'; 0 disables")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--dry-run", action="store_true", help="build and check requests; no network, no API key")
    ap.add_argument("--dry-run-n", type=int, default=3, help="requests to print in --dry-run")
    a = ap.parse_args()

    rows = select_rows(a.split, a.n)
    if a.dry_run:
        dry_run(rows, a)
        return
    if not (a.base or a.model_path):
        ap.error("one of --model-path or --base is required")
    if not a.out:
        ap.error("--out is required")
    asyncio.run(run(rows, a))


if __name__ == "__main__":
    main()
