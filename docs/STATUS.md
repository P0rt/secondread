# SecondRead status, 2026-10-14

All numbers below come from running the code on this Mac today: the brew llama-server
build 11429 on `127.0.0.1:9931` with the base Qwen3.5-4B Q4_K_M and **no LoRA**. Commands use
`py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python "$@"; }`.

## Done and verified

| Area | What was run | Result |
|---|---|---|
| G0 disk | `df -h` | 20 GiB free |
| G2 labels | `POST :9931/tokenize` and the Python tokenizer | `Major` → [33331], `OK` → [3793] in both; `" Major"` → [16617], so the prompt must not put a space before the label |
| Splits | `scripts/split.py` with ratios 0.7/0.15/0.15 (en-ru) and 0.7/0.1/0.2 (en-es) | Reproduces the committed `data/splits/*.docs.json` exactly. See the split table below |
| SFT v1 | `py scripts/build_sft.py` | 14,791 rows, of which 3,308 Major (22.4%). 0 rows over 1,000 tokens; the longest is 537. 53 prompts appear with both labels (label noise) |
| SFT v2 | `build_sft.py --synthetic data/synth/rules.train.jsonl` | 15,900 rows, of which 4,172 Major (26.2%), incl. 1,109 synthetic. With `rules.test.jsonl` the build stops: "doc … is in the test split -> leak, refusing" |
| Training (dry run) | `train_tinker.py --data data/sft_v1.jsonl --dry-run` | Batch 64, 231 steps, LR 4.905e-4, LoRA rank 16 MLP-only (`train_attn=False`, `train_unembed=False`). Of 192 datums checked, all have loss only on label + `<\|im_end\|>` and a prefix identical to the inference prompt. Cost: $1.07 if every token is billed, $0.022 if only weighted tokens are. With v2 and `--max-steps 10`: $0.05 |
| Tinker eval (dry run) | `eval_tinker.py data/splits/en-ru.dev.jsonl --base --dry-run` | 2,972 requests, 253,135 prompt tokens, about $0.087 |
| Rule synthetic data | `synth_rules.py --out-dir <scratch>` | 1,109 train and 313 test rows. Same md5 as the existing `data/synth/` files, so the generator is deterministic |
| LLM synthetic data (offline) | `synth_llm.py --self-test`, `--dry-run --per-type 2` | 15 validator cases pass. The dry run plans 40 calls (worst case $0.0067) and writes no files |
| Metrics refactor | `probe_local.py data/splits/en-ru.dev.jsonl 30` | Risks identical to `results/base_enru_dev300.jsonl` (max difference 0.0) |
| Baseline (a), local | `report.py base=results/base_enru_dev300.jsonl --n-boot 300` | Base model on 300 en-ru dev rows, 30 docs, Major rate 0.247: AUROC 0.565 [0.479, 0.666], PR-AUC 0.332 [0.223, 0.462], recall@10% 0.149 [0.077, 0.238] |
| G10 XLIFF | `py tests/test_xliff.py`, with the Okapi SDLXLIFF sample | 20/20 pass. `make_demo_xliff.py` rebuilds `demo/` byte for byte |
| Backend, files | `POST /api/review` (multipart) with the three `demo/` files, plus a raw XML body | HTTP 200 for each: 51 segments, 18 flagged, 161–178 ms per segment |
| Backend, JSON | `POST /api/review` with 3 pairs | Dropped "not" → `negation_dropped` with a span. 5 → 7 → `number_changed` with spans on both sides. "before 5 p.m." vs "до 17:00" is correctly not a number issue |
| Backend, errors | Invalid JSON; a non-XLIFF file | 422 `bad_request` with field errors; 400 `xliff_invalid` |
| Write-back | `xliff.annotate()` fed with the response `id`s, then reparse, in all 3 formats | All 51 ids match the file keys. Text is unchanged after the round trip and all 51 risks are stored |
| Demo vs gold | Base-model scores for the 51 demo segments (14 Major) | AUROC 0.728, PR-AUC 0.570. At threshold 0.5: precision 0.50, recall 0.64 |
| G9 latency | `bench_review.py data/splits/en-ru.dev.jsonl 100` | 157 ms per segment (server time). Model call p50 599 ms, p95 759 ms. AUROC 0.630, PR-AUC 0.386 (Major rate 0.27). The rule checks did not change the ranking on this sample |
| G4 without a real checkpoint | Fake Tinker adapter → `export_adapter.sh` | 36 MB GGUF. llama-server reports "loaded 192 tensors from lora file". P(Major) moves by up to 0.655 between scale 0 and 1. 23 s end to end |
| Backend with LoRA | Second backend against a llama-server started with that GGUF | `/api/health` lists the adapter. `lora_scale: 0` reproduces the base server's risks exactly (0.883 and 0.167). Scale 1 or no scale gives `lora_active: true` and the "LoRA" label |
| UI | `npm run build`; `vite preview` proxying to the backend; demo page; `ui/src/api.ts` run in Node on real responses | Builds clean: 21.45 kB JS, 13.86 kB CSS, no Sentry SDK in the bundle without a DSN. Health and multipart requests go through the proxy. The demo shows 10 segments, 5 flagged, with the placeholder banner and attribution. The normaliser reads counts, threshold, model note and spans from real backend output |
| G8, offline part | `py scripts/canary_check.py` | **PASS**: the canary is in 0 of 12 envelopes. With the scrubbers off, the same search finds 4 hits. No outbound connection was attempted. Spans recorded: `invoke_agent` 4, `execute_tool` 15 (all 4 tools), `chat` 7 |
| Repo hygiene | `git status -uall`, `git check-ignore`, secret grep | Everything under `data/` is ignored except the split manifests, including paths for ACES, HalOmi and WMT25 that are not downloaded yet. No keys, DSNs or `.env` anywhere in the tree |

Splits (units; Major in brackets):

| Pair | Documents train / dev / test | Train | Dev | Test |
|---|---|---|---|---|
| en-ru | 114 / 31 / 33 | 12,747 | 2,972 (803) | 3,573 (723) |
| en-es | 109 / 12 / 25 | 5,865 | 1,650 (95) | 1,815 (168) |

**Changes made in this integration pass:**
- `.gitignore`: `data/*` is now ignored with explicit exceptions, instead of a list of
  known files. Also added `/.env.*` and `.DS_Store`.
- `README.md` written.
- No code changes were needed. The UI and the server agree on the API shape, the agent uses
  `secondread.xliff` correctly, and `probe_local.py` works with the shared metrics module.
  All three were checked by running them.

**Correction to an agent report:** `v0.6.0` in llama.cpp is an annotated tag. Its tag
object id is `69ecae9`, but it points to commit `d81235049`, the same commit as `b11429` and
the brew build. Item 10 in `02-corrections.md` (`--branch v0.6.0`) was therefore correct, and
either tag works.

## Blocked

### `TINKER_API_KEY`
You need to set up the account, billing and key yourself. Card details are not something
Claude can enter, a virtual card included.
- **G1:** `get_server_capabilities()` lists `Qwen/Qwen3.5-4B` as trainable.
- **G3:** 10-step smoke run. Check that the step time is 30 s or less, and run
  `tinker billing usage` to see whether weight-0 prompt tokens are billed ($1.07 vs $0.022 per
  epoch).
- **G4 with a real checkpoint:** do the key names, dtype, `adapter_config.json` fields and the
  folder layout from `tinker checkpoint download` match the fake adapter? If they don't,
  `export_adapter.sh` step 1 names the failing key.
- **Eval API:** does the live service accept `target_prompt_logprobs` and
  `prompt_logprobs_last_n`? If not, use `eval_tinker.py --method pair` or `--topk 0`.
- **G5:** agreement between the Tinker sampler and local Q4+LoRA. Target: Spearman ≥ 0.9 and
  mean difference ≤ 0.05.
- The main training run, Tinker-side evaluation, and the Tinker version of baseline (a). The
  local baseline (a) already works through `probe_local.py` and `lora_scale: 0`.
- **G11:** the Tinker promo credit.

### `DO_MODEL_KEY`
- **G6:** `GET /v1/models` and one chat call. Record whether `logprobs` is null, the
  `x-ratelimit-*` headers (tokens per day) and any reasoning field.
- **Real `synth_llm.py` run:** prompt following, JSON compliance and rejection rates.
- **DO "AI Model Terms":** whether Gemma outputs may be used for training is UNVERIFIED.
  Until someone has read the terms, `llm.train.jsonl` is eval-only.
- **G7:** the GEMBA-MQM baseline. The script is not written yet; see below.

### `SENTRY_DSN` (EU org) and `VITE_SENTRY_DSN`
- **G8:** the trace showing up in the Agents dashboard. The canary part already passes
  offline.
- The browser SDK against real ingest (so far tested only against a fake local ingest).
- One real debugging story taken from a trace, for the post.

## Not built yet (no key needed)

- **GEMBA-MQM baseline (G7):** a script plus the prompt in its own CC BY-SA 4.0 file with
  attribution. Map its output with the same "meaning family" label rule.
- **Annotated-file download:** `/api/review` returns XLIFF keys as `id` and `xliff.annotate()`
  works, but no endpoint returns the annotated file yet.
- **Other eval sets:** WMT25 en-ru OOD test conversion (only `scripts/wmt25_count.py` exists),
  and ACES / HalOmi eval loaders (eval only, NC licenses).
- **Calibration:** `DEFAULT_THRESHOLD` (0.5) is a placeholder. Choose the threshold per
  language pair on dev with `metrics.best_f1_threshold` once the LoRA exists.
- **Rule confidences:** the 0.25–0.75 values are hand-set. Tune them on dev plus synthetic
  data. On natural WMT data their precision is low, because these error types are rare
  there.
- **`pyproject.toml` gaps, left for the lead:**
  - `httpx` and `numpy` are imported directly but only installed transitively.
  - `pytest` is absent, so `tests/test_xliff.py` runs as a plain script.
- **No `LICENSE` file yet.** The GEMBA prompt must stay outside the code license.
- **`docs/prep/` is tracked.** It holds internal planning notes (prize strategy, budget,
  local scratchpad paths). Decide before the repo goes public.
- **Sentry span limit:** Sentry keeps about 1,000 spans per transaction, so jobs over about
  990 segments lose some `chat` spans.

## Next steps, in order

1. **You:** create the Tinker account, billing and API key, and put `TINKER_API_KEY` in `.env`.
   Claim the promo credit.
2. G1.
3. G3:
   `py scripts/train_tinker.py --data data/sft_v1.jsonl --max-steps 10 --log-path runs/g3`,
   then `tinker billing usage`.
4. G4: run 3–5 steps, then `tinker checkpoint download` → `scripts/export_adapter.sh ckpt` →
   `llama-server --lora`. Check P(Major) at scale 0 and 1. The decision deadline for the
   fallback path is the evening of 15.10.
5. v1 main run on human labels only, on the morning of 15.10:
   `py scripts/train_tinker.py --data data/sft_v1.jsonl --log-path runs/v1` (231 steps, at
   most about $1.07).
6. Export v1 and restart `:9931` with `--lora models/secondread-lora.gguf`. Then run G5 on 30
   dev rows with `eval_tinker.py` vs `probe_local.py --lora-scale 1`.
7. Evaluation:
   - pick the threshold on en-ru dev;
   - score en-ru test and en-es test, base (scale 0) vs LoRA;
   - `report.py --n-boot 1000`;
   - for the synthetic test set, `--by synth_type`.
8. **You:** create the DigitalOcean account and a model-access key for `gemma-4-31B-it`, and
   turn off auto-reload. Then:
   - G6;
   - `synth_llm.py --per-type 20`, and inspect the rejects;
   - read the DO terms;
   - scale up.
9. v2: `build_sft.py --synthetic data/synth/rules.train.jsonl`, plus the LLM data if the
   terms allow. Then train, export and evaluate as for v1.
10. GEMBA-MQM baseline on DO (G7).
11. **You:** create the Sentry org in the EU region and set `SENTRY_DSN`. Take the G8
    dashboard screenshot and write up a real debugging story.
12. Polish:
    - per-pair thresholds and rule confidences;
    - `make_demo_results.py --api http://127.0.0.1:8787` with the LoRA loaded (this drops the
      placeholder banner);
    - `bench_review.py --lora-scale 1`.
13. Annotated-XLIFF download endpoint, the tester session, the static demo on DO App Platform,
    then the post.
