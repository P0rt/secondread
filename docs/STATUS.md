# SecondRead status, 2026-10-14 (evening)

Commands use `py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }`. Keys are
in `.env`; set them with `set -a && . ./.env && set +a`.

All numbers come from running the code today. Local runs used the brew llama-server, build
11429. Tinker runs used Qwen/Qwen3.5-4B. DigitalOcean runs used `gemma-4-31B-it` on serverless
inference.

**Results:** see [RESULTS.md](RESULTS.md). On en-ru test (3,573 segments), LoRA v1 running locally
on Q4 gets AUROC 0.801 and PR-AUC 0.537. The base model gets 0.666 and 0.369, and Gemma 4 31B
direct gets 0.794 and 0.494. On en-es test, v1 gets 0.815 and 0.517, base 0.776 and 0.413,
Gemma direct 0.807 and 0.311.

## Spend (hard budgets: Tinker $10, DigitalOcean $5 credit)

| | Spent | Source | Left |
|---|---|---|---|
| **Tinker** | **$1.33 billed** up to the 14:00 CEST bucket. **About $1.66** once the 14:00–15:00 bucket lands | `tinker -f json billing usage`; the rest is estimated from the run logs | **about $8.34** of $10 |
| Tinker storage | 22 checkpoints, 2.83 GB, $0.28/month | `tinker billing checkpoint-storage` | ongoing |
| **DigitalOcean** | **at most $0.77** at list price | token counts in both caches: baseline $0.348, synthetic data $0.418 | **at least $4.23** of $5 |
| Sentry | $0 | Developer plan | |

**Tinker bills every token of the datum, including weight-0 prompt tokens.** This answers the
G3 question. G3 was billed for 62,820 tokens and G4 for 31,353, exactly the `num_tokens` in
their logs.
- **Rates:** training $0.737/M tokens, sampling prefill $0.33/M, sampled tokens $1.005/M.
- **Dev scoring during training** (`--dev-n`) is billed as training tokens. Each LR probe was
  billed for 159,307 tokens: 125,171 for training plus 34,136 for two dev evaluations.
- **Totals so far:** 1,922,669 training tokens ($1.42), 303,084 forward tokens for dev scoring
  ($0.22), and about 59k sampling tokens ($0.02).
- **Planning rule:** an epoch costs $0.737 × the total tokens. For SFT v2 (19,917 rows,
  311 steps) that is about $1.52.

## In flight

- **Nothing on Tinker or DigitalOcean.** v1 finished at 14:02 CEST, and no training or eval
  process is alive.
- **`:9931` still serves the base model without the LoRA,** with default flags (no `-cram 0`).
  The integration pass used it read-only.
- The integration pass also ran a LoRA server on `:9936` for local scoring. It has been stopped.

## Done and verified this afternoon

| Item | What was run | Result |
|---|---|---|
| G1 | `scripts/check_keys.py` | `Qwen/Qwen3.5-4B` is trainable on Tinker |
| G3 | `train_tinker.py --max-steps 10 --log-path runs/g3` | 10 steps of batch 64, about 11–13 s per step, 99 s in all. At the default LR 4.9e-4 the label NLL swings: 0.75, 4.12, 1.78, 12.2, 1.05, … The billing question is answered above |
| LR probes | 3 × 20 steps at a constant LR, `runs/lrprobe_{2e-4,1e-4,5e-5}` | All stable after step 3. Dev AUROC on 100 rows: 0.604 → 0.691 / 0.670 / 0.671. **2e-4** was chosen |
| G4 with a real checkpoint | 5 steps (`runs/g4`), `scripts/fetch_adapter.py`, `export_adapter.sh` | `tinker checkpoint download` hung for over 10 min, and the SDK marks its endpoint deprecated. `fetch_adapter.py` (`save_weights_external`) works in 2–5 s. The 72 MB adapter has the same keys as the fake one (96 MLP pairs) but is float32, with a full PEFT config. `export_adapter.sh` needed no change, and llama-server loaded 192 tensors |
| Tinker eval API | `eval_tinker.py --method target` | `target_prompt_logprobs` and `prompt_logprobs_last_n` are accepted, and exact label logprobs come back |
| **v1** | `train_tinker.py --data data/sft_v1.jsonl --lr 2e-4 --log-path runs/v1` | 231 steps and 1,452,983 tokens in 9 min 24 s. Dev AUROC on 200 rows: 0.552 at step 0, 0.739 at the end. `final` sampler and state checkpoints are saved (run `479fd56b…:train:0`). Exported to `models/secondread-lora.gguf` (36 MB) |
| **G5** | `probe_local.py --lora-scale 1` vs `eval_tinker.py`, 300 en-ru dev rows | **Passes.** Spearman 0.989, mean difference −0.025, mean absolute difference 0.029. Base vs base: Spearman 0.944, mean difference +0.018 |
| G6 | `scripts/check_keys.py` | `gemma-4-31B-it` returns logprobs and top_logprobs, with no reasoning field. Limits: 120 requests/min, 2M tokens/min; the per-day header reads 6M remaining |
| **G7** | `baseline_gemma.py --method direct` on all of en-ru dev and test and en-es test (8,360 calls); `--method gemba` on 300 dev and 500 test rows | No failed calls and 0 of 800 GEMBA parse failures. The numbers are in RESULTS.md. Prompt: `prompts/gemba_mqm.txt` (CC BY-SA 4.0 header) |
| DO terms | Read Gemma 4 license, Gemma ToU, DO Service Terms / TOS / AUP | **Gemma outputs may train the LoRA.** Gemma 4 is Apache-2.0 and is not under the Gemma ToU, and DO puts no restriction on training on outputs. See `docs/LICENSES-NOTES.md` |
| LLM synthetic data | `synth_llm.py`, several runs merged by id | 4,017 train rows and 763 test rows (neg_remove, neg_insert, omission, addition, ok_paraphrase). The integration pass re-checked split hygiene independently: every row's document is in its own split, no source sentence is shared with another split, and train and test share 0 sources |
| Latency: the 21 s first job | Backend vs pure llama-server, with and without Sentry, idle tests | The cause is llama-server's RAM prompt cache, which swaps out after a few idle minutes. The fix is **`-cram 0`**: steady 3.7 GB, and 8.4–9.0 s for the 51-segment demo even after idle. Backend overhead is about 0.7 ms per segment. Sentry made no difference |
| **Local test scoring** (this pass) | `probe_local.py … 100000 --workers 4`. Base on `:9931`, v1 on `:9936` | Full en-ru test, en-es test and en-es dev for both systems, 0 labels missing from the top 200. `--workers` is new in `probe_local.py`: 1 is the default and reproduces `results/base_enru_dev300.jsonl` exactly, and 4 shifts risks by at most 0.003 |
| Regression checks (this pass) | `tests/test_xliff.py`, `canary_check.py`, `synth_llm.py --self-test` | 20/20 pass, the canary is in 0 envelopes (4 with the control), and 15 validator cases pass |
| Repo hygiene (this pass) | Scanned tracked and untracked files, plus everything on disk except `.env`, for the three key values; checked git history; ran `git status --ignored` | 0 hits in files and 0 in history. `data/`, `models/`, `ckpt/`, `runs/` and `docs/prep/` are ignored. `docs/prep/` was never committed. No ACES or HalOmi data is on disk. The `results/*.jsonl` files hold ids, gold labels and scores only, with no WMT text or model quotes |

## Needs you

- **Rotate `DO_MODEL_KEY`** if any agent transcript from today will be shared. Part of the
  key appeared in a process listing in one track's tool output. No file holds it.
- **Restart `:9931`** with the LoRA and the cache fix, after checking that nothing needs the
  base-only server:
  ```sh
  llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9931
  ```
- **Checkpoint cleanup.** Decide whether to delete the G3, G4 and LR-probe checkpoints on
  Tinker. All 22 checkpoints together cost $0.28 a month, and `final` checkpoints never expire.
- **G8:** check that the review traces are in Sentry's Agents dashboard and take the
  screenshot. `SENTRY_DSN` is set, and the latency track sent about 22 traces with metadata
  only.
- **The tester session,** the post.

## Not built yet

- **Annotated-file download:** `/api/review` returns XLIFF keys as `id`, and
  `xliff.annotate()` works. But no endpoint returns the annotated file yet.
- **Thresholds in the app:** `DEFAULT_THRESHOLD` (0.5) is still a placeholder. The
  dev-optimal F1 thresholds for v1 are 0.253 for en-ru (from the 300 dev rows) and 0.702 for
  en-es (from en-es dev). Put them into the server config per language pair, and better, take
  en-ru from the full dev split.
- **Rule confidences:** the values from 0.25 to 0.75 are set by hand. Tune them on dev plus the
  synthetic test sets.
- **Other eval sets:**
  - WMT25 en-ru OOD conversion (only `scripts/wmt25_count.py` exists);
  - ACES and HalOmi loaders (eval only, NC licenses).
- **`pyproject.toml` gaps:**
  - `httpx` and `numpy` are imported directly but only installed transitively;
  - `pytest` is absent.
- **No `LICENSE` file.** `prompts/gemba_mqm.txt` is CC BY-SA 4.0 and must stay outside the
  code license; its header already says so.
- **Sentry span limit:** Sentry keeps about 1,000 spans per transaction, so jobs over about
  990 segments lose some `chat` spans.

## Next steps, in order

1. **Serve v1:** restart `:9931` with `--lora … -cram 0` (see "Needs you"). Then run:
   - `make_demo_results.py --api http://127.0.0.1:8787`, which drops the placeholder banner;
   - `bench_review.py --lora-scale 1`.
2. **Thresholds:** score the full en-ru dev split locally, which is free and takes about
   10 min:
   ```sh
   py scripts/probe_local.py data/splits/en-ru.dev.jsonl 100000 --workers 4 --lora-scale 1 --port <lora-port> \
      --out results/local_v1_enru_dev.jsonl
   ```
   Then set per-pair thresholds in the server.
3. **v2 on Tinker,** about $1.52 at the billing rate measured above.
   - Build the data:
     ```sh
     py scripts/build_sft.py --synthetic data/synth/rules.train.jsonl data/synth/llm.train.jsonl \
        --out data/sft_v2.jsonl
     ```
     The synth track reported 19,917 rows, 37.1% Major and 311 steps. Always pass `--out`:
     the default is `data/sft_v1.jsonl`, which would overwrite the v1 data.
   - Train with `--lr 2e-4`, the same as v1.
   - Optional first: drop the 82 `neg_insert` rows whose MT is a question. In Russian, «Вы не
     …?» can be polite with the same meaning, so these labels may be wrong.
   - Then `fetch_adapter.py` → `export_adapter.sh` → local scoring on the same splits.
   - Score the synthetic test sets with `report.py --by synth_type` for recall per error type:
     - `rules.test.jsonl` (313 rows) and `llm.test.jsonl` (763 rows) for base, v1 and v2,
       locally and free;
     - Gemma direct on those 1,076 rows, about $0.02.
   - Keep v1 if v2 does not beat it on en-ru and en-es test PR-AUC.
4. **Optional:** Tinker exact eval of v1 on en-ru test, about $0.10. G5 already shows that
   local Q4+LoRA tracks Tinker (Spearman 0.989), so this only confirms it.
5. **G8** Sentry screenshot and a real debugging story: the `-cram 0` latency hunt is a good
   candidate.
6. Annotated-XLIFF download endpoint, the tester session, the static demo on DO App Platform,
   then the post.

---

## Earlier today: the pipeline on the base model (morning pass, still valid)

The local numbers in this section used `:9931` with the base Q4_K_M model and **no LoRA**.

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

**Changes made in the morning integration pass:**
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
