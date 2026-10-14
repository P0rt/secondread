# SecondRead status, 2026-10-14 (evening, after the Ship phase)

Commands use `py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }`. Keys are
in `.env`; set them with `set -a && . ./.env && set +a`.

All numbers come from running the code today. Local runs used the brew llama-server, build
11429. Tinker runs used Qwen/Qwen3.5-4B. DigitalOcean runs used `gemma-4-31B-it` on serverless
inference.

**Results:** see [RESULTS.md](RESULTS.md).

| Test split | v1 (shipped) AUROC / PR-AUC | v2 | Gemma 4 31B direct | base |
|---|---|---|---|---|
| en-ru, 3,573 segments | **0.801 / 0.537** | 0.799 / 0.504 | 0.794 / 0.494 | 0.666 / 0.369 |
| en-es, 1,815 segments | **0.815 / 0.517** | 0.804 / 0.446 | 0.807 / 0.311 | 0.776 / 0.413 |

All scores come from local Q4 runs; Gemma runs on DigitalOcean.

- **v1 ships.** v2 (plus 5,095 synthetic rows) is much better on the synthetic test sets but not
  on human labels: on en-es its PR-AUC is significantly lower.
- **`:9931` serves v1 with `-cram 0`.**
- **The backend uses per-pair thresholds calibrated on dev:** en-ru 0.1609, en-es 0.7019.
- **`/api/health` names the adapter version (v1).** The backend finds it from the adapter file's
  sha256.

## Spend (budgets: Tinker about $18 available, DigitalOcean $5 credit)

| | Spent | Source | Left |
|---|---|---|---|
| **Tinker** | **$1.661 billed** through the 12:00–13:00 UTC bucket (everything before v2), read at 18:05 CEST. **About $3.38** once v2 lands: +$1.52 training, +$0.20 dev scoring | `tinker -f json billing usage`; v2 is estimated from its token counts at $0.737/M | **about $14.6** of ~$18 |
| Tinker storage | 22 checkpoints before v2 (2.83 GB, $0.28/month), plus 7 from v2 (the 6 periodic ones expire after 7 days) and one external-weights export | `tinker billing checkpoint-storage` | ongoing |
| **DigitalOcean** | **at most $0.79** at list price | token counts in both caches: baseline $0.372 (10,236 calls), synthetic data $0.418 (5,626 calls) | **at least $4.21** of $5 |
| Sentry | $0 | Developer plan | |
| Ship phase | $0 | local only; one `tinker billing usage` read | |

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

- **Nothing on Tinker or DigitalOcean.**
- **`:9931` serves LoRA v1 with `-cram 0`** (PID 22074, started 18:00 CEST with `nohup` from the
  repo root, log `/tmp/claude-501/llama-9931.log`):
  ```sh
  llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9931
  ```
- **No backend is running.** The Ship phase ran one on `:8787` for the demo rebuild and the checks,
  then stopped it. Ports 9940–9944 are free.

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
| **v2 track** (evening) | `build_sft.py --synthetic rules.train llm.train` (drops the 31 en-ru `neg_insert` questions), `train_tinker.py` (run `4087b7c0…:train:0`, 310 steps, 8 min 37 s), export, local scoring of all splits and both synthetic test sets, Gemma direct on the synthetic sets | See RESULTS.md, "v2 and ablation": ties v1 on AUROC, lower PR-AUC on both test splits (en-es −0.071, significant); AUROC vs parents 0.95–1.00 on synthetic types. `models/secondread-lora-v2.gguf` |
| **Ship phase** (evening) | `report.py` re-run on all v2-track files; `:9931` restarted; `scripts/pick_thresholds.py`; backend on `:8787`; `make_demo_results.py --api`; `npm run build` and `build:demo`; `tests/test_xliff.py`; `canary_check.py` | All v2 numbers reproduce exactly. **v1 ships** (RESULTS.md, "Ship decision"). The served adapter matches `local_v1` scores (max diff 0.0012 on 40 rows). Thresholds are in `secondread/thresholds.json` and `/api/health` reports `adapter_version: v1`. Demo results are real LoRA v1 output (`placeholder: false`). XLIFF tests pass 20/20; the canary is in 0 of 12 envelopes, with 4 hits when the scrubbers are off |

## Needs you

- **Rotate `DO_MODEL_KEY`** if any agent transcript from today will be shared. Part of the
  key appeared in a process listing in one track's tool output. No file holds it.
- **Review and commit the Ship-phase changes** (list in "Changed in the Ship phase" below).
  Nothing is committed or pushed.
- **Checkpoint cleanup.** Decide whether to delete the G3, G4 and LR-probe checkpoints on
  Tinker. The 22 checkpoints from before v2 cost $0.28 a month, v2 added 7, and `final`
  checkpoints never expire.
- **Rule confidences.** In the app's fused score the rule checks lower test AUROC by about 0.01
  (RESULTS.md, "The app's fused score vs the model alone"). Decide whether to recalibrate them from
  the train split before the post, or to say so in it.
- **The post:** `docs/POST_DRAFT.md` has the v2, synthetic, spend and status placeholders filled.
  Still open: the title, video, live demo URL, screenshots, the first user and the optional
  agent session. Re-check the Tinker total against the bill first.
- **G8:** check that the review traces are in Sentry's Agents dashboard and take the
  screenshot. `SENTRY_DSN` is set, and the latency track sent about 22 traces with metadata
  only.
- **The tester session,** the post.

## Not built yet

- **Annotated-file download:** `/api/review` returns XLIFF keys as `id`, and
  `xliff.annotate()` works. But no endpoint returns the annotated file yet.
- **Rule confidences:** the values from 0.25 to 0.75 are still set by hand.
  - A rule hit is a Major unit far less often than that: 31% of the time on en-ru train, 19% on
    en-es train, and 6–8% on test.
  - Dev has too few hits to tune on: 16 rows in en-ru, 33 in en-es.
  - Calibrate the confidences on the train split. The rules are not fit to it.
  - Known false alarms (found on test): "£" vs «фунтов», "whether or not", "no doubt", "24/7"
    vs «круглосуточно», ordinal words («втором»).
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

Done in the Ship phase: v1 served, full en-ru dev scored, per-pair thresholds, v2 trained and
evaluated, demo results rebuilt. What is left:

1. **Rule confidences** from the train split, then check on dev (free; see "Not built yet").
2. **G8:** the Sentry screenshot and the debugging story, the `-cram 0` latency hunt. Run the
   backend with `.env` loaded so `SENTRY_DSN` is set.
3. **Static demo on DO App Platform.** `ui/dist` holds the demo build from the Ship phase; both
   builds write `ui/dist`, so rerun `npm run build:demo` before deploying. Then fill
   `{{LIVE_DEMO_URL}}` in the post.
4. **Annotated-XLIFF download** endpoint and button, the tester session, and the post's remaining
   placeholders.
5. **Optional, free:** evaluate the one-pass v1+v2 blend, with both adapters loaded at scale 0.5,
   on dev first (RESULTS.md, "Next experiments").
6. **Optional, about $1.6:** v2b with a capped synthetic share per pair.
7. **Optional, about $0.10:** Tinker exact eval of v1 on en-ru test. G5 already shows local
   tracks Tinker.

## Changed in the Ship phase (not committed)

- **New:**
  - `scripts/pick_thresholds.py`;
  - `secondread/thresholds.json`, with thresholds and stats only, no text.
- **Backend:**
  - `secondread/agent.py`: per-adapter, per-pair thresholds, adapter version by sha256, new
    summary fields;
  - `secondread/server.py`: health fields `adapter_version`, `scoring`, `thresholds` and
    `threshold_source`.
- **Scripts:** `scripts/make_demo_results.py` puts the version in the demo note.
- **UI and contract:**
  - `ui/public/demo-results.json`, rebuilt;
  - `ui/API.md`.
- **Docs:**
  - `docs/RESULTS.md`, new section "Ship decision";
  - this file, `README.md`, `docs/RUNBOOK.md`;
  - `docs/POST_DRAFT.md` and `docs/POST_CHECKLIST.md`, still untracked.
- **Git-ignored:**
  - `models/secondread-lora-v1.gguf`, a copy of v1;
  - `runs/ship/`, the fused scores.

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
