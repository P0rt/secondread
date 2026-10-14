# SecondRead status, 2026-10-15 (after the real-document gold-set scoring)

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

All scores come from local Q4 runs; Gemma runs on DigitalOcean. v3a and v3b have no test numbers:
they were not final candidates.

- **v1 ships, again.** Neither v2 nor v3 beat it on human labels.
  - v2 (plus 5,095 synthetic rows) is much better on the synthetic test sets, but on en-es test its
    PR-AUC is significantly lower.
  - v3a and v3b (plus wrong-sense synthetic rows) failed the dev selection rule fixed before their
    scores existed: both lose about 0.035 PR-AUC on en-ru dev (RESULTS.md, "v3 (wrong-sense
    synthetic) — dev selection"). They gain on the synthetic sense set, which comes from their own
    generator.
- **`:9931` serves v1 with `-cram 0`** (restarted in the Ship phase; 40 en-ru dev rows match the
  selected model's dev scores to within 0.00044).
- **The backend flags at the `review` operating point by default** (RESULTS.md, "Review operating
  point"): per pair, the highest threshold with dev recall of Major ≥ 0.60 on the app's fused score.
  - v1: en-ru 0.1857 and en-es 0.4984. The F1 line is kept as the `f1` point (en-ru 0.1609, en-es
    0.7019): `SECONDREAD_OPERATING_POINT=f1`, or `"operating_point": "f1"` in a request.
  - **Measured once on test** (RESULTS.md, "v3 — test and real documents"):

    | Pair | Line | Recall [95% CI] | Precision | Flag rate | F1 |
    |---|---|---|---|---|---|
    | en-ru | review 0.1857 | 0.744 [0.626, 0.835] | 0.392 | 38.4% | 0.513 |
    | en-ru | f1 0.1609 | 0.769 | 0.372 | 41.8% | 0.502 |
    | en-es | review 0.4984 | **0.482** [0.421, 0.581] | 0.544 | 8.2% | 0.511 |
    | en-es | f1 0.7019 | 0.304 | 0.773 | 3.6% | 0.436 |

  - **en-es misses its dev target:** the whole recall CI is below 0.60. The line was not moved,
    because moving it now would be a choice made on test. en-ru overshoots (0.744).
  - **Real documents: the gold set** (REALDOCS_GOLD.md, section 8; scored once on 2026-10-15).
    10 public documents, 720 segments, OPUS-MT and DeepSeek-V3.1 outputs, 1,440 units, 84 Majors
    (80 OPUS-MT, 4 DeepSeek), labelled blind before any detector was scored. 2,000
    document-bootstrap replicates:

    | System | AUROC [95% CI] | PR-AUC | recall@10% | Review line: flags (Major) | Recall [95% CI] | Precision | Flag rate |
    |---|---|---|---|---|---|---|---|
    | v1, app score | 0.910 [0.884, 0.928] | 0.520 | 0.619 | 0.4984: 22 (17) | **0.202** [0.139, 0.296] | 0.773 | 1.5% |
    | base, app score | 0.839 [0.770, 0.885] | 0.400 | 0.571 | 0.6183: 113 (41) | 0.488 [0.350, 0.640] | 0.363 | 7.8% |
    | Gemma 4 31B direct | 0.940 [0.911, 0.957] | 0.627 | 0.750 | 0.1212 (en-ru dev): 175 (69) | 0.821 [0.712, 0.894] | 0.394 | 12.2% |

    - **v1 − Gemma:** AUROC −0.030 [−0.051, −0.009], PR-AUC −0.107, recall@10% −0.131, all three
      CIs excluding 0. On the OPUS-MT outputs alone the AUROC gap is within noise (−0.013
      [−0.051, +0.017]). **v1 − base:** AUROC +0.071 [+0.036, +0.119].
    - **v1's review line is far too quiet here:** recall 0.20 against its dev target of 0.60 and
      0.48 on WMT en-es test. The F1 line flags 4 units, all Major. The line was not moved.
    - **False flags:** v1's 5 are all on minor errors. On DeepSeek's 716 OK units: v1 2, Gemma 16,
      base 25. The official Spanish scored as MT (720 segments): v1 3 flags, Gemma 79.
    - The first pass (REALDOCS.md: 3 documents, 9 majors, 3 caught) is superseded and its labels
      do not carry over.
- **The static demo is at the review line:** the FDA page, 39 segments, 3 flagged (2 verified
  majors and 1 minor error), rebuilt through the backend.
- **`/api/health` names the adapter version (v1) and the active operating point.** The backend
  finds the version from the adapter file's sha256.
- **The rule checks are calibrated on train** (RESULTS.md, "Rule calibration").
  - The app's score is the model log-odds plus each finding's train-measured weight
    (`secondread/rule_calibration.json`).
  - On test, AUROC is now −0.002 against v1 alone on both pairs; it was −0.010 and −0.011.
  - The findings are still shown as reasons.

## Spend (budgets: Tinker about $18 available, DigitalOcean $5 credit)

| | Spent | Source | Left |
|---|---|---|---|
| **Tinker** | **$3.424 billed** through the 20:00–21:00 UTC bucket, read at 23:55 CEST: v1 and everything before it, v2 ($1.716), and $0.043 of DeepSeek-V3.1 sampling for the real-document gold set. **About $6.27** once v3 lands: v3a ≈ $1.36 and v3b ≈ $1.48, not in the bill yet | `tinker -f json billing usage`; v3 is estimated from its token counts at $0.737/M (RESULTS.md, "The candidates") | **about $11.7** of ~$18 |
| Tinker storage | 22 checkpoints before v2 (2.83 GB, $0.28/month), plus those of v2, v3a and v3b (periodic ones expire after 7 days, `final` never) and their external-weights exports | `tinker billing checkpoint-storage` | ongoing |
| **DigitalOcean** | **at most $1.64** at list price | token counts in the caches: baseline track $0.413 (WMT $0.372 for 10,236 calls; 696 calls for REALDOCS.md, $0.012, `data/realdocs/scored_summary.json`; 1,739 calls for the gold set, $0.029, `runs/gold_eval/meta_gemma*.json`), synthetic data $0.418 (5,626 calls), wrong-sense data for v3 $0.807 (7,101 calls, `data/synth/sense.report.json`) | **at least $3.36** of $5 |
| Sentry | $0 | Developer plan | |
| Ship phases | $0 | local only; one `tinker billing usage` read each | |
| Gold-set scoring (15 Oct) | $0.029 DigitalOcean, $0 Tinker | 1,059 Gemma calls for the gold units and 680 for the official-Spanish control, under a $0.15 cap; 421 byte-identical requests came from the cache | |

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
- **`:9931` serves LoRA v1 with `-cram 0`** (PID 39580, restarted 14 Oct 23:45 CEST with `nohup`
  from the repo root; log `/tmp/claude-501/llama-9931.log`; `/lora-adapters` lists
  `models/secondread-lora.gguf` at scale 1.0). The gold-set scoring used it without a restart and
  ran the base model on its own server on `:9980`, stopped afterwards:
  ```sh
  llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9931
  ```
- **No backend is running** once the Ship phase ends. It ran one on `:8787` (Sentry off) for the
  demo rebuild, the checks and the video capture. The v3 phases used their own llama-servers on
  `:9970`–`:9973` and stopped them.
- **The static demo on App Platform** builds from `main` (`deploy/app.yaml`,
  `npm ci && npm run build:demo`).

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
| **Review operating point** (night) | `pick_thresholds.py --target-recall 0.60`, backend checks on `:9970` | RESULTS.md, "Review operating point". v1: en-ru 0.1857, en-es 0.4984, the new backend default |
| **v3 track** (night) | `synth_sense.py` (7,101 Gemma calls), `train_tinker.py` for v3a and v3b, export, full dev splits and the synthetic sense set scored locally, `select_v3.py` | RESULTS.md, "v3 (wrong-sense synthetic) — dev selection". Neither candidate is eligible; **v1 kept** |
| **Test and real documents** (night) | `final_test.py`, `realdocs_score.py --no-gemma` on `:9973`, `realdocs_eval.py` | RESULTS.md, "v3 — test and real documents"; REALDOCS.md rewritten for both lines. en-es review-line recall 0.482 on test |
| **Ship phase** (night) | `:9931` restarted with v1; 40 en-ru dev rows; backend on `:8787`; `make_demo_results.py --api … --doc fda-medicine-disposal --n 0`; `npm run build` and `build:demo`; `tests/test_xliff.py`, `tests/test_thresholds.py`, `canary_check.py` | Max difference 0.00044 against `results/local_v1_enru_dev.jsonl`. `/api/health`: `adapter_version: v1`, `operating_point: review`. Demo: 3 of 39 flagged at 0.4984 (was 1 at the F1 line). Tests 20/20 and 7/7; the canary is in 0 of 12 envelopes (4 with the scrubbers off). RESULTS.md, "v3 — test and real documents", "Ship" |
| **Gold-set scoring** (15 Oct, 00:18–00:40) | `gold_score.py --export-inputs`; `gold_predict.py` v1 (`:9931`), base (`:9980`, LoRA scale 0), gemma, control and freeze; `gold_score.py` for both labellings; `gold_analyze.py`; an independent re-derivation | REALDOCS_GOLD.md, section 8. v1 AUROC 0.910, review-line recall 0.202 at precision 0.773; Gemma 0.940 and 0.821. The backend's flags equal `risk >= t` on every unit, and v1 risks match the earlier real-document snapshot within 0.0011. The re-derivation matched every headline number and fixed one rounding slip |

## Needs you

- **en-es review line:** it finds 48% of the Majors on WMT test, not 60%, and only 20% on the
  real-document gold set, where v1's risks run low (median 0.26 on the Majors) while its ranking
  holds (AUROC 0.910). Decide how to present it (README, RESULTS.md and REALDOCS_GOLD.md say it
  plainly). Any new line must be picked on new dev data, for example held-out real public
  documents, or reported as informed by these results.
- **Rule-only mode no longer flags.** With llama-server down, a segment's risk is now the largest
  train P(Major | finding), at most 0.47, so nothing reaches the 0.5 fallback threshold. The
  findings are still listed and sort first. Decide whether that mode should flag every finding,
  for example with a lower rules-only threshold.
- **Rotate `DO_MODEL_KEY`** if any agent transcript from today will be shared. Part of the
  key appeared in a process listing in one track's tool output. No file holds it.
- **Checkpoint cleanup.** Decide whether to delete the G3, G4, LR-probe and v3 checkpoints on
  Tinker. `final` checkpoints never expire.
- **The post:** the draft (kept outside the repo) now reports the gold set in its real-documents
  section. It still needs the title, the video upload, screenshots, the first user and the
  optional agent session. Re-check the Tinker total against the bill first (v3 is not billed
  yet).
- **G8:** check that the review traces are in Sentry's Agents dashboard and take the
  screenshot. `SENTRY_DSN` is set, and the latency track sent about 22 traces with metadata
  only.
- **The tester session,** the post.

## Not built yet

- **Annotated-file download:** `/api/review` returns XLIFF keys as `id`, and
  `xliff.annotate()` works. But no endpoint returns the annotated file yet.
- **Rule false alarms (found on test, not fixed):** "£" vs «фунтов», "whether or not", "no
  doubt", "24/7" vs «круглосуточно», ordinal words («втором»). The calibrated weights make them
  cheap, but they still show up as reasons. Fixing them in `checks.py` would be test-informed;
  report it that way (RESULTS.md, "Rule calibration").
- **Other eval sets:**
  - WMT25 en-ru OOD conversion (only `scripts/wmt25_count.py` exists);
  - ACES and HalOmi loaders (eval only, NC licenses).
- **`pyproject.toml` gaps:**
  - `httpx` and `numpy` are imported directly but only installed transitively;
  - `pytest` is absent.
- **Licence:** MIT in `LICENSE`, third-party notes in `NOTICE`. `prompts/gemba_mqm.txt` is
  CC BY-SA 4.0 and stays outside the code licence; its header says so.
- **A real-document dev set.** The gold set (`docs/REALDOCS_GOLD.md`) is test only. A new en-es
  line for documents like these needs held-out real documents, labelled the same way.
- **Sentry span limit:** Sentry keeps about 1,000 spans per transaction, so jobs over about
  990 segments lose some `chat` spans.

## Next steps, in order

Done: v1 served and calibrated at two operating points, v2 and v3 trained and rejected on dev,
the review line measured once on test and on the real documents, the static demo rebuilt at the
review line and redeployed, and the real-document gold set scored once (REALDOCS_GOLD.md,
section 8). What is left:

1. **G8:** the Sentry screenshot and the debugging story, the `-cram 0` latency hunt. Run the
   backend with `.env` loaded so `SENTRY_DSN` is set.
2. **Annotated-XLIFF download** endpoint and button, the tester session, and the post's remaining
   placeholders.
3. **Optional, free:** evaluate the one-pass v1+v2 blend, with both adapters loaded at scale 0.5,
   on dev first (RESULTS.md, "Next experiments").
4. **Optional, about $0.10:** Tinker exact eval of v1 on en-ru test. G5 already shows local
   tracks Tinker.
5. **Later:** a held-out real-document dev set, if the en-es line is to be re-picked for documents
   like the gold set's.

## Changed in the gold-set scoring pass (2026-10-15)

- **Scores:** `docs/REALDOCS_GOLD.md` section 8, computed once under the rules of its section 6.
  The prediction files were frozen (sha256 in `runs/gold_eval/PREDICTIONS_MANIFEST.json`, 00:37
  CEST) before `scripts/gold_score.py` first read `labels.jsonl`. No label, threshold, prompt,
  rule weight or model was changed or chosen on the set. A separate skeptic pass rebuilt the
  headline numbers with its own code and found one rounding slip in a table cell (fixed).
- **New scripts:** `scripts/gold_predict.py` (predictions from the blind inputs only; it refuses
  to run with `SECONDREAD_THRESHOLD` or `SECONDREAD_OPERATING_POINT` set) and
  `scripts/gold_analyze.py` (error analysis; it refuses prediction files that changed after the
  freeze).
- **Committed results:** `results/gold_eval/report_primary.json`, `report_preregistered.json` and
  `PREDICTIONS_MANIFEST.json`: metrics, counts and sha256 only, no text and no labels. The
  predictions, the analysis (it quotes MT spans) and the labels stay in the git-ignored
  `runs/gold_eval/` and `data/realdocs/gold/`.
- **Docs:** `docs/REALDOCS.md` is now the superseded first pass and points to REALDOCS_GOLD.md;
  `docs/RESULTS.md` (pointers in "v3 — test and real documents"), `README.md` and this file.
- **Servers and spend:** v1 on `:9931` (not restarted), base on `:9980` at LoRA scale 0 (stopped
  afterwards); $0.029 DigitalOcean.

## Changed in the v3 and Ship passes (committed together with the review operating point pass)

- **v3 data and selection:** `scripts/synth_sense.py` (wrong-sense generator, with the
  real-document integrity word list), `scripts/subsample_synth.py`, `scripts/mix_synth.py`,
  `scripts/verify_synth.py sense` (integrity checks), `scripts/select_v3.py`.
- **Test and real documents:** `scripts/final_test.py`, `scripts/realdocs_eval.py`.
- **Thresholds:** `scripts/pick_thresholds.py` lists v3a and v3b; `secondread/thresholds.json` has
  their `review` blocks. v1's blocks and the default point are unchanged.
- **Scores** (`results/`, ids, labels and scores only): `local_{v3a,v3b}_{enru,enes}_dev.jsonl`,
  `local_{v1,v3a,v3b}_synth_sense_{test,parents}.jsonl`.
- **Real-document gold set (separate track, scored on 15 Oct; see the next section):**
  `docs/REALDOCS_GOLD.md`, `docs/ANNOTATION_PROTOCOL.md`, `scripts/gold_*.py`.
- **Demo:** `ui/public/demo-results.json`, the FDA page through the backend at the review line.
- **Small fixes:** `scripts/export_adapter.sh` starts its check server with `-cram 0`.
- **Docs:** `README.md`, `docs/RESULTS.md` (v3 sections and "Ship"), `docs/REALDOCS.md`,
  `docs/RUNBOOK.md`, this file.

## Changed in the review operating point pass (committed with the Ship phase)

- **Thresholds:**
  - `scripts/pick_thresholds.py` has a new `--target-recall R` mode, the `review` point. It takes
    the highest threshold with dev recall ≥ R on the fused score, rounded down, with
    document-bootstrap CIs. It reads dev only, and reports the pooled threshold when a recall CI
    is wider than ±0.15.
  - The F1 mode is unchanged and now writes the `f1` point. `--skip-test` leaves out its test
    statistics, and each mode keeps the other's blocks.
  - `secondread/metrics.py` has the new `max_threshold_at_recall`.
  - `secondread/thresholds.json`, new layout:
    - `default_operating_point: "review"`;
    - `operating_points` with each point's method;
    - `review` and `f1` blocks per adapter. The `f1` values are moved over unchanged.
- **Backend:**
  - `secondread/agent.py`:
    - the operating point comes from the request, then `SECONDREAD_OPERATING_POINT`, then the
      file, then `review`;
    - `threshold_source` is `dev_review` or `dev_f1`;
    - the summary has `operating_point`;
    - an unknown point is refused;
    - it warns when a point's rule-calibration sha256 is stale.
  - `secondread/server.py`:
    - `operating_point` is accepted in JSON, multipart and the query string;
    - `/api/health` has `operating_point`, `operating_point_source` and `operating_points`.
- **Tests:** `tests/test_thresholds.py` (new, 7 tests). Among them: the review mode opens no test
  file.
- **Docs:** `ui/API.md`, `README.md`, `docs/RUNBOOK.md`, `docs/RESULTS.md` (new section "Review
  operating point" and a note in "Ship decision"), this file.

## Changed in the rule-calibration pass (committed in b52bd19)

- **New:**
  - `secondread/rulecal.py`: the combination, logit(risk) = logit(model) + the strongest weight
    of each check;
  - `secondread/rule_calibration.json`: weights and train statistics, numbers only, no text;
  - `scripts/calibrate_rules.py`: the train fit, plus each candidate's combined scores for dev
    and test;
  - `results/local_v1_{enru,enes}_train.jsonl`: v1 scores of the train rows (ids, labels and
    scores only).
- **Backend:**
  - `secondread/agent.py`: `aggregate` uses the calibration. Reasons carry the train
    P(Major | finding) as `confidence` plus a `weight`, and `signals.*.risk` is that probability;
  - `secondread/server.py`: `/api/health` has `rule_calibration`;
  - `secondread/checks.py`: the docstrings now say the hand-set confidences are a fallback only.
- **Thresholds:**
  - `scripts/pick_thresholds.py` fuses with the new `aggregate` and records the sha256 of the
    calibration file;
  - `secondread/thresholds.json` re-picked: v1 is unchanged, base en-es goes 0.7506 → 0.7345.
- **UI and contract:**
  - `ui/public/demo-results.json`, rebuilt through the backend: only segment 10 changes, 0.896 →
    0.675, still flagged;
  - `ui/API.md`: `confidence`, `weight` and `rule_calibration` documented;
  - `ui/dist`, rebuilt with `build:demo` last.
- **Docs:** `docs/RESULTS.md` (new section "Rule calibration", plus pointers in "Ship decision"),
  this file, `README.md` and `docs/RUNBOOK.md`.
- **Git-ignored:** `runs/rulecal/`, the candidates' combined scores and the app's fused scores.

## Changed in the Ship phase (committed in 00b756d)

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
