# SecondRead runbook: Tinker v1 run, evaluation, export

Commands assume the repo root and this setup in each new shell. The keys stay in `.env`;
nothing below prints them.

```sh
cd secondread
set -a && . ./.env && set +a
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
```

## 1. The v1 run (2026-10-14, 13:53:04 → 14:02:28 CEST, finished)

| | |
|---|---|
| Command | `nohup py scripts/train_tinker.py --data data/sft_v1.jsonl --log-path runs/v1 --lr 2e-4 --save-every 50 --dev-split data/splits/en-ru.dev.jsonl --dev-n 200 --eval-every 50 > runs/v1/train.log 2>&1 &` |
| PIDs | `uv` 24293, its `python3` child 24296 (reparented to launchd, so it survives the shell). Both have exited |
| Tinker run | `479fd56b-2a66-539a-ae24-5b020ba97bbb:train:0` ([console](https://tinker.thinkingmachines.ai/training_runs/479fd56b-2a66-539a-ae24-5b020ba97bbb:train:0)) |
| Data | `data/sft_v1.jsonl` (sha 32736ed384d6092a): 14,791 rows of human labels only, 1 epoch = 231 steps of 64 (7 rows dropped) |
| Model | Qwen/Qwen3.5-4B, LoRA rank 16, MLP only, LR 2e-4 linear decay to 0, seed 0 |
| Checkpoints | state + sampler weights after steps 50, 100, 150, 200 (TTL 7 days), `final` after 231 (no TTL) |
| Dev scoring | 200 en-ru dev rows (the first 200 of the `--n 300` sample), exact p(Major)/p(OK), at steps 0, 50, 100, 150, 200 and after the last step |
| Duration | 9 min 24 s for 231 steps, 6 dev evaluations and 5 checkpoints (2.9–5.7 s per step). Expect up to about 46 min at the 12 s per step seen in G3 |
| Cost | 1,452,983 input tokens: **$1.07**, plus about $0.15 for the 6 dev evaluations (200,676 forward tokens). The bill confirms that every token is charged, weight-0 prompt tokens included (see Spend below) |

**Why LR 2e-4 and not the default 4.905e-4 (`get_lr`):** G3 at 4.9e-4 swung from
"always OK" (Major NLL 18.3 at step 1) to "always Major" (OK NLL 14.4 at step 3). Constant-LR
probes of 20 steps (`runs/lrprobe_*`) were stable after step 3 at 2e-4, 1e-4 and 5e-5. On
100 dev rows, AUROC went from 0.604 to 0.691, 0.670 and 0.671; PR-AUC went from 0.407 to 0.511,
0.455 and 0.450. 2e-4 still swings at step 1 (Major NLL 15.6) but recovers by step 4.

### Monitor

```sh
ps -p 24296 -o pid,etime,command | cut -c1-120          # still running?
tail -n 3 runs/v1/train.log                               # last log lines, errors
cat runs/v1/checkpoints.jsonl                             # checkpoints so far
python3 -c "
import json
for l in open('runs/v1/metrics.jsonl'):
    m = json.loads(l); s = m.get('step')
    if 'dev/auroc' in m:
        print('DEV', s, 'auroc=%.3f pr_auc=%.3f label_nll=%.3f mean_risk=%.3f' % (m['dev/auroc'], m['dev/pr_auc'], m['dev/label_nll'], m['dev/mean_risk']))
    elif s % 10 == 0:
        print(s, 'label_nll=%.3f lr=%.2e step_s=%.1f' % (m['label_nll'], m['learning_rate'], m['time_step']))
"
```

The run is done when `train.log` ends with `sampler weights: tinker://.../sampler_weights/final`
and `checkpoints.jsonl` has a row with `"name": "final"`. v1 has both.

v1 dev scoring on 200 en-ru dev rows (`runs/v1/metrics.jsonl`):

| step | 0 | 50 | 100 | 150 | 200 | 231 |
|---|---|---|---|---|---|---|
| AUROC | 0.552 | 0.695 | 0.733 | 0.728 | 0.733 | 0.739 |
| PR-AUC | 0.324 | 0.481 | 0.521 | 0.502 | 0.524 | 0.528 |
| label NLL | 0.771 | 0.657 | 0.532 | 0.613 | 0.609 | 0.575 |
| mean risk | 0.487 | 0.122 | 0.299 | 0.251 | 0.165 | 0.178 |

Mean training label NLL per 25-step block: 0.690, 0.383, 0.385, 0.326, 0.300, 0.299, 0.293,
0.270, 0.273, and 0.221 for the last 6 steps.

If the process dies, rerun the same command. It resumes from the last checkpoint in `runs/v1`,
with the same batches and the same LR schedule.

### Spend

```sh
uv run --frozen tinker -f json billing usage 2026-10-14T00:00:00Z 2026-10-15T00:00:00Z   # lags by up to a few hours
uv run --frozen tinker billing checkpoint-storage
```

**What the bill showed** (read at 15:50 CEST, data through the 13:00–14:00 CEST bucket): $1.334 in total.
- **Every token is billed.** G3 was billed for 62,820 training tokens and G4 for 31,353,
  exactly the `num_tokens` in their logs. Weight-0 prompt tokens are charged at the
  training rate, so an epoch costs $0.737/M × all tokens, not $0.02.
- **Dev scoring inside `train_tinker.py`** is billed as training: each LR probe was billed
  for 159,307 tokens, which is 125,171 training plus 34,136 for two dev evaluations.
- **Sampling** (`eval_tinker.py`) costs $0.33/M prompt tokens plus $1.005/M sampled tokens,
  about $0.029 per 1k en-ru segments.
- **Storage:** 22 checkpoints, 2.83 GB, $0.28 a month.
- **Expected total** once the 14:00–15:00 bucket lands is about $1.66 for all of today's
  Tinker work.

## 2. Paths of the final checkpoint

```sh
FINAL=$(python3 -c "import json; r=[json.loads(l) for l in open('runs/v1/checkpoints.jsonl') if '\"final\"' in l][-1]; print(r['sampler_path'], r['state_path'])")
SAMPLER=${FINAL% *}   # tinker://479fd56b-2a66-539a-ae24-5b020ba97bbb:train:0/sampler_weights/final
STATE=${FINAL#* }     # tinker://479fd56b-2a66-539a-ae24-5b020ba97bbb:train:0/weights/final
echo "$SAMPLER $STATE"
```

## 3. Evaluate on Tinker (exact logprobs, about $0.33 for all four splits)

Each output is resumable: a rerun skips ids already in `--out`.

```sh
py scripts/eval_tinker.py data/splits/en-ru.dev.jsonl  --model-path "$SAMPLER" --out results/v1_enru_dev.jsonl    # 2,972 rows, ~$0.09
py scripts/eval_tinker.py data/splits/en-ru.test.jsonl --model-path "$SAMPLER" --out results/v1_enru_test.jsonl   # 3,573 rows, ~$0.11
py scripts/eval_tinker.py data/splits/en-es.dev.jsonl  --model-path "$SAMPLER" --out results/v1_enes_dev.jsonl    # 1,650 rows, ~$0.06 (en-es threshold)
py scripts/eval_tinker.py data/splits/en-es.test.jsonl --model-path "$SAMPLER" --out results/v1_enes_test.jsonl   # 1,815 rows, ~$0.08
```

The base model on the same splits, for the baseline columns (about another $0.33):

```sh
py scripts/eval_tinker.py data/splits/en-ru.dev.jsonl  --base --out results/tinker_base_enru_dev.jsonl
py scripts/eval_tinker.py data/splits/en-ru.test.jsonl --base --out results/tinker_base_enru_test.jsonl
py scripts/eval_tinker.py data/splits/en-es.dev.jsonl  --base --out results/tinker_base_enes_dev.jsonl
py scripts/eval_tinker.py data/splits/en-es.test.jsonl --base --out results/tinker_base_enes_test.jsonl
```

Report: the threshold is picked on dev and applied unchanged to test, with a document-level
bootstrap.

```sh
py scripts/report.py base=results/tinker_base_enru_test.jsonl,results/tinker_base_enru_dev.jsonl \
                     v1=results/v1_enru_test.jsonl,results/v1_enru_dev.jsonl --n-boot 1000
py scripts/report.py base=results/tinker_base_enes_test.jsonl,results/tinker_base_enes_dev.jsonl \
                     v1=results/v1_enes_test.jsonl,results/v1_enes_dev.jsonl --n-boot 1000
```

## 4. Export to GGUF and load locally

`tinker checkpoint download` hung for more than 10 minutes on 2026-10-14. It uses the
checkpoint-archive endpoint, which tinker 0.33.1 marks deprecated. Called directly from the
SDK, the same request ended in `tinker.APITimeoutError` after about 10 minutes. `scripts/fetch_adapter.py`
uses the supported path instead: it loads the **state** checkpoint into a fresh MLP-only
training client, then calls `save_weights_external` and downloads the signed file URLs. In
G4 this took 5 s for the save and a few seconds for 72 MB. It starts a new Tinker training
run id, but no training tokens are used.

```sh
py scripts/fetch_adapter.py "$STATE" --out ckpt
#   -> ckpt/<new-run-id>_external_weights_final/{adapter_model.safetensors, adapter_config.json}
scripts/export_adapter.sh --port 9936 ckpt/<new-run-id>_external_weights_final models/secondread-lora.gguf
```

**Done for v1 (2026-10-14 14:03 CEST):** `save_weights_external` returned
`tinker://e2cadc53-a67c-59fb-9233-b13ab8dae922:train:0/external_weights/final` in 2 s. The
files are in `ckpt/e2cadc53-a67c-59fb-9233-b13ab8dae922:train:0_external_weights_final/`,
and `models/secondread-lora.gguf` is 36 MB. llama-server reported "loaded 192 tensors from
lora file". The smoke test, scale 0 → 1:

| example | risk at scale 0 | risk at scale 1 |
|---|---|---|
| dropped negation (Major) | 0.882 | 0.826 |
| wrong number (Major) | 0.847 | 0.826 |
| faithful (OK) | 0.259 | 0.003 |

`:9931` was not restarted at export time. **Ship phase (2026-10-14, 18:00 CEST):** it now serves
v1 with `-cram 0` (command below), started with `nohup` from the repo root, log
`/tmp/claude-501/llama-9931.log`.

`export_adapter.sh` defaults to port 9932. Pass a free port: 9931 is the app's server, and
9932 was in use by another job on 2026-10-14. The script runs a smoke test on that port (adapter
loaded, P(Major) at scale 0 vs 1) and then stops the server.

To serve the adapter, restart the app's server with the LoRA. This restarts `:9931`, so
check that nothing else needs the base-only server first:

```sh
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9931
```

## 5. G5: Tinker sampler vs local Q4 + LoRA

Run this on the same 300 dev rows as the baselines. The local server needs the v1 GGUF; use
a spare port such as 9936 and stop the server afterwards.

```sh
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9936 &
py scripts/probe_local.py data/splits/en-ru.dev.jsonl 300 --lora-scale 1 --port 9936 --out results/local_v1_enru_dev300.jsonl
py scripts/eval_tinker.py data/splits/en-ru.dev.jsonl --model-path "$SAMPLER" --n 300 --out results/tinker_v1_enru_dev300.jsonl
kill %1
py scripts/report.py tinker=results/tinker_v1_enru_dev300.jsonl local=results/local_v1_enru_dev300.jsonl \
   --ref data/splits/en-ru.dev.jsonl --n-boot 1000
```

Target: Spearman of the risks ≥ 0.9 and mean difference ≤ 0.05. For base vs base on 300 rows,
Spearman was 0.944, mean difference +0.018 and mean |difference| 0.056.

**Done for v1: G5 passes.** Spearman 0.989, Pearson 0.990, mean difference (Tinker − local)
−0.025, mean |difference| 0.029, max |difference| 0.207. On these 300 en-ru dev rows (30
documents, Major rate 0.247, document bootstrap with 1,000 replicates):

| system | AUROC | PR-AUC | recall@10% |
|---|---|---|---|
| base, local Q4 | 0.565 | 0.332 | 0.149 |
| base, Tinker | 0.547 [0.455, 0.644] | 0.309 [0.203, 0.435] | 0.149 [0.062, 0.224] |
| v1, local Q4 + LoRA | 0.717 [0.628, 0.796] | 0.488 [0.340, 0.622] | 0.230 [0.155, 0.322] |
| v1, Tinker | 0.716 [0.624, 0.796] | 0.478 [0.327, 0.614] | 0.234 [0.152, 0.313] |

Paired against base local: v1 local ΔAUROC +0.151 [+0.070, +0.229], ΔPR-AUC +0.156
[+0.063, +0.233]. These are dev rows, and 100 of them were used to choose the LR. Headline
numbers come from the test splits (section 6 and `docs/RESULTS.md`).

## 6. Score full splits locally (free)

Because G5 passes, local Q4+LoRA scores can stand in for Tinker scores, and they cost nothing.
The base model runs on the app server and v1 on a second server. On 2026-10-14 both ran at
once and shared the GPU:
- en-ru test: 2 × 3,573 rows in 19.7 min;
- en-es test: 2 × 1,815 rows in 13.7 min.

```sh
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 \
  --host 127.0.0.1 --port 9936 & SRV=$!
until curl -sf http://127.0.0.1:9936/health >/dev/null; do sleep 1; done
for s in en-ru.test en-es.test en-es.dev; do t=$(echo $s | tr -d - | tr . _)
  py scripts/probe_local.py data/splits/$s.jsonl 100000 --workers 4 --out results/local_base_$t.jsonl & B=$!  # :9931, base
  py scripts/probe_local.py data/splits/$s.jsonl 100000 --workers 4 --lora-scale 1 --port 9936 \
     --out results/local_v1_$t.jsonl; wait $B
done
kill $SRV
```

- **`N` = 100000** means "every row": `probe_local.py` shuffles with seed 0 and takes the
  first N rows.
- **`--workers 4`** matches `-np 4`. It changes risks by at most 0.003 against sequential
  scoring.
- **The output file is written only at the end,** so an interrupted run starts again from
  scratch.
- **The full en-ru dev split has not been scored locally yet.** The en-ru F1 thresholds
  therefore come from the 300-row dev files. About 10 min: same loop with `s=en-ru.dev`.
- **Report commands** are in `docs/RESULTS.md`.

## 7. Ship: serve, calibrate, rebuild the demo (2026-10-14, evening, done; rule calibration and the review line added at night, last run 23:40 CEST)

```sh
# serve the shipped adapter on the app port (stop the old server first)
nohup llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 \
  --host 127.0.0.1 --port 9931 > /tmp/claude-501/llama-9931.log 2>&1 < /dev/null &
curl -s http://127.0.0.1:9931/health; curl -s http://127.0.0.1:9931/lora-adapters
# rule weights from the train splits (RESULTS.md, "Rule calibration"); needs no server
py scripts/calibrate_rules.py                                # -> secondread/rule_calibration.json
# thresholds per adapter and pair from the full dev splits (fused risk), one block per operating point;
# each run rewrites only its own point and keeps the other
py scripts/pick_thresholds.py --fused-dir runs/rulecal/app   # "f1": F1-optimal (reads test for information)
py scripts/pick_thresholds.py --target-recall 0.60           # "review", the backend default: highest
                                                             # threshold with dev recall >= 0.60; dev only
py tests/test_thresholds.py                                  # rule, file and backend choice, no server
# backend (no keys needed; without .env Sentry stays off), demo results, UI builds
py -m secondread.server &                                    # 127.0.0.1:8787
curl -s http://127.0.0.1:8787/api/health                     # adapter "secondread-lora v1", operating_point, thresholds
py scripts/make_demo_results.py --api http://127.0.0.1:8787 \
   --answer-key demo/answer_key.en-es.json --doc fda-medicine-disposal --n 0   # ui/public/demo-results.json (FDA page)
kill %1
(cd ui && npm run build && npm run build:demo)               # both write ui/dist; the last one wins
py tests/test_xliff.py && py scripts/canary_check.py
```

`models/secondread-lora.gguf` is v1 (sha256 `5337b2b7…`), `models/secondread-lora-v1.gguf` is a
copy. `models/secondread-lora-v2.gguf` (`9dda3f68…`), `-v3a.gguf` (`eac1fb1a…`) and `-v3b.gguf`
(`64f78f05…`) are kept. To switch adapters, restart `:9931` with the other `--lora`. The backend
recognises all four by hash and applies their own thresholds. v3a and v3b have only a `review`
block, so at `f1` they get the 0.5 fallback, as does an unknown file.

The backend flags at the `review` operating point unless told otherwise:
`SECONDREAD_OPERATING_POINT=f1 py -m secondread.server` switches the server's default, and a request
can send `"operating_point": "f1"` (JSON or multipart field, or `?operating_point=f1`). An unknown value
is refused: a 400 `bad_operating_point` for a request, a start-up error for the variable.

## Results so far (2026-10-14)

| Step | Result |
|---|---|
| G3 | 10 steps at batch 64, `runs/g3`. 99 s wall time in total, about 11–13 s per step when run alone. Label NLL by step: 0.75, 4.12, 1.78, 12.2, 1.05, 0.72, 1.24, 1.11, 1.24, 0.75 (unstable at LR 4.9e-4). 62,820 input tokens |
| G4 | 5 steps, `runs/g4` → `fetch_adapter.py` → `export_adapter.sh` → `models/secondread-lora-g4.gguf` (36 MB, 192 tensors loaded by llama-server). Keys are the same as the fake adapter (`base_model.model.model.layers.N.mlp.{gate,up,down}_proj.lora_{A,B}.weight`, 96 pairs). The dtype is float32, not bf16. `adapter_config.json` is a full PEFT config with `target_modules: "all-linear"`, `r=16`, `lora_alpha=32`. No code change was needed. This 5-step adapter is not a classifier: every top-1 is OK |
| Baseline, Tinker | `results/tinker_base_enru_dev300.jsonl`: AUROC 0.547, PR-AUC 0.309, recall@10% 0.149. The local Q4 baseline gives 0.565, 0.332 and 0.149. Spearman of the risks 0.944, Pearson 0.951, mean difference +0.018 |
| G5 on the G4 adapter (100 rows) | Spearman 0.856, mean difference −0.043. The Tinker risk is lower than the local one on every row. The adapter's risks span only 0.05–0.2, so the ranks are noisy. On v1 it passes (section 5) |
| LR probes | `runs/lrprobe_{2e-4,1e-4,5e-5}`: 20 steps each at a constant LR, with 100 dev rows scored at steps 0 and 20 (see section 1) |
| v1 | Section 1. Exported to `models/secondread-lora.gguf` (section 4); G5 passes (section 5) |
| Test splits, local | Section 6, `docs/RESULTS.md`. en-ru test: AUROC 0.666 (base) → 0.801 (v1), PR-AUC 0.369 → 0.537. en-es test: AUROC 0.776 → 0.815, PR-AUC 0.413 → 0.517 |
