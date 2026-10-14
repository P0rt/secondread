# SecondRead results, 2026-10-14

The task is to rank segments by the risk of a major meaning error. `Major` is the gold label when
the unit has at least one major or critical MQM error in the meaning family. That family is
accuracy, terminology and non-translation (`scripts/convert.py`). Each WMT unit has one annotator,
so the labels are noisy.

**How the numbers are computed.** Every row comes from `scripts/report.py` with 1,000
document-level bootstrap replicates (whole documents are resampled). The systems in one block
are compared on the same rows with the same replicates, so the differences below are paired.

- **F1:** each system's F1-optimal threshold is picked on its own dev scores and then applied
  unchanged to test.
- **recall@10%:** the share of Major units in the riskiest 10% of the segments. That is the
  review budget the tool is built for. A random ranking gets 0.100.
- **PR-AUC:** with a random ranking it equals the Major rate.

**Systems:**

| Name | What it is | Where it runs |
|---|---|---|
| base, local | Qwen3.5-4B Q4_K_M GGUF, no adapter, first-token p(Major)/(p(Major)+p(OK)) | llama-server, this Mac |
| base, Tinker | the same prompt on the bf16 base model, exact label logprobs (`eval_tinker.py --base`) | Tinker |
| **v1, local** | Q4_K_M plus LoRA v1 (`models/secondread-lora.gguf`, 36 MB) | llama-server, this Mac |
| v1, Tinker | the v1 sampler weights, exact label logprobs | Tinker |
| Gemma direct | `gemma-4-31B-it`, same classifier prompt, label probability from top-10 logprobs | DigitalOcean serverless |
| Gemma GEMBA | `gemma-4-31B-it` with the GEMBA-MQM few-shot prompt; risk = GEMBA penalty over meaning-family errors / 25 | DigitalOcean serverless |

LoRA v1 is rank 16, MLP only. It was trained for one epoch on the 14,791 human-labelled train
units (`data/sft_v1.jsonl`, no synthetic data) at LR 2e-4. See `docs/RUNBOOK.md`.

## Main table

| Pair, split | System | n | AUROC [95% CI] | PR-AUC [95% CI] | recall@10% [95% CI] | F1 at dev threshold [95% CI] | $ per 1k segments | Speed |
|---|---|---:|---|---|---|---|---:|---|
| **en-ru test** (33 docs, Major 0.202) | base, local | 3,573 | 0.666 [0.606, 0.718] | 0.369 [0.256, 0.469] | 0.227 [0.184, 0.288] | 0.339 [0.246, 0.418] ¹ | 0 | ~0.17 s/seg ² |
| | **v1, local** | 3,573 | **0.801** [0.743, 0.851] | **0.537** [0.407, 0.635] | 0.303 [0.246, 0.380] | **0.535** [0.443, 0.606] | 0 | ~0.17 s/seg ² |
| | Gemma direct | 3,573 | 0.794 [0.754, 0.840] | 0.494 [0.390, 0.588] | **0.307** [0.254, 0.385] | 0.519 [0.428, 0.589] | 0.017 | p50 0.58 s, p95 0.99 s ³ |
| **en-ru test, GEMBA sample** (500 rows, seed 0; 33 docs, Major 0.188) | base, local | 500 | 0.691 [0.612, 0.763] | 0.429 [0.297, 0.580] | 0.277 [0.197, 0.377] | 0.316 [0.223, 0.404] ¹ | 0 | as above |
| | **v1, local** | 500 | **0.838** [0.767, 0.896] | **0.611** [0.415, 0.755] | **0.372** [0.275, 0.470] | **0.539** [0.420, 0.635] | 0 | as above |
| | Gemma direct | 500 | 0.816 [0.760, 0.872] | 0.509 [0.385, 0.655] | 0.330 [0.239, 0.470] | 0.506 [0.410, 0.591] | 0.017 | as above |
| | Gemma GEMBA | 500 | 0.772 [0.718, 0.837] | 0.448 [0.326, 0.574] | 0.323 [0.243, 0.438] | 0.523 [0.426, 0.608] ⁴ | 0.242 (≈0.10 cached) | p50 1.58 s, p95 4.04 s ³ |
| **en-es test** (25 docs, Major 0.093) | base, local | 1,815 | 0.776 [0.723, 0.826] | 0.413 [0.334, 0.497] | 0.452 [0.358, 0.529] | 0.395 [0.336, 0.452] (thr 0.751) | 0 | ~0.2 s/seg ² |
| | **v1, local** | 1,815 | **0.815** [0.776, 0.853] | **0.517** [0.442, 0.591] | **0.518** [0.437, 0.602] | **0.429** [0.355, 0.498] (thr 0.702) ⁸ | 0 | ~0.2 s/seg ² |
| | Gemma direct | 1,815 | 0.807 [0.767, 0.843] | 0.311 [0.282, 0.368] | 0.458 [0.367, 0.558] | 0.363 [0.315, 0.420] ⁵ | 0.024 | as above |
| **en-ru dev, 300-row sample** (30 docs, Major 0.247) ⁶ | base, local | 300 | 0.565 [0.470, 0.660] | 0.332 [0.219, 0.459] | 0.149 [0.071, 0.229] | (dev) | 0 | |
| | base, Tinker | 300 | 0.547 [0.455, 0.644] | 0.309 [0.203, 0.435] | 0.149 [0.062, 0.224] | (dev) | 0.029 ⁷ | batch |
| | v1, local | 300 | 0.717 [0.628, 0.796] | **0.488** [0.340, 0.622] | 0.230 [0.155, 0.322] | (dev) | 0 | |
| | v1, Tinker | 300 | 0.716 [0.624, 0.796] | 0.478 [0.327, 0.614] | **0.234** [0.152, 0.313] | (dev) | 0.029 ⁷ | batch |
| | Gemma direct | 300 | **0.741** [0.666, 0.808] | 0.446 [0.318, 0.595] | 0.216 [0.148, 0.287] | (dev) | 0.017 | |
| | Gemma GEMBA | 300 | 0.704 [0.625, 0.783] | 0.394 [0.281, 0.524] | 0.185 [0.117, 0.256] | (dev) | 0.242 | |
| **en-es dev** (1,650 rows, 12 docs, Major 0.058; the threshold source for en-es) | base, local | 1,650 | 0.801 [0.757, 0.815] | 0.329 [0.264, 0.420] | 0.463 [0.333, 0.523] | (dev) | 0 | |
| | **v1, local** | 1,650 | **0.869** [0.832, 0.886] | **0.453** [0.371, 0.548] | **0.600** [0.369, 0.639] | (dev) | 0 | |

Random ranking: AUROC 0.500; PR-AUC equal to the Major rate; recall@10% 0.100.

1. The base model's dev-optimal threshold (0.159, from the 300 dev rows) flags almost
   everything: precision 0.205, recall 0.992 on the full en-ru test. Base scores are poorly
   separated, so there is no useful operating point.
2. Throughput on this Apple Silicon Mac with `-np 4` and 4 requests in flight. For the full test
   runs, the base and v1 servers shared the GPU: en-ru took 7,146 requests in 19.7 min, and en-es
   took 3,630 requests in 13.7 min. The 51-segment demo through the backend takes 8.4–9.0 s with
   `-cram 0` (`README.md`). Each server alone, 300 dev rows with 4 in flight, including about 3 s of
   start-up: base 0.172 s/seg (en-ru) and 0.220 s/seg (en-es); v1 0.183 and 0.214. Per-request
   p50 was 0.60–0.69 s.
3. Per call, as measured at concurrency 8. The account limit of 120 requests/min caps
   throughput at about 0.5 s per segment.
4. GEMBA's threshold comes from only 300 dev rows (Gemma direct uses all 2,972). GEMBA's own
   rule, "at least one major or critical meaning error", gives precision 0.429 and recall
   0.670 here.
5. There is no Gemma en-es dev run, so this threshold (0.971) comes from en-ru dev. Qwen's
   en-es thresholds come from en-es dev (1,650 rows, 12 docs).
6. Dev rows only. 100 of these 300 rows were used to choose the v1 learning rate. G5 (local
   Q4+LoRA vs the Tinker sampler on these rows): Spearman 0.989, mean difference −0.025. For
   base vs base: Spearman 0.944, mean difference +0.018.
7. From the Tinker bill: prompt prefill at $0.33/M tokens and sampling at $1.005/M. The 300 rows
   cost $0.0086.
8. en-es dev has only 12 documents and 95 Major units, so the en-es thresholds are coarse.
   At 0.702, v1 gets precision 0.769 and recall 0.298 on en-es test. The F1 column for en-es is
   the weakest number in this table.

**$ per 1k segments.**
- Gemma prices are DigitalOcean list prices ($0.18/M input, $0.50/M output) applied to the
  token counts in the cache. Cached input is billed lower, so these are upper bounds.
- Local runs cost nothing beyond electricity.
- Training v1 cost about $1.07 on Tinker, plus $0.15 for dev scoring during training. That is a one-off cost and is not counted per segment.

## Paired differences on test

These use the same bootstrap replicates as the main table. A CI that excludes 0 is a
significant difference.

| Block | Comparison | ΔAUROC | ΔPR-AUC | Δrecall@10% |
|---|---|---|---|---|
| en-ru test | v1 − base | +0.135 [+0.090, +0.191] | +0.168 [+0.113, +0.216] | +0.076 [+0.044, +0.112] |
| en-ru test | Gemma direct − v1 | −0.006 [−0.047, +0.054] | −0.043 [−0.107, +0.049] | +0.004 [−0.041, +0.055] |
| en-ru test 500 | v1 − base | +0.147 [+0.094, +0.212] | +0.181 [+0.066, +0.253] | +0.096 [+0.011, +0.163] |
| en-ru test 500 | Gemma direct − v1 | −0.022 [−0.083, +0.059] | −0.102 [−0.215, +0.078] | −0.043 [−0.126, +0.069] |
| en-ru test 500 | Gemma GEMBA − v1 | −0.067 [−0.141, +0.017] | **−0.163** [−0.249, −0.043] | −0.050 [−0.108, +0.027] |
| en-ru test 500 | Gemma GEMBA − Gemma direct | **−0.045** [−0.087, −0.004] | −0.061 [−0.158, +0.004] | −0.007 [−0.092, +0.061] |
| en-es test | v1 − base | **+0.039** [+0.003, +0.070] | **+0.104** [+0.058, +0.159] | **+0.065** [+0.022, +0.122] |
| en-es dev | v1 − base | **+0.068** [+0.036, +0.110] | **+0.124** [+0.076, +0.182] | +0.137 [+0.000, +0.210] |
| en-es test | Gemma direct − v1 | −0.008 [−0.041, +0.016] | **−0.206** [−0.286, −0.099] | −0.060 [−0.130, +0.022] |

**What this shows:**
- **v1 beats the base model on every test block.** On en-ru test, AUROC rises 0.666 → 0.801
  and PR-AUC 0.369 → 0.537.
- **v1 matches Gemma 4 31B, a model about 8× larger, on ranking.** The AUROC differences are
  within ±0.02 and their CIs include 0.
- **v1 beats Gemma on PR-AUC for en-es, by +0.206.** Gemma direct's risks are saturated:
  27% of en-es test rows score ≥ 0.97 against a Major rate of 9.3%. Its riskiest segments are
  therefore one large tied group.
- **GEMBA-MQM is the weakest Gemma variant here,** and it costs about 14× more per segment
  than direct.

## Caveats

- **One annotator per unit.** WMT22 en-ru and WMT24 en-es have one MQM rater per unit.
  Because of that label noise, no system will look precise.
- **Contamination.** WMT22 and WMT24 have been public since 2022 and 2024. Gemma 4 and
  Qwen3.5 may have seen the source texts.
- **Non-greedy decoding on DigitalOcean.** At temperature 0, the first sampled token was not
  the most likely one in 8.8% of Gemma calls. Gemma direct sums the label probability over
  the first two positions (`scripts/baseline_gemma.py` docstring). In 15% of rows one label
  was missing from the top-10 list, so its probability is a bound.
- **Local batching noise.** Running 4 requests in flight changes local risks by up to about
  0.003 against sequential scoring. Ranking metrics are not affected at the reported precision.
- **The en-ru local thresholds come from the 300-row dev sample** (`results/base_enru_dev300.jsonl`
  and `results/local_v1_enru_dev300.jsonl`). The full en-ru dev split has not been scored
  locally; doing so is free and would make the F1 column steadier. **Update:** it is now
  scored for base, v1 and v2. With the full-dev threshold (0.161), v1's en-ru test F1 is
  0.504, not 0.535. See "v2 and ablation" below.
- **No v1 test numbers on Tinker.** They are not needed for the comparison: G5 passes, and
  Tinker scoring costs about $0.03 per 1k segments if wanted.

## Reproduce

```sh
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
# local scoring (free): base on the app server, v1 on a second llama-server started with --lora
py scripts/probe_local.py data/splits/en-ru.test.jsonl 100000 --workers 4 --out results/local_base_enru_test.jsonl
py scripts/probe_local.py data/splits/en-ru.test.jsonl 100000 --workers 4 --lora-scale 1 --port 9936 \
   --out results/local_v1_enru_test.jsonl
# (same for en-es.test, en-es.dev; Gemma files come from scripts/baseline_gemma.py, see README)
R=results
py scripts/report.py base_local=$R/local_base_enru_test.jsonl,$R/base_enru_dev300.jsonl \
   v1_local=$R/local_v1_enru_test.jsonl,$R/local_v1_enru_dev300.jsonl \
   gemma_direct=$R/gemma_direct_enru_test.jsonl,$R/gemma_direct_enru_dev.jsonl --n-boot 1000
py scripts/report.py base_local=$R/local_base_enru_test.jsonl,$R/base_enru_dev300.jsonl \
   v1_local=$R/local_v1_enru_test.jsonl,$R/local_v1_enru_dev300.jsonl \
   gemma_direct=$R/gemma_direct_enru_test.jsonl,$R/gemma_direct_enru_dev.jsonl \
   gemma_gemba=$R/gemma_gemba_enru_test500.jsonl,$R/gemma_gemba_enru_dev300.jsonl --n-boot 1000
py scripts/report.py base_local=$R/local_base_enes_test.jsonl,$R/local_base_enes_dev.jsonl \
   v1_local=$R/local_v1_enes_test.jsonl,$R/local_v1_enes_dev.jsonl \
   gemma_direct=$R/gemma_direct_enes_test.jsonl,$R/gemma_direct_enru_dev.jsonl --n-boot 1000
py scripts/report.py base_local=$R/base_enru_dev300.jsonl base_tinker=$R/tinker_base_enru_dev300.jsonl \
   v1_local=$R/local_v1_enru_dev300.jsonl v1_tinker=$R/tinker_v1_enru_dev300.jsonl \
   gemma_direct=$R/gemma_direct_enru_dev.jsonl gemma_gemba=$R/gemma_gemba_enru_dev300.jsonl \
   --ref data/splits/en-ru.dev.jsonl --n-boot 1000
```

To get a "− v1" row, put `v1_local` first in the `report.py` call.

---

## v2 and ablation: what synthetic training data buys (2026-10-14, evening)

**Verdict: ship v1.** On real, human-labelled test data v2 is no better than v1. Its
PR-AUC is lower on both pairs, and on en-es the drop is significant: −0.071 [−0.113, −0.009].
On the held-out synthetic test sets, v2 is far better than v1 on every error type. On each
whole set it also beats Gemma 4 31B. Those sets come from the same generators as v2's training
data, so they measure in-distribution skill. The human labels do not reward it, and on real
omissions and additions v2 catches no more than v1.

### What changed between v1 and v2

v1 and v2 differ only in the training rows. Everything else is the same: Qwen3.5-4B, LoRA
rank 16 MLP-only, LR 2e-4 with linear decay, batch 64, one epoch, seed 0. The same 200 en-ru
dev rows were scored during training.

| | v1 | v2 |
|---|---|---|
| Data | `data/sft_v1.jsonl` (sha 32736ed384d6092a) | `data/sft_v2.jsonl` (sha 3480c0b9b4093155) |
| Rows | 14,791 human | 19,886: the same 14,791 human rows, 1,109 rule rows (`rules.train`) and 3,986 Gemma rows (`llm.train`) |
| Major share | 22.4% | 37.0%. en-ru 33.8% (21% of its rows synthetic); **en-es 50.7% (47% of its rows synthetic)** |
| Dropped | none | 31 en-ru `neg_insert` rows whose MT contains "?" (`build_sft.py --drop-neg-questions`, on for en-ru by default) |
| Steps, tokens | 231 steps, 1,452,983 tokens | 310 steps, 2,060,301 tokens (46 rows fall into the partial last batch) |
| Tinker run | `479fd56b-…:train:0` | `4087b7c0-b94b-5bbc-87a9-10ac847ad928:train:0`, 16:48:19 → 16:56:56 CEST (8 min 37 s, 3.1 s per step) |
| Training cost | $1.07 + $0.15 dev scoring | **≈ $1.72**: $1.52 training + $0.20 for 8 dev evaluations (267,568 forward tokens). An estimate from the token counts at $0.737/M; the bill had not caught up yet |
| GGUF | `models/secondread-lora.gguf` | `models/secondread-lora-v2.gguf` (36 MB, 192 tensors). Exported from `ckpt/f8637867-63a2-5676-97cb-83ac563a1b44:train:0_external_weights_final/` |

**About the "82 question rows" in STATUS.** 82 counts every `neg_insert` row with "?" in its MT:
both language pairs, train and test (31 + 26 in train, 8 + 17 in test). Only the 31 Russian
train rows were dropped. In the Russian ones, the inserted «не» mostly turns a question into
its polite form, for example «Не могу ли я ещё чем-нибудь помочь?». The meaning stays the
same, so the Major label is noise. The Spanish rows and the test rows were kept.

Dev scoring during training (200 en-ru dev rows, exact Tinker logprobs, AUROC / PR-AUC):

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | end |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.552 / 0.324 | 0.695 / 0.481 | 0.733 / 0.521 | 0.728 / 0.502 | 0.733 / 0.524 | | | 0.739 / 0.528 (231) |
| v2 | 0.552 / 0.324 | 0.678 / 0.438 | 0.683 / 0.456 | 0.682 / 0.477 | 0.716 / 0.482 | 0.733 / 0.510 | 0.720 / 0.491 | 0.722 / 0.495 (310) |

Export smoke test, from LoRA scale 0 to 1 (same three examples as v1): dropped negation
0.882 → 1.000, wrong number 0.847 → 1.000, faithful 0.259 → 0.004. v1 gave 0.826, 0.826 and 0.003.

### How this was scored

- **Local and free:** Q4_K_M base plus the adapter in a llama-server with `-np 4 -cram 0`,
  scored by `probe_local.py --workers 4`.
  - base and v1 used one server on `:9941`, started with `--lora models/secondread-lora.gguf`, at
    `--lora-scale 0` and `1`. Scale 0 reproduces the earlier base-only scores: on 40 en-ru dev rows the
    largest difference from `results/base_enru_dev300.jsonl` is 0.003 (0.0014 for v1 against
    `local_v1_enru_dev300.jsonl`).
  - v2 used its own server on `:9943`. Both servers are stopped.
- **Reused files** (same model, same rows):
  - `local_{base,v1}_{enru_test,enes_test,enes_dev}.jsonl`;
  - `gemma_direct_{enru_test,enru_dev,enes_test}.jsonl`.
- **New files in `results/`:**
  - `local_{base,v1,v2}_enru_dev.jsonl` (the full 2,972 rows);
  - `local_v2_{enru_test,enes_test,enes_dev}.jsonl`;
  - `local_{base,v1,v2}_synth_{rules,llm}_test.jsonl`;
  - `gemma_direct_synth_{rules,llm}_test.jsonl`.

  They hold ids, gold labels and scores only.
- **Thresholds:** each system's F1-optimal threshold comes from its own scores on the **full**
  en-ru dev split (2,972 rows) or on en-es dev. Gemma has no en-es dev run, so its en-es
  threshold comes from en-ru dev, as before.
  - The en-ru thresholds are now base 0.367, v1 0.161 and v2 0.083. On the 300-row sample, v1's was 0.253.
  - en-ru dev has a higher Major rate than test (0.270 vs 0.202), so these thresholds flag a lot of
    en-ru test: v1 flags 42% and v2 50%.

### Main table (real data, human labels)

1,000 document-level bootstrap replicates, shared by all systems in a block, so differences are
paired.

| Pair, split | System | AUROC [95% CI] | PR-AUC [95% CI] | recall@10% [95% CI] | dev thr | F1 at dev thr [95% CI] | flagged |
|---|---|---|---|---|---:|---|---:|
| **en-ru test** (3,573 rows, 33 docs, Major 0.202) | base | 0.666 [0.606, 0.718] | 0.369 [0.256, 0.469] | 0.227 [0.184, 0.288] | 0.367 | 0.381 [0.280, 0.462] | 63% |
| | **v1** | **0.801** [0.743, 0.851] | **0.537** [0.407, 0.635] | 0.303 [0.246, 0.380] | 0.161 | 0.504 [0.419, 0.573] | 42% |
| | v2 | 0.799 [0.752, 0.843] | 0.504 [0.393, 0.595] | 0.290 [0.243, 0.356] | 0.083 | 0.485 [0.406, 0.549] | 50% |
| | Gemma direct | 0.794 [0.754, 0.840] | 0.494 [0.390, 0.588] | **0.307** [0.254, 0.385] | 0.971 | **0.519** [0.428, 0.589] | 33% |
| **en-es test** (1,815 rows, 25 docs, Major 0.093) | base | 0.776 [0.723, 0.826] | 0.413 [0.334, 0.497] | 0.452 [0.358, 0.529] | 0.751 | 0.395 [0.336, 0.452] | 14% |
| | **v1** | **0.815** [0.776, 0.853] | **0.517** [0.442, 0.591] | **0.518** [0.437, 0.602] | 0.702 | 0.429 [0.355, 0.498] | 4% |
| | v2 | 0.804 [0.749, 0.847] | 0.446 [0.372, 0.525] | 0.488 [0.360, 0.570] | 0.572 | **0.464** [0.386, 0.530] | 9% |
| | Gemma direct | 0.807 [0.767, 0.843] | 0.311 [0.282, 0.368] | 0.458 [0.367, 0.558] | 0.971 ¹ | 0.363 [0.315, 0.420] | 27% |
| en-ru dev, full (2,972 rows, 31 docs, Major 0.270) | base | 0.602 [0.549, 0.661] | 0.379 [0.277, 0.469] | 0.188 [0.141, 0.239] | (dev) | | |
| | v1 | **0.739** [0.676, 0.798] | **0.546** [0.398, 0.644] | **0.239** [0.183, 0.314] | | | |
| | v2 | 0.737 [0.693, 0.784] | 0.524 [0.381, 0.624] | 0.237 [0.186, 0.308] | | | |
| | Gemma direct | 0.714 [0.649, 0.771] | 0.436 [0.309, 0.553] | 0.178 [0.135, 0.231] | | | |
| en-es dev (1,650 rows, 12 docs, Major 0.058) | base | 0.801 [0.757, 0.815] | 0.329 [0.264, 0.420] | 0.463 [0.333, 0.523] | (dev) | | |
| | v1 | 0.869 [0.832, 0.886] | **0.453** [0.371, 0.548] | **0.600** [0.369, 0.639] | | | |
| | v2 | **0.876** [0.817, 0.900] | 0.446 [0.383, 0.518] | 0.558 [0.392, 0.619] | | | |

1. From en-ru dev; there is no Gemma en-es dev run.

Paired differences (ΔF1 compares each system at its own dev threshold):

| Block | A − B | ΔAUROC | ΔPR-AUC | Δrecall@10% | ΔF1 |
|---|---|---|---|---|---|
| en-ru test | **v2 − v1** | −0.002 [−0.026, +0.041] | −0.033 [−0.071, +0.017] | −0.012 [−0.048, +0.021] | −0.019 [−0.052, +0.028] |
| en-ru test | **v2 − Gemma** | +0.005 [−0.026, +0.039] | +0.010 [−0.054, +0.066] | −0.017 [−0.062, +0.019] | −0.034 [−0.078, +0.017] |
| en-ru test | v1 − Gemma | +0.006 [−0.054, +0.047] | +0.043 [−0.049, +0.107] | −0.004 [−0.055, +0.041] | −0.015 [−0.076, +0.039] |
| en-ru test | v2 − base | **+0.133** [+0.087, +0.196] | **+0.135** [+0.082, +0.191] | **+0.064** [+0.032, +0.097] | **+0.104** [+0.064, +0.160] |
| en-es test | **v2 − v1** | −0.011 [−0.041, +0.010] | **−0.071** [−0.113, −0.009] | −0.030 [−0.117, +0.027] | +0.035 [−0.030, +0.107] |
| en-es test | **v2 − Gemma** | −0.002 [−0.046, +0.039] | **+0.134** [+0.036, +0.214] | +0.030 [−0.048, +0.077] | +0.102 [−0.014, +0.196] |
| en-es test | v1 − Gemma | +0.008 [−0.016, +0.041] | **+0.206** [+0.099, +0.286] | +0.060 [−0.022, +0.130] | +0.066 [−0.045, +0.165] |
| en-es test | v2 − base | +0.029 [−0.025, +0.073] | +0.033 [−0.024, +0.124] | +0.036 [−0.029, +0.090] | +0.069 [−0.022, +0.136] |
| en-ru dev | v2 − v1 | −0.002 [−0.029, +0.033] | −0.023 [−0.058, +0.012] | −0.002 [−0.027, +0.022] | (dev) |
| en-ru dev | v2 − Gemma | +0.023 [−0.036, +0.086] | **+0.088** [+0.015, +0.153] | **+0.059** [+0.018, +0.110] | (dev) |
| en-es dev | v2 − v1 | +0.007 [−0.028, +0.020] | −0.006 [−0.105, +0.049] | −0.042 [−0.082, +0.108] | (dev) |

**What this shows:**
- **On ranking, v2 ties v1.** All four AUROC differences are within ±0.011 with CIs across 0.
- **On the top of the ranking, v2 is worse.**
  - PR-AUC is lower on every split.
  - On en-es test the drop is significant (−0.071).
  - recall@10%, the review budget, is 0.012 lower on en-ru and 0.030 lower on en-es (CIs include 0).
- **en-es takes the larger hit, and its training mix changed the most.** 47% of v2's en-es rows
  are synthetic, against 21% of en-ru.
  - In the riskiest 10% of en-es test, v2 swaps 54 of 181 segments relative to v1.
  - The 54 that come in hold 9 Major units; the 54 that go out held 14.
- **The F1 gain on en-es (+0.035) comes from the threshold, not the model.** v1's en-es dev
  threshold (0.702) flags only 4% of test, so its recall is 0.298. v2's threshold flags 9%.
  The CI includes 0, and en-es dev has only 95 Major units.
- **Against Gemma 4 31B, v2 keeps v1's position.**
  - en-ru: AUROC and PR-AUC are within noise.
  - en-es: PR-AUC is +0.134 better.
  - On F1, v2 is 0.034 below Gemma on en-ru and 0.102 above it on en-es (both CIs include 0).

### Held-out synthetic test sets

- **Origin:** `data/synth/rules.test.jsonl` (313 rows) and `data/synth/llm.test.jsonl` (763
  rows) are made only from clean units of the test documents. The generators, prompts and
  validators are the same ones that made v2's training rows. **So these sets are in-distribution
  for v2 and out-of-distribution for v1, base and Gemma direct.**
- **Gemma's position:** Gemma 4 itself wrote the LLM edits. Whether that helps or hurts Gemma
  direct when it scores them is unknown.
- **Russian question rows:** the 8 Russian `neg_insert` test rows with "?" are kept and are
  likely partly mislabelled (see above). They are reported separately.
- **Gemma direct cost:** 1,076 calls, $0.024 at list price. Track spend went from $0.348 to $0.372,
  with a cap of $0.498.

**Whole set, synthetic Major rows vs synthetic OK variants** (AUROC, document bootstrap):

| Set, pair | n (OK) | base | v1 | v2 | Gemma direct | v2 − v1 | v2 − Gemma |
|---|---|---|---|---|---|---|---|
| rules, en-ru | 247 (54) | 0.927 [0.875, 0.962] | 0.900 [0.836, 0.954] | **0.997** [0.992, 1.000] | 0.944 [0.877, 0.987] | +0.097 [+0.044, +0.160] | +0.053 [+0.010, +0.120] |
| rules, en-es | 66 (12) | 0.827 [0.727, 0.994] | 0.949 [0.917, 1.000] | **1.000** [1.000, 1.000] | 0.904 [0.842, 0.997] | +0.051 [+0.000, +0.083] | +0.096 [+0.003, +0.158] |
| LLM, en-ru | 392 (76) | 0.766 [0.714, 0.818] | 0.798 [0.754, 0.847] | **0.977** [0.964, 0.991] | 0.927 [0.897, 0.954] | +0.179 [+0.140, +0.213] | +0.051 [+0.026, +0.080] |
| LLM, en-es | 371 (77) | 0.736 [0.684, 0.777] | 0.760 [0.710, 0.831] | **0.992** [0.982, 0.998] | 0.913 [0.877, 0.943] | +0.232 [+0.164, +0.276] | +0.079 [+0.050, +0.115] |

**Per error type.** The first block is recall of the Major rows at each system's dev threshold;
for OK types it is the false-flag rate. The second block is the AUROC of the edited rows
against their own clean parents: the unedited test units they were made from, scored by the
same system. That AUROC does not depend on a threshold, and it compares each edit with the
same sentence before the edit. For OK types (meaning-preserving edits), 0.5 is ideal. "–"
means fewer than 5 rows. The last rows give the false-flag rate on the clean parents, which is
the price paid for the recall in the first block.

en-ru (thresholds: base 0.367, v1 0.161, v2 0.083, Gemma 0.971):

| type | n | recall: base | v1 | v2 | Gemma | AUROC vs parents: base | v1 | v2 | Gemma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| date_day | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 1.00 | 0.83 |
| date_month | 10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| date_year | 18 | 1.00 | 1.00 | 1.00 | 1.00 | 0.89 | 0.87 | 1.00 | 0.86 |
| num_currency | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.84 | 0.88 | 1.00 | 1.00 |
| num_digit | 65 | 1.00 | 0.94 | 1.00 | 0.97 | 0.95 | 0.89 | 1.00 | 0.96 |
| num_magnitude | 51 | 1.00 | 0.88 | 1.00 | 1.00 | 0.88 | 0.82 | 1.00 | 0.95 |
| num_percent | 10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| num_scale | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.96 | 0.96 | 0.96 |
| num_swap | 11 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.96 | 1.00 | 1.00 |
| num_unit | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |
| addition | 88 | 0.78 | 0.59 | **0.97** | 0.57 | 0.58 | 0.61 | **0.95** | 0.71 |
| neg_insert (not a question) | 68 | 0.99 | 0.99 | 1.00 | 0.99 | 0.94 | 0.91 | 1.00 | 0.97 |
| neg_insert, MT is a question | 8 | 0.50 | 0.50 | 0.88 | 0.62 | 0.72 | 0.73 | 0.89 | 0.70 |
| neg_remove | 61 | 0.95 | 0.95 | 0.97 | 0.97 | 0.87 | 0.88 | 0.96 | 0.95 |
| omission | 91 | 0.68 | 0.62 | **0.98** | 0.77 | 0.64 | 0.68 | **0.97** | 0.84 |
| ok_abbrev (OK: false-flag rate) | 8 | 0.62 | 0.25 | 0.50 | 0.38 | 0.52 | 0.56 | 0.58 | 0.55 |
| ok_currency (OK) | 5 | 0.40 | 0.20 | 0.40 | 0.40 | 0.56 | 0.56 | 0.48 | 0.52 |
| ok_grouping (OK) | 1 | 1.00 | 0.00 | 0.00 | 0.00 | – | – | – | – |
| ok_nbsp (OK) | 33 | 0.52 | 0.55 | 0.58 | 0.33 | 0.46 | 0.48 | 0.48 | 0.49 |
| ok_percent (OK) | 3 | 0.67 | 0.33 | 0.33 | 0.33 | – | – | – | – |
| ok_percent_word (OK) | 4 | 0.75 | 0.00 | 0.00 | 0.25 | – | – | – | – |
| ok_paraphrase (OK) | 76 | 0.57 | 0.33 | 0.37 | 0.05 | 0.45 | 0.46 | 0.45 | 0.33 |
| clean parents of the rule rows (false-flag rate) | 194 | 0.58 | 0.41 | 0.44 | 0.31 | | | | |
| clean parents of the LLM rows (false-flag rate) | 373 | 0.55 | 0.39 | 0.47 | 0.26 | | | | |

en-es (thresholds: base 0.751, v1 0.702, v2 0.572; Gemma 0.971 from en-ru dev):

| type | n | recall: base | v1 | v2 | Gemma | AUROC vs parents: base | v1 | v2 | Gemma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| date_year | 3 | 1.00 | 0.67 | 1.00 | 1.00 | – | – | – | – |
| num_digit | 19 | 0.84 | 0.16 | **1.00** | 1.00 | 0.90 | 0.98 | 1.00 | 0.89 |
| num_magnitude | 21 | 0.76 | 0.19 | **1.00** | 0.95 | 0.81 | 0.93 | 1.00 | 0.80 |
| num_scale | 3 | 1.00 | 0.33 | 1.00 | 1.00 | – | – | – | – |
| num_swap | 1 | 1.00 | 1.00 | 1.00 | 1.00 | – | – | – | – |
| num_unit | 7 | 1.00 | 0.71 | 1.00 | 1.00 | 0.92 | 0.92 | 1.00 | 0.94 |
| addition | 80 | 0.16 | 0.04 | **0.96** | 0.76 | 0.63 | 0.66 | **0.99** | 0.79 |
| neg_insert (not a question) | 60 | 0.63 | 0.13 | **1.00** | 1.00 | 0.85 | 0.83 | 1.00 | 0.95 |
| neg_insert, MT is a question | 17 | 0.88 | 0.18 | 0.94 | 1.00 | 0.91 | 0.88 | 1.00 | 0.99 |
| neg_remove (not a question) | 50 | 0.66 | 0.04 | **0.96** | 0.96 | 0.92 | 0.92 | 1.00 | 0.98 |
| neg_remove, MT is a question | 10 | 0.20 | 0.10 | 0.50 | **1.00** | 0.61 | 0.57 | 0.88 | 0.93 |
| omission | 77 | 0.12 | 0.01 | 0.81 | 0.83 | 0.58 | 0.66 | **0.98** | 0.81 |
| ok_abbrev (OK: false-flag rate) | 3 | 0.67 | 0.00 | 0.00 | 0.67 | – | – | – | – |
| ok_grouping (OK) | 1 | 1.00 | 0.00 | 0.00 | 1.00 | – | – | – | – |
| ok_nbsp (OK) | 8 | 0.25 | 0.00 | 0.00 | 0.38 | 0.45 | 0.41 | 0.44 | 0.50 |
| ok_paraphrase (OK) | 77 | 0.08 | 0.00 | 0.01 | 0.23 | 0.47 | 0.48 | 0.46 | 0.43 |
| clean parents of the rule rows (false-flag rate) | 54 | 0.30 | 0.02 | 0.02 | 0.48 | | | | |
| clean parents of the LLM rows (false-flag rate) | 357 | 0.11 | 0.01 | 0.05 | 0.26 | | | | |

**What this shows:**
- **v2 learned the synthetic error types.**
  - On the edited sentence against its own parent, v2 reaches AUROC 0.95–1.00 for every Major
    type with at least 5 rows.
  - The exceptions are the negated questions: 0.89 for the 8 Russian rows with a negation
    inserted, and 0.88 for the 10 Spanish rows with one removed.
  - v1 reaches 0.57–1.00.
  - The largest gains are on **addition** (en-ru 0.61 → 0.95, en-es 0.66 → 0.99) and
    **omission** (0.68 → 0.97, 0.66 → 0.98). Gemma direct is also weak on these two types
    (0.71–0.84).
- **At similar false-flag rates on clean text, v2 recalls far more.**
  - On en-es, v1 and v2 both flag 1–5% of the clean parents.
  - Yet v1 recalls 1–19% of the synthetic additions, omissions, inserted negations and digit or
    magnitude changes, and v2 81–100%.
  - On en-ru, both flag 39–47% of the clean parents. On the Major types other than the
    question rows, v2 recalls 0.97–1.00 against v1's 0.59–1.00.
- **v2 does not just flag rewritten text.**
  - Meaning-preserving Gemma paraphrases score no higher than their parents: AUROC 0.45–0.46,
    about the same as v1 (0.46–0.48).
  - v2's false-flag rate on the paraphrases is at or below its rate on clean parents.
  - So v2 did not learn a "Gemma touched this" shortcut. Gemma direct is the system that
    treats paraphrases as safer than the originals (0.33 on en-ru).
- **Number and date edits were already easy.** Base and v1 rank the edited sentence above its
  parent in 98–100% of pairs. The app's `number_check` rule catches them too. The new skill v2 brings
  that the rules do not cover is addition and omission.
- **The Russian question rows confirm the label noise.**
  - base and v1 flag only half of the 8 «не …?» rows, and Gemma flags 5 of 8.
  - v2 flags 7 of 8. On these rows a high recall partly rewards flagging a polite form.
- **The synthetic addition and omission skill does not carry over to the human-labelled ones.**
  - In en-ru test, 55 Major units carry a major omission span and 54 a major addition span.
  - Of the omission units, v1 puts 25% in its riskiest 10% and v2 24%. Of the addition units,
    both put 26%.
  - In en-es the counts are too small to tell (3 omission and 9 addition units; addition
    4/9 → 6/9).
  - Most real Major units are mistranslations: 76% in en-ru test and 66% in en-es. They contain
    no synthetic type.

### Recommendation: ship v1, keep v2 as a candidate

1. **The ship rule in STATUS fails for v2.** The rule was "keep v1 unless v2 beats it on en-ru
   and en-es test PR-AUC". v2 is lower on both: −0.033 (not significant) and −0.071
   (significant). AUROC and recall@10% are also slightly lower on both test splits. No test
   metric favours v2 except en-es F1, and that gain comes from the threshold.
2. **v2's gains are real but unproven on real MT.**
   - The synthetic test sets share generators with the training data.
   - On real en-ru omissions and additions, which are the types v2 improved most on synthetic
     data, v2 catches no more than v1.
   - Real Major units are mostly mistranslations (66–76%), which the synthetic data does not cover.
   - So v2 gives up some precision at the top of the real ranking, and in return gains recall
     on clean, injected errors.
3. **The number and negation part of v2's gain is already covered at run time.**
   `number_check` and `negation_check` feed the noisy-OR aggregate. What v1 misses and the rules
   cannot see is additions and omissions.

**Next experiments, all free or cheap:**
- **Combine v1 and v2.** As a post-hoc check, the mean of v1 and v2 risks, with no tuning,
  matches v1 on real test:
  - en-ru PR-AUC 0.541 vs 0.537 (Δ +0.004 [−0.011, +0.027]); en-es 0.518 vs 0.517;
  - AUROC 0.806 and 0.816.
  - It keeps most of v2's synthetic gains: AUROC vs parents for addition 0.86 (en-ru) and 0.93
    (en-es), for omission 0.92 and 0.95.

  The idea was tried after seeing test results, and output averaging costs two forward passes.
  The one-pass version is to load both adapters in one llama-server at scale 0.5 each. That
  averages the weights, not the outputs, and needs its own evaluation, starting on dev: about
  1 h of local GPU.
- **v2b on Tinker,** about $1.6:
  - cap the synthetic share per language pair. en-es is now 47% synthetic and its Major share
    went from 25% to 51%;
  - or train on addition, omission and negation rows only, since rules cover numbers and dates.

### Spend of this track

| | Amount | Basis |
|---|---|---|
| Tinker | **≈ $1.72** | 2,060,301 training tokens ($1.52) plus 267,568 dev-scoring tokens ($0.20) at $0.737/M. `fetch_adapter.py` uses no training tokens. At 17:46 CEST the bill had data through the 12:00–13:00 UTC bucket, with $1.661 for everything before v2, so the expected total is about $3.38 of the $18. Storage: 7 new checkpoints (state + sampler; the 6 periodic ones expire after 7 days) and one external-weights export |
| DigitalOcean | **$0.024** | 1,076 Gemma direct calls on the synthetic test sets. Baseline track $0.348 → $0.372; all DO use about $0.79 of $5 |
| Local | $0 | about 57 min of GPU: base and v1 8,096 rows, v2 11,086 rows |

### Reproduce

```sh
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
set -a && . ./.env && set +a
# data: drops the 31 en-ru neg_insert questions by default (--drop-neg-questions none keeps them)
py scripts/build_sft.py --synthetic data/synth/rules.train.jsonl data/synth/llm.train.jsonl --out data/sft_v2.jsonl
py scripts/train_tinker.py --data data/sft_v2.jsonl --log-path runs/v2 --lr 2e-4 --save-every 50 \
   --dev-split data/splits/en-ru.dev.jsonl --dev-n 200 --eval-every 50
py scripts/fetch_adapter.py tinker://4087b7c0-b94b-5bbc-87a9-10ac847ad928:train:0/weights/final --out ckpt
scripts/export_adapter.sh --port 9942 "ckpt/f8637867-63a2-5676-97cb-83ac563a1b44:train:0_external_weights_final" \
   models/secondread-lora-v2.gguf
# local scoring: one server per adapter (base = the v1 server at --lora-scale 0)
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora-v2.gguf -c 16384 -np 4 -cram 0 \
   --host 127.0.0.1 --port 9943 &
for s in data/splits/en-ru.dev data/splits/en-es.dev data/splits/en-ru.test data/splits/en-es.test \
         data/synth/rules.test data/synth/llm.test; do
  t=$(basename $s | sed 's/\.test$/_test/; s/\.dev$/_dev/; s/^en-ru/enru/; s/^en-es/enes/; s/^rules/synth_rules/; s/^llm/synth_llm/')
  py scripts/probe_local.py $s.jsonl 100000 --workers 4 --lora-scale 1 --port 9943 --out results/local_v2_$t.jsonl
done
# Gemma direct on the synthetic sets (cached; --cap is for the whole baseline track)
py scripts/baseline_gemma.py --method direct data/synth/rules.test.jsonl --out results/gemma_direct_synth_rules_test.jsonl --cap 0.498
py scripts/baseline_gemma.py --method direct data/synth/llm.test.jsonl --out results/gemma_direct_synth_llm_test.jsonl --cap 0.498
# main tables
R=results
py scripts/report.py base=$R/local_base_enru_test.jsonl,$R/local_base_enru_dev.jsonl \
   v1=$R/local_v1_enru_test.jsonl,$R/local_v1_enru_dev.jsonl v2=$R/local_v2_enru_test.jsonl,$R/local_v2_enru_dev.jsonl \
   gemma_direct=$R/gemma_direct_enru_test.jsonl,$R/gemma_direct_enru_dev.jsonl \
   --n-boot 1000 --diff v2:v1 --diff v2:gemma_direct --diff v1:gemma_direct --diff v2:base
py scripts/report.py base=$R/local_base_enes_test.jsonl,$R/local_base_enes_dev.jsonl \
   v1=$R/local_v1_enes_test.jsonl,$R/local_v1_enes_dev.jsonl v2=$R/local_v2_enes_test.jsonl,$R/local_v2_enes_dev.jsonl \
   gemma_direct=$R/gemma_direct_enes_test.jsonl,$R/gemma_direct_enru_dev.jsonl \
   --n-boot 1000 --diff v2:v1 --diff v2:gemma_direct --diff v1:gemma_direct --diff v2:base
# per error type: SET = rules|llm, LP = en-ru (lp=enru) or en-es (lp=enes); Gemma's dev file is always en-ru
py scripts/report.py \
   base=$R/local_base_synth_${SET}_test.jsonl,$R/local_base_${lp}_dev.jsonl,$R/local_base_${lp}_test.jsonl \
   v1=$R/local_v1_synth_${SET}_test.jsonl,$R/local_v1_${lp}_dev.jsonl,$R/local_v1_${lp}_test.jsonl \
   v2=$R/local_v2_synth_${SET}_test.jsonl,$R/local_v2_${lp}_dev.jsonl,$R/local_v2_${lp}_test.jsonl \
   gemma_direct=$R/gemma_direct_synth_${SET}_test.jsonl,$R/gemma_direct_enru_dev.jsonl,$R/gemma_direct_${lp}_test.jsonl \
   --ref data/synth/$SET.test.jsonl --lp $LP --by synth_type --n-boot 1000 --diff v2:v1 --diff v2:gemma_direct
```

The question split in the per-type tables comes from a scratch copy of `llm.test.jsonl` that
has an extra field. In it, `neg_insert`/`neg_remove` rows whose MT contains "?" get the suffix
`_question`. That copy is passed as `--ref … --by synth_type_q`.

**`report.py` changes made in this pass:**
- a third file per system, `NAME=TEST,DEV,PARENTS`, for the AUROC-vs-parents table;
- `--lp`, to restrict to one language pair;
- `--diff A:B`, for extra paired differences including ΔF1;
- in `--by` tables, false-flag rates for OK-only groups and for the clean parents.

The main table and the "vs first system" differences are unchanged. The en-es block of the
v1 tables above reproduces exactly.

---

## Ship decision and app calibration (2026-10-14, 18:10 CEST)

**v1 ships.** `:9931` now serves Qwen3.5-4B Q4_K_M with `--lora models/secondread-lora.gguf`
(LoRA v1) and `-cram 0`.

### Check of the v2 numbers

The Ship phase re-ran the `report.py` commands above (main tables for en-ru and en-es, and
`--by synth_type` for both synthetic sets and both pairs). Every number in "v2 and ablation"
came out identical: same files, same seed, same 1,000 replicates.

### Why v1 and not v2

The rule for this phase: ship v2 if it is better on real held-out data, or equal there and much
better on the synthetic error types.

- **On real data v2 is not equal.**
  - On en-es test its PR-AUC is significantly lower: −0.071 [−0.113, −0.009].
  - On en-ru test every top-of-ranking estimate is lower, though within noise: PR-AUC −0.033,
    recall@10% −0.012, F1 −0.019. On en-ru dev, PR-AUC is −0.023.
- **v2 for en-ru only was considered and rejected.**
  - Its en-ru numbers are no better than v1's.
  - Its synthetic gains on omissions and additions do not show up on the real ones (24% vs
    25% and 26% vs 26% in the riskiest 10%).
  - One adapter keeps the app and its thresholds simple.

### What is installed

| File | sha256 (first 12) | What |
|---|---|---|
| `models/secondread-lora.gguf` | `5337b2b7b17a` | **v1, served on `:9931`** (the 14:03 export, unchanged) |
| `models/secondread-lora-v1.gguf` | `5337b2b7b17a` | a copy of v1 under an explicit name |
| `models/secondread-lora-v2.gguf` | `9dda3f68f273` | v2, kept |

- **Server:** `llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9931`,
  started with `nohup` from the repo root, log in `/tmp/claude-501/llama-9931.log`.
  - `/health` returns `ok`.
  - `/lora-adapters` lists `models/secondread-lora.gguf` at scale 1.0.
  - The flags (`-c 16384 -np 4`, 4,096 tokens per slot) are the same as before.
- **The served adapter is v1.** `probe_local.py` scored 40 en-ru dev rows with no `lora` field in
  the request, so the server's own scale was used. Against `results/local_v1_enru_dev.jsonl` the
  largest risk difference is 0.0012. Against v2 it is 0.305, and against base 0.608.

### Flag thresholds per adapter and language pair

> **Since the review pass (2026-10-14, night) these are the `f1` operating point.** The backend now
> flags at the `review` point by default (dev recall of Major ≥ 0.60); see "Review operating
> point" below. `SECONDREAD_OPERATING_POINT=f1` restores these values.

The backend flags a segment when its **fused** risk reaches the threshold of the active adapter
for the segment's language pair. At ship time the fused risk was the noisy-OR of the model risk
and the two rule checks (replaced since; see "Rule calibration"). `scripts/pick_thresholds.py` picks each threshold this way:
1. it runs `number_check` and `negation_check` on every unit of the **full** dev split;
2. it fuses them with that adapter's local dev scores, using `secondread.agent.aggregate`;
3. it takes the F1-optimal threshold (`metrics.best_f1_threshold`) and rounds it to 4 decimals.

The result is written to `secondread/thresholds.json` (committed; ids and numbers only). The test
columns below are for information only and played no part in the choice.

| Adapter | Pair | Threshold | dev n | dev F1 | dev flagged | test precision | test recall | test F1 | test flagged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | en-ru | 0.3668 | 2,972 | 0.445 | 66% | 0.250 | 0.781 | 0.378 | 63% |
| base | en-es | 0.7506 | 1,650 | 0.361 | 7% | 0.318 | 0.518 | 0.394 | 15% |
| **v1** | **en-ru** | **0.1609** | 2,972 | 0.536 | 37% | 0.367 | 0.769 | 0.497 | 42% |
| **v1** | **en-es** | **0.7019** | 1,650 | 0.442 | 4% | 0.712 | 0.309 | 0.431 | 4% |
| v2 | en-ru | 0.0827 | 2,972 | 0.534 | 46% | 0.337 | 0.840 | 0.481 | 50% |
| v2 | en-es | 0.5717 | 1,650 | 0.421 | 8% | 0.419 | 0.464 | 0.441 | 10% |

- **Fusion leaves the thresholds unchanged.** Each one equals the model-only F1 threshold, because
  rule hits are rare on dev (0.5% of en-ru rows, 2.0% of en-es).
- **The en-ru line is generous.** It flags about 4 segments in 10, a consequence of F1 on a split
  with 27% Major. The product's main output is the riskiest-10% view, which does not depend on the
  threshold.
- **The en-es line is strict.** It flags 4%, with precision 0.71. It comes from only 95 Major units
  in 12 dev documents.
- **Order of precedence in the backend:** a `threshold` in the request, then
  `SECONDREAD_THRESHOLD`, then this file, then 0.5.
  - The 0.5 fallback also applies to an adapter file whose sha256 is not in the file, to a
    `lora_scale` other than 0 or 1, and to rule-only reviews when llama-server is down.
  - `lora_scale` 0 uses the base thresholds.
- **`/api/health`** reports `adapter: "secondread-lora v1"`, `adapter_version: "v1"`, the
  thresholds and their source.
  - The version comes from the sha256 of the file llama-server loaded, not from its name.
  - Review summaries carry `thresholds`, `threshold_source` and `adapter_version`.
  - The model reason reads "Model (LoRA v1): P(Major) …".

### The app's fused score vs the model alone

> **Superseded (2026-10-14, later that evening):** the rule weights are now measured on the train
> split, and the app's score no longer uses the hand-set noisy-OR described here. See
> "Rule calibration" below. This subsection is kept as the record of why.

Every table above scores the model alone. The app ranks by the fused score. Paired, with 1,000
document replicates, on the same rows (`runs/ship/fused_*`, git-ignored):

| Split | v1 model only: AUROC / PR-AUC / recall@10% | v1 in the app (fused) | Δ AUROC | Δ PR-AUC | Δ recall@10% |
|---|---|---|---|---|---|
| en-ru test | 0.801 / 0.537 / 0.303 | 0.790 / 0.512 / 0.275 | **−0.010** [−0.020, −0.003] | **−0.026** [−0.048, −0.009] | **−0.028** [−0.063, −0.001] |
| en-es test | 0.815 / 0.517 / 0.518 | 0.804 / 0.486 / 0.500 | **−0.011** [−0.019, −0.001] | −0.031 [−0.056, +0.003] | −0.018 [−0.076, +0.019] |
| en-ru dev | 0.739 / 0.546 / 0.239 | 0.738 / 0.545 / 0.240 | −0.001 [−0.002, +0.001] | −0.001 [−0.004, +0.002] | +0.001 [−0.004, +0.008] |
| en-es dev | 0.869 / 0.453 / 0.600 | 0.876 / 0.431 / 0.547 | +0.007 [−0.015, +0.019] | −0.022 [−0.075, +0.010] | −0.053 [−0.090, +0.051] |

**The rule checks do not improve the ranking on human labels, and on test they lower it a
little.** They still supply the reasons and highlighted spans the UI shows.

- **Cause: the hand-set rule confidences (0.35–0.75) are far above how often a rule hit is a
  real Major.**

  | Split | Rows with a rule hit | Distinct sources | Major among hits | Major rate of the split |
  |---|---:|---:|---:|---:|
  | en-ru train | 267 | 85 | 31% | 22% |
  | en-ru dev | 16 | 6 | 50% | 27% |
  | en-ru test | 66 | 18 | 6% | 20% |
  | en-es train | 221 | 73 | 19% | 9% |
  | en-es dev | 33 | 18 | 18% | 6% |
  | en-es test | 71 | 16 | 8% | 9% |

- **The false alarms on test come from a handful of source sentences, each repeated across up to
  16 MT systems.** Found while diagnosing, not to be tuned on:
  - "£25" vs «25 фунтов» read as a unit change (15 rows);
  - "whether or not" read as a negation (12 rows);
  - "season 2" vs «во втором сезоне», an ordinal word (12 rows);
  - "24/7" vs «круглосуточно» (10 rows);
  - "no doubt" vs «несомненно» (5 rows).
- **The test-free fix:** set each rule's confidence from its Major rate on the **train** split. The
  rules are not fit to that split, and it has 267 + 221 hits. Then check on dev. Because these false
  alarms were found on test, a later test run of fixes aimed at them would be test-informed. Report
  it that way.

### End-to-end checks through the backend

The backend ran on `:8787` temporarily with `.env` not loaded, so Sentry was off. It was stopped
afterwards.

- **Demo job,** `demo/secondread-demo.en-ru.xliff20.xlf`: 51 segments, 14 of them gold Major.
  - v1 gets AUROC 0.905, PR-AUC 0.724 and recall@10% 0.286. The base model got 0.728 and 0.570 in
    the morning pass.
  - At 0.1609 it flags 23 segments: precision 0.57, recall 0.93.
  - The whole job takes 9.1 s (178.6 ms per segment, model call p50 660 ms).
  - These segments were picked from test documents for having known errors, so they are not a
    benchmark.
- **`bench_review.py data/splits/en-ru.dev.jsonl 100`:** 163 ms per segment, model call p50
  652 ms and p95 745 ms. On these 100 rows, fused AUROC 0.794 equals the model-only one.
- **`ui/public/demo-results.json`,** rebuilt with `make_demo_results.py --api`:
  - `placeholder` is now false, with the note "fine-tuned LoRA v1 active";
  - 7 of 10 segments flagged at 0.1609, which include 3 of the 4 gold Major ones.

---

## Rule calibration (2026-10-14, night)

**Verdict: the rule checks now cost the ranking almost nothing, and they still do not improve
it.** The app's score now adds each finding's weight, measured on the train split, to the model's
log-odds. The hand-set noisy-OR is gone.
- **Against the hand-set version (test):** en-ru AUROC +0.009 [+0.003, +0.016] and PR-AUC +0.022
  [+0.007, +0.043]; en-es AUROC +0.009 and PR-AUC +0.028 (both CIs include 0).
- **Against the model alone (test):** −0.002 AUROC on both pairs. On en-ru that is just
  significant ([−0.004, −0.000]). recall@10% is unchanged.
- **Reasons:** every finding is still returned as a reason and highlighted, however small its
  weight.

### What was measured on train

`scripts/calibrate_rules.py` runs `number_check` and `negation_check` on every unit of
`data/splits/{en-ru,en-es}.train.jsonl`, exactly as the server does. The rules were never fit to
any split, so train is a fair place to measure them. Dev and test played no part in any number
below.

- **Clustering:** one source sentence is translated by up to 16 MT systems, and a finding usually
  fires on several of them. In the fit, each hit of a code is weighted by (distinct source
  sentences of the code) / (hits of the code). The code keeps its row-level Major rate but counts
  as only as many observations as it has distinct sources.
- **Weight:** a weighted logistic regression per pair, intercept plus one 0/1 feature per code,
  with a N(0, 1) prior on each code's coefficient (Newton's method, deterministic). The weight is
  the coefficient, clamped at 0. It is the log odds ratio of Major with vs without the finding,
  i.e. its likelihood ratio, shrunk towards "no evidence" when a code has few sources.
- **P(Major | finding):** the smoothed probability, sigmoid(intercept + coefficient).
- **CI:** a document-level bootstrap on train (1,000 replicates) of the raw rate.

P(Major | no finding): en-ru 0.218, en-es 0.083 (Major rates of the splits: 0.219 and 0.087).

| Pair | Code | Hits | Major | Sources | Raw P(Major \| hit) [95% CI] | Smoothed | Coefficient | **Weight shipped** | Coefficient given v1 ¹ | Hand-set (old) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| en-ru | `negation_added` | 136 | 23 | 35 | 0.17 [0.04, 0.35] | 0.18 | −0.26 | **0.00** | −0.14 | 0.35 |
| | `negation_dropped` | 22 | 7 | 14 | 0.32 [0.07, 0.57] | 0.29 | +0.39 | **0.39** | −0.31 | 0.75 |
| | `negation_fewer` | 78 | 38 | 12 | 0.49 [0.16, 0.76] | 0.41 | +0.92 | **0.92** | +0.17 | 0.40 |
| | `number_changed` | 4 | 2 | 3 | 0.50 [0.00, 1.00] | 0.32 | +0.53 | **0.53** | +0.32 | 0.75 |
| | `number_extra` | 4 | 3 | 4 | 0.75 [0.00, 1.00] | 0.47 | +1.14 | **1.14** | +0.48 | 0.25 |
| | `number_missing` | 23 | 9 | 18 | 0.39 [0.18, 0.64] | 0.35 | +0.68 | **0.68** | +0.48 | 0.55 |
| en-es | `negation_added` | 90 | 11 | 23 | 0.12 [0.07, 0.20] | 0.11 | +0.29 | **0.29** | +0.21 | 0.35 |
| | `negation_dropped` | 29 | 8 | 12 | 0.28 [0.14, 0.71] | 0.19 | +0.96 | **0.96** | +0.55 | 0.75 |
| | `negation_fewer` | 23 | 12 | 10 | 0.52 [0.37, 0.78] | 0.34 | +1.75 | **1.75** | +1.28 | 0.40 |
| | `number_changed` | 29 | 3 | 12 | 0.10 [0.00, 0.17] | 0.09 | +0.12 | **0.12** | −0.10 | 0.75 |
| | `number_extra` | 29 | 4 | 8 | 0.14 [0.04, 0.38] | 0.11 | +0.25 | **0.25** | +0.23 | 0.25 |
| | `number_missing` | 25 | 4 | 14 | 0.16 [0.04, 0.36] | 0.11 | +0.27 | **0.27** | −0.08 | 0.55 |
| | `unit_changed` | 1 | 1 | 1 | 1.00 | 0.17 | +0.83 | **0.83** | +0.51 | 0.70 |

1. Fitted with logit(v1 risk on the same train rows) as an offset (candidate `logodds_offset`
   below). Not shipped.

- **Hit rates:** a finding is Major in 10–52% of hits, against hand-set confidences of 0.25–0.75.
- **Numbers:** the number codes are weak in en-es, where `number_changed` is Major only 10% of
  the time. In en-ru they have very few hits (4 for `number_changed`).
- **Negation:** `negation_fewer` is the strongest code in both pairs. `negation_added` is
  evidence *against* Major in en-ru (17% vs 22%), so its weight is clamped to 0.
- **Read with care:** the weight is the odds multiplier e^w. For example, en-es `negation_fewer`
  multiplies the model's odds by 5.7, and en-ru `negation_dropped` by 1.5.
- **Other pairs:** a "*" entry, the same fit on both pairs pooled, serves any pair without its
  own entry (for example en-de).

**v1's train scores are memorised, so they are a poor base for a fit.** v1 was trained on these
rows for one epoch:
- en-ru train AUROC is 0.971, against 0.739 on dev and 0.801 on test.
- en-es train AUROC is 0.906, against 0.869 on dev. It is lower there because 3,821 of the
  5,354 en-es OK rows were not in `sft_v1` (en-es OK was subsampled 3:1). Those rows have mean
  risk 0.171, against 0.158 for the OK rows trained on.

A fit that conditions on these scores sees a model that nearly knows the labels. It therefore
under-weights the rules: compare the "given v1" column with the shipped one. Scoring train cost
nothing: about 70 min of local GPU on `:9950` with `probe_local.py --workers 4`. The output is
`results/local_v1_{enru,enes}_train.jsonl`, with ids, labels and scores only.

### Candidates, and the choice on dev

All candidates are fitted on train only.

| Candidate | Combined risk | Needs model scores on train |
|---|---|---|
| `logodds_rules` (**shipped**) | logit(risk) = logit(model) + Σ over checks of the strongest finding's weight (table above) | no |
| `noisy_or_train` | the old noisy-OR, each confidence set to the smoothed train P(Major \| finding) | no |
| `logodds_offset` | as `logodds_rules`, with weights fitted using logit(v1) as an offset | yes |
| `logreg` | sigmoid(a + b·logit(v1) + Σ weights), all fitted per pair | yes |
| reference: `model` | v1 alone | |
| reference: `handset` | the noisy-OR with the hand-set 0.25–0.75 the app used until now | |

**The selection rule was fixed before the dev run:**
1. Measure ΔAUROC, ΔPR-AUC and Δrecall@10% against `model` on the full en-ru dev split (2,972
   rows) and on en-es dev (1,650 rows): 6 deltas in all. Use a paired document bootstrap with
   1,000 replicates.
2. A candidate is admissible when no delta has a 95% CI entirely below 0.
3. Pick the admissible candidate with the highest mean of the 6 point deltas.
4. Prefer the simpler candidate when it is within 0.002 of the best. The order from simplest is
   `logodds_rules`, `noisy_or_train`, `logodds_offset`, `logreg`.
5. With no admissible candidate, ship weight 0, so the rules only give reasons.

Dev results, paired against v1 alone:

| Candidate | en-ru ΔAUROC | en-ru ΔPR-AUC | en-ru Δrecall@10% | en-es ΔAUROC | en-es ΔPR-AUC | en-es Δrecall@10% | mean of 6 |
|---|---|---|---|---|---|---|---:|
| handset (reference) | −0.0008 [−0.0021, +0.0005] | −0.0014 [−0.0045, +0.0023] | +0.0012 [−0.0041, +0.0084] | +0.0069 [−0.0145, +0.0190] | −0.0221 [−0.0746, +0.0099] | −0.0526 [−0.0899, +0.0513] | −0.0115 |
| **logodds_rules** | −0.0000 [−0.0001, +0.0000] | −0.0001 [−0.0005, +0.0003] | +0.0000 [−0.0023, +0.0025] | +0.0032 [−0.0034, +0.0069] | −0.0027 [−0.0168, +0.0021] | −0.0105 [−0.0367, +0.0282] | −0.0017 |
| noisy_or_train | −0.0005 [−0.0012, +0.0003] | −0.0002 [−0.0012, +0.0012] | +0.0012 [−0.0038, +0.0047] | +0.0059 [−0.0037, +0.0118] | +0.0015 [−0.0053, +0.0052] | −0.0105 [−0.0299, +0.0357] | −0.0004 |
| logodds_offset | 0 ² | 0 ² | 0 ² | +0.0019 [−0.0019, +0.0046] | −0.0019 [−0.0128, +0.0009] | −0.0211 [−0.0257, +0.0186] | −0.0035 |
| logreg | 0 ² | 0 ² | 0 ² | +0.0016 [−0.0005, +0.0038] | −0.0004 [−0.0034, +0.0009] | −0.0211 [−0.0241, +0.0179] | −0.0033 |

2. No en-ru dev row moves: the only codes that fire on en-ru dev (`negation_added` 13 rows,
   `negation_dropped` 3) get weight 0 in this fit.

**Choice: `logodds_rules`.**
- All four candidates are admissible.
- `noisy_or_train` has the best mean (−0.0004). `logodds_rules` is 0.0013 behind it, within the
  0.002 margin, and is simpler. The log-odds form also does not count the base rate twice, as a
  noisy-OR of P(Major | finding) does.
- **Dev cannot separate the candidates.** It has only 16 en-ru and 33 en-es rows with a finding.
  One en-es Major unit is worth 0.0105 of recall@10%.
- The choice rests on the pre-set rule and on the train evidence, not on dev precision. Its
  weights are model-free, so they apply unchanged to base and v2 as well.

### Test, run once after the choice

Same `report.py` settings: 1,000 document replicates shared by all rows of a block, and F1 at
each system's own F1-optimal dev threshold.

| Pair, split | System | AUROC [95% CI] | PR-AUC [95% CI] | recall@10% [95% CI] | dev thr | F1 at dev thr [95% CI] |
|---|---|---|---|---|---:|---|
| **en-ru test** (3,573 rows, 33 docs, Major 0.202) | v1 alone | 0.801 [0.743, 0.851] | 0.537 [0.407, 0.635] | 0.303 [0.246, 0.380] | 0.161 | 0.504 [0.419, 0.573] |
| | app, hand-set noisy-OR (old) | 0.790 [0.735, 0.839] | 0.512 [0.376, 0.615] | 0.275 [0.225, 0.353] | 0.161 | 0.498 [0.413, 0.569] |
| | **app, calibrated (new)** | 0.799 [0.742, 0.848] | 0.533 [0.404, 0.633] | 0.303 [0.246, 0.379] | 0.161 | 0.502 [0.415, 0.572] |
| **en-es test** (1,815 rows, 25 docs, Major 0.093) | v1 alone | 0.815 [0.776, 0.853] | 0.517 [0.442, 0.591] | 0.518 [0.437, 0.602] | 0.702 | 0.429 [0.355, 0.498] |
| | app, hand-set noisy-OR (old) | 0.804 [0.767, 0.846] | 0.486 [0.427, 0.560] | 0.500 [0.392, 0.590] | 0.702 | 0.432 [0.340, 0.508] |
| | **app, calibrated (new)** | 0.813 [0.775, 0.851] | 0.514 [0.439, 0.590] | 0.524 [0.425, 0.596] | 0.702 | 0.436 [0.355, 0.508] |

| Block | A − B | ΔAUROC | ΔPR-AUC | Δrecall@10% | ΔF1 |
|---|---|---|---|---|---|
| en-ru test | **calibrated − v1 alone** | −0.002 [−0.004, −0.000] | −0.004 [−0.008, −0.000] | +0.000 [−0.006, +0.004] | −0.002 [−0.006, +0.000] |
| en-ru test | **calibrated − hand-set** | **+0.009** [+0.003, +0.016] | **+0.022** [+0.007, +0.043] | **+0.028** [+0.002, +0.058] | +0.004 [+0.000, +0.012] |
| en-ru test | hand-set − v1 alone | **−0.010** [−0.020, −0.003] | **−0.026** [−0.048, −0.009] | **−0.028** [−0.063, −0.001] | |
| en-es test | **calibrated − v1 alone** | −0.002 [−0.004, +0.001] | −0.003 [−0.013, +0.006] | +0.006 [−0.030, +0.016] | +0.007 [+0.000, +0.017] |
| en-es test | **calibrated − hand-set** | +0.009 [−0.000, +0.017] | +0.028 [−0.002, +0.048] | +0.024 [−0.018, +0.070] | +0.004 [−0.005, +0.028] |
| en-es test | hand-set − v1 alone | **−0.011** [−0.019, −0.001] | −0.031 [−0.056, +0.003] | −0.018 [−0.076, +0.019] | |

**What this shows:**
- **The fix removes most of the damage.**
  - On en-ru test the old fused score lost 0.010 AUROC, 0.026 PR-AUC and 0.028 recall@10%
    against the model alone. The calibrated score loses 0.002, 0.004 and 0.000.
  - en-es is the same: −0.011 / −0.031 / −0.018 before, −0.002 / −0.003 / +0.006 now.
- **It does not make the rules useful for ranking.** The calibrated score is still a little
  below the model alone, and on en-ru that is just significant. The train evidence, a rule hit
  being Major in 10–50% of hits, is real. But v1 already sees most of the same evidence: it gave
  the hit rows a higher mean risk than the rest (0.32 vs 0.24 on en-ru train). And test findings
  are rarely Major: 4 of 66 en-ru hits, 6 of 71 en-es hits.
- **Why the old score hurt, in one number.**
  - The riskiest 10% of en-ru test is 357 rows. The hand-set noisy-OR put 53 of the 66 hit rows
    there, and only 4 of them were Major. The model alone put 4 there, the calibrated score 7,
    with all 4 Major hits among those 7.
  - On en-es test the counts were 43, 6 and 13 hit rows in the riskiest 182.
- **The threshold does not move.** The v1 thresholds stay at en-ru 0.1609 and en-es 0.7019.
  - en-es test F1 rises from 0.429 to 0.436 (+0.007 [+0.000, +0.017]).
  - en-ru F1 falls 0.002, within noise.

**The other candidates on the same test run, recorded for completeness; they do not change the
choice:**

| Candidate | en-ru ΔAUROC vs v1 alone | en-ru ΔPR-AUC | en-es ΔAUROC | en-es ΔPR-AUC |
|---|---|---|---|---|
| noisy_or_train | −0.005 [−0.010, −0.002] | −0.007 [−0.015, −0.002] | −0.004 [−0.006, +0.001] | −0.003 [−0.009, +0.004] |
| logodds_offset | −0.001 [−0.003, −0.000] | −0.002 [−0.004, −0.000] | −0.001 [−0.003, +0.001] | −0.002 [−0.007, +0.004] |
| logreg | −0.000 [−0.001, +0.000] | −0.001 [−0.002, +0.000] | −0.001 [−0.002, +0.001] | −0.001 [−0.005, +0.004] |

- The fits conditioned on v1's memorised train scores come out closest to the model alone,
  because they give the rules the smallest weights.
- No candidate beats the model alone on test.
- Switching to one of them now would be a choice made on test. If the rules should not move the
  ranking at all, weight 0 is the honest version of that.

### In the app

- **Score:** `secondread/rulecal.py` computes logit(risk) = logit(model risk) + Σ over checks of
  the strongest finding's weight (`secondread/rule_calibration.json`).
  - A segment with no finding keeps exactly the model risk. A finding never lowers it.
- **Reasons:** every finding is returned as a reason with its spans, whatever its weight.
  - `confidence` is now the train P(Major | finding) for the pair. The new `weight` field is the
    log-odds it adds. The UI ignores both and shows chips, notes and highlights as before (`ui/API.md`).
  - `signals.<check>.risk` is that P(Major | finding), no longer the hand-set prior.
- **Rule checks only (llama-server down):** the risk is the largest train P(Major | finding) of
  the segment, at most 0.47.
  - This is a behaviour change. At the 0.5 fallback threshold, nothing is flagged in this mode
    any more. The findings are still listed and sort first. Before, a single `number_changed`
    (0.75) was flagged.
- **`/api/health`** has a new `rule_calibration` field: method, fit, date and pairs.
  - `secondread/thresholds.json` records the sha256 of the calibration file that its thresholds
    were picked with.
- **Thresholds** (`scripts/pick_thresholds.py`, re-run on the new fused score; F1-optimal on
  full dev):
  - v1 unchanged: en-ru 0.1609 (dev F1 0.537, 37% flagged) and en-es 0.7019 (dev F1 0.449, 3%
    flagged). For information, on test: en-ru P 0.372, R 0.769, F1 0.502; en-es P 0.773, R
    0.304, F1 0.436.
  - base en-es moves from 0.7506 to 0.7345. The other base and v2 thresholds are unchanged.
- **Demo** (`ui/public/demo-results.json`, rebuilt through the backend):
  - Segment 10, a dropped negated clause and gold Major, goes from 0.896 to 0.675 (model 0.586
    plus `negation_dropped`, weight 0.39). It is still flagged and still shows its negation
    reason and highlight.
  - The other 9 segments are unchanged, and 7 of 10 are flagged as before.
  - The 51-segment demo job has no rule findings, so its scores are unchanged: 23 flagged,
    9.1 s.
- **Checks:** `tests/test_xliff.py` 20/20. `scripts/canary_check.py` PASS: the canary is in 0 of
  12 envelopes, and in 4 with the scrubbers off.

### Reproduce

```sh
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
# v1 scores of the train rows (only the two candidates that condition on the model need them)
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf -c 16384 -np 4 -cram 0 \
  --host 127.0.0.1 --port 9950 &
py scripts/probe_local.py data/splits/en-es.train.jsonl 100000 --workers 4 --lora-scale 1 --port 9950 --out results/local_v1_enes_train.jsonl
py scripts/probe_local.py data/splits/en-ru.train.jsonl 100000 --workers 4 --lora-scale 1 --port 9950 --out results/local_v1_enru_train.jsonl
kill %1
# fit on train, write secondread/rule_calibration.json and every candidate's combined dev scores
py scripts/calibrate_rules.py --candidates-dir runs/rulecal --splits dev
R=runs/rulecal; for t in enru enes; do
  py scripts/report.py model=$R/model_${t}_dev.jsonl handset=$R/handset_${t}_dev.jsonl \
     logodds_rules=$R/logodds_rules_${t}_dev.jsonl noisy_or_train=$R/noisy_or_train_${t}_dev.jsonl \
     logodds_offset=$R/logodds_offset_${t}_dev.jsonl logreg=$R/logreg_${t}_dev.jsonl --n-boot 1000; done
# after the choice: the test run (the calibration file comes out byte-identical)
py scripts/calibrate_rules.py --candidates-dir runs/rulecal --splits dev test
for t in enru enes; do
  py scripts/report.py model=$R/model_${t}_test.jsonl,$R/model_${t}_dev.jsonl \
     handset=$R/handset_${t}_test.jsonl,$R/handset_${t}_dev.jsonl \
     logodds_rules=$R/logodds_rules_${t}_test.jsonl,$R/logodds_rules_${t}_dev.jsonl \
     noisy_or_train=$R/noisy_or_train_${t}_test.jsonl,$R/noisy_or_train_${t}_dev.jsonl \
     logodds_offset=$R/logodds_offset_${t}_test.jsonl,$R/logodds_offset_${t}_dev.jsonl \
     logreg=$R/logreg_${t}_test.jsonl,$R/logreg_${t}_dev.jsonl \
     --n-boot 1000 --diff logodds_rules:model --diff logodds_rules:handset; done
py scripts/pick_thresholds.py --fused-dir runs/rulecal/app   # app fusion == runs/rulecal/logodds_rules_* (max diff 0.0)
```

`runs/rulecal/` is git-ignored. Without the train score files, `calibrate_rules.py` still fits
and writes the shipped `logodds_rules` weights; it skips the two candidates that need them.

---

## Review operating point (2026-10-14, night)

**Verdict: the app now flags at a recall-oriented "review" line by default.** For v1 it finds 60%
of the dev Majors in both pairs. On en-es that costs 3× the flags of the F1 line (10% of segments
instead of 3%) at half the precision (0.34 instead of 0.63). On en-ru the F1 line already found 64%,
so the review line sits slightly *above* it and flags a little less (35% instead of 37%). Only dev
was used. No test split and no real document was scored for this section.

**Why:** at the F1 line, en-es is precise and quiet, and its recall is low: 0.35 on dev and 0.30 on
test. REALDOCS.md shows the same on real documents (2 of 9 verified majors). A post-editor reading a
flag list wants more of the majors in it, and can afford more flags.

### The rule, fixed before any number was computed

For each language pair, on the **full** dev split of that pair, using the app's combined score
(model log-odds plus the train-calibrated rule weights, `secondread.agent.aggregate`, the same
fusion `scripts/pick_thresholds.py` uses for the F1 line):
1. choose the **highest** threshold whose dev recall of Major is ≥ 0.60
   (`metrics.max_threshold_at_recall`), stored rounded *down* to 4 decimals, so the stored value
   still has recall ≥ 0.60. For v1 the stored value flags exactly the same rows as the exact one
   on both pairs; for base en-ru it adds one OK row;
2. report dev precision and flag rate at it, with a 95% document-bootstrap CI for recall
   (1,000 replicates, seed 0 for en-ru and seed 1 for en-es, recall at the fixed threshold);
3. if a pair's recall CI is wider than ±0.15 (read as hi − lo > 0.30), also report the threshold
   the same rule gives on en-ru + en-es dev pooled, but still use the per-pair one.

Model and threshold selection use dev only. The review mode of `pick_thresholds.py` opens no test
file (`tests/test_thresholds.py` checks this).

### v1, dev

| Pair | Point | Threshold | dev n (docs, Major) | Flagged | Recall [95% CI] | Precision [95% CI] | Flag rate [95% CI] | F1 |
|---|---|---:|---|---:|---|---|---|---:|
| en-ru | **review** | **0.1857** | 2,972 (31, 803) | 1,027 | **0.600** [0.466, 0.705] | 0.469 [0.330, 0.575] | 34.6% [26.7%, 42.2%] | 0.527 |
| | f1 (before) | 0.1609 | | 1,102 | 0.638 | 0.465 | 37.1% | 0.538 |
| en-es | **review** | **0.4984** | 1,650 (12, 95) | 167 | **0.600** [0.500, 0.673] | 0.341 [0.312, 0.433] | 10.1% [6.8%, 18.4%] | 0.435 |
| | f1 (before) | 0.7019 | | 52 | 0.347 | 0.635 | 3.1% | 0.449 |

- **Rows:** en-ru flags 1,027 rows at the review line, 482 of the 803 Majors. en-es flags 167,
  57 of the 95. 0.60 × 95 = 57 exactly, and the next distinct score up gives 56 (0.589).
- **Recall CIs:** en-ru ±0.119 and en-es ±0.087, both within ±0.15. So the pooled threshold was
  not required and is not reported. The asymmetric reading (largest distance from the point
  estimate) gives 0.134 and 0.100, also within ±0.15.
- **The CI is for recall at the fixed threshold.** It does not include the uncertainty of the
  threshold itself, which is chosen on the same rows. Re-picking the threshold on each bootstrap
  replicate gives en-ru [0.096, 0.314] and en-es [0.426, 0.533]. Expect the recall on new data
  to move more than the recall CI suggests, especially on en-ru, where many rows sit at low risk.
- **Rule weights barely matter here.** The model-only threshold under the same rule is the same
  for en-ru (0.1857) and 0.4960 for en-es.
- **The F1 counts** are from the F1 block of `secondread/thresholds.json`, computed when that line
  was picked. They have no CIs.

### Other adapters, same rule (dev, for the record)

| Adapter | Pair | Review threshold | Recall [95% CI] | Precision | Flag rate | F1 threshold | F1-line recall / precision / flag rate |
|---|---|---:|---|---:|---:|---:|---|
| base | en-ru | 0.4431 | 0.600 [0.501, 0.703] | 0.317 | 51.1% | 0.3668 | 0.766 / 0.314 / 66.0% |
| base | en-es | 0.6183 | 0.600 [0.509, 0.735] | 0.213 | 16.2% | 0.7345 | 0.421 / 0.313 / 7.8% |
| v2 | en-ru | 0.1403 | 0.600 [0.489, 0.704] | 0.441 | 36.7% | 0.0827 | 0.724 / 0.424 / 46.1% |
| v2 | en-es | 0.4704 | 0.600 [0.511, 0.679] | 0.310 | 11.2% | 0.5717 | 0.516 / 0.366 / 8.1% |

All recall CIs are within ±0.15, so the pooled report was not triggered for any adapter. These
values make the base model (`lora_scale` 0) and v2 work at the review point too.

### Not measured yet

> **Measured since:** the test splits and the real documents at the review line are in "v3 — test
> and real documents" below. The static demo was rebuilt at the review line in the Ship phase
> (same section, "Ship").

- **Test.** The review line has not been applied to either test split. Under the integrity rules
  each final candidate's test split is scored once and reported whatever the outcome. That run
  belongs to the Ship phase, together with the v3 candidate.
- **Real documents.** REALDOCS.md, `data/realdocs/scored.jsonl` and the en-es demo answer key are
  all at the F1 line (0.7019). They are evaluation only and were not looked at for this choice.
- **The static demo.** `ui/public/demo-results.json` was built through the backend at the F1
  line. It needs `make_demo_results.py --api` and `npm run build:demo` again before it shows the
  review line.

### In the app

- **`secondread/thresholds.json`:** each adapter has a `review` block and an `f1` block, each with
  `thresholds` and `stats`. The review stats are dev only: counts, recall, precision and flag rate
  with CIs, the re-picked threshold CI and the model-only threshold. The top level holds
  `default_operating_point: "review"` and each point's method, date and rule-calibration sha256.
  The older one-point format still loads, as the `f1` point.
- **Choosing the point:** a request's `operating_point` (`review` or `f1`, in a JSON body,
  multipart field or query), then `SECONDREAD_OPERATING_POINT`, then the file's default, then
  `review`. A `threshold` in the request or `SECONDREAD_THRESHOLD` still overrides any point.
  - An unknown point is refused: a 400 `bad_operating_point` for a request, a start-up error for
    the variable.
  - A point with no value for the adapter or the pair falls back to 0.5, and the source says so
    (`dev_review+fallback`).
- **`/api/health`:**
  - `operating_point` and `operating_point_source` (`file` here);
  - `thresholds` and `threshold_source` (`dev_review`) at that point;
  - `operating_points`, the thresholds each point would apply to the loaded adapter.

  Review summaries carry `operating_point` as well.
- **Checked through a backend on `:9970`** against the served v1 on `:9931`:
  - **Health:** `review` from the file: en-es 0.4984, en-ru 0.1857.
  - **30 en-es and 30 en-ru dev rows, at both points:** every risk is within 0.0017 of the
    fused dev file. The flag counts match that file at each threshold: en-ru 14 at review and 16 at
    F1, en-es 13 and 1.
  - **Switching:** `?operating_point=f1` and the multipart field both switch the point, and
    `"recall"` returns a 400.
  - **The variable:** `SECONDREAD_OPERATING_POINT=f1` gives `operating_point_source: env` and the
    F1 thresholds. An unknown value stops the start-up.
- **Other checks:** `tests/test_thresholds.py` 7/7, `tests/test_xliff.py` 20/20,
  `scripts/canary_check.py` PASS (the canary is in 0 envelopes, and in 4 with the scrubbers off).

### Reproduce

```sh
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
py scripts/pick_thresholds.py --target-recall 0.60   # review blocks of base, v1, v2; the f1 blocks are kept
py tests/test_thresholds.py
```

---

## v3 (wrong-sense synthetic) — dev selection (2026-10-14, night)

**Verdict: keep v1.** Neither candidate passes the selection rule that was fixed before their dev
scores existed. On the full en-ru dev split both lose about 0.035 PR-AUC against v1, more than
the 0.01 margin. v3a also loses 0.0105 AUROC on en-ru and 0.020 PR-AUC on en-es. Both candidates
are clearly better than v1 on the held-out synthetic sense set. That set comes from the same
generator as their training rows, and the gain does not show up on human labels: on dev, the
real mistranslations are not ranked any higher. This is the v2 pattern again.

**What was scored:** the full en-ru and en-es dev splits for v3a and v3b, and the synthetic sense
test set for v1, v3a and v3b. v1's dev scores were reused. No test split and no real document was
scored. The one exception is the 380 clean parents of the sense test rows. They are gold-OK units
from test documents, and the "AUROC vs clean parents" column needs their scores. No Major unit
of a test split was scored, and no human-label test metric was computed. The selection uses dev
only.

### The candidates

Everything except the training rows is as in v1: Qwen3.5-4B, LoRA rank 16 MLP-only, LR 2e-4
with linear decay, batch 64, 1 epoch, seed 0. Both runs score the same 200 en-ru dev rows during
training.

| | v1 (shipped) | v3a | v3b |
|---|---|---|---|
| Data | `data/sft_v1.jsonl`, 14,791 human rows | `data/sft_v3a.jsonl` (sha `9945949ac643b8cd`), 16,258 rows: v1's 14,791 human rows plus 1,467 wrong-sense rows (`data/synth/sense.v3a.train.jsonl`, `scripts/subsample_synth.py`) | `data/sft_v3b.jsonl` (sha `2b677b743e808c21`), 17,830 rows: the same human rows plus 1,613 wrong-sense rows and 1,426 v2 synthetic rows (`data/synth/mix_v3b/`, `scripts/mix_synth.py`) |
| Synthetic share | 0% | en-ru 8.5%, en-es 12.0% (cap 12% per pair; Major:OK about 3:1) | wrong-sense: en-ru 8.6%, en-es 12.0% (cap 12%). v2 types: 8.0% in both (cap 8%). All synthetic: en-ru 16.6%, en-es 20.0% |
| Steps, tokens | 231, 1,452,983 | 254, 1,607,700 | 278, 1,768,395 |
| Tinker run | `479fd56b-…:train:0` | `abc2567b-8635-5c1b-9e87-20f536e95a06:train:0` | `7df264ae-f959-55df-aa64-e32f372a36e7:train:0` |
| Cost (estimated from token counts) | $1.07 + $0.15 | ≈ $1.36 | ≈ $1.48 |
| GGUF (sha256) | `models/secondread-lora.gguf` (`5337b2b7…`) | `models/secondread-lora-v3a.gguf` (`eac1fb1a…`) | `models/secondread-lora-v3b.gguf` (`64f78f05…`) |
| 200 en-ru dev rows at the end of training (AUROC / PR-AUC) | 0.739 / 0.528 | 0.728 / 0.482 | 0.733 / 0.506 |

The wrong-sense rows come from `scripts/synth_sense.py` (Gemma 4 31B on train documents): the
types `wrong_sense`, `false_friend` and `antonym_term` are Major, and `synonym_ok` is OK.
`scripts/verify_synth.py sense` reports 0 problems on the train file, including the checks for
real-document stories and integrity words.

### The selection rule, fixed before any v3a or v3b dev score was computed

1. On each pair's full dev split, take the **model-only** AUROC and PR-AUC (point estimates) of
   v1, v3a and v3b.
2. A candidate is **eligible** only if, on **both** pairs, its PR-AUC ≥ v1's − 0.01 **and** its
   AUROC ≥ v1's − 0.01.
3. Among the eligible candidates, pick the one with the highest mean of (ΔPR-AUC en-ru,
   ΔPR-AUC en-es) against v1.
4. If no candidate is eligible, or the best mean ΔPR-AUC is not above 0, keep v1.

The rule names the PR-AUC as model-only and does not say which AUROC. `scripts/select_v3.py`
uses model-only for both and also applies the rule to the app's combined score. Both readings
give the same decision.

### Dev results, full splits

1,000 document-bootstrap replicates (seed 0), shared by the three systems of a pair, so the
differences are paired. "Model" is the adapter alone. "App" is the score the backend ranks by:
the model log-odds plus the train-calibrated rule weights (`secondread.agent.aggregate`).

| Pair (rows, docs, Major) | System | Model AUROC [95% CI] | Model PR-AUC [95% CI] | Model recall@10% [95% CI] | App AUROC / PR-AUC / recall@10% |
|---|---|---|---|---|---|
| **en-ru dev** (2,972, 31, 803) | **v1** | **0.739** [0.676, 0.798] | **0.546** [0.398, 0.644] | **0.239** [0.183, 0.314] | 0.739 / 0.546 / 0.239 |
| | v3a | 0.728 [0.670, 0.785] | 0.510 [0.369, 0.614] | 0.235 [0.182, 0.293] | 0.728 / 0.510 / 0.235 |
| | v3b | 0.739 [0.690, 0.793] | 0.511 [0.365, 0.613] | 0.228 [0.171, 0.295] | 0.739 / 0.511 / 0.228 |
| **en-es dev** (1,650, 12, 95) | **v1** | 0.869 [0.832, 0.886] | **0.453** [0.371, 0.548] | **0.600** [0.369, 0.639] | 0.872 / 0.450 / 0.589 |
| | v3a | 0.869 [0.824, 0.886] | 0.433 [0.335, 0.574] | 0.558 [0.408, 0.607] | 0.871 / 0.434 / 0.568 |
| | v3b | 0.866 [0.804, 0.890] | 0.452 [0.362, 0.561] | 0.579 [0.400, 0.625] | 0.866 / 0.450 / 0.568 |

| Pair | Candidate − v1 | ΔAUROC (model) | ΔPR-AUC (model) | Δrecall@10% (model) | ΔAUROC / ΔPR-AUC (app) |
|---|---|---|---|---|---|
| en-ru dev | v3a − v1 | −0.011 [−0.031, +0.016] | **−0.036** [−0.067, −0.005] | −0.004 [−0.037, +0.013] | −0.011 / −0.036 |
| en-ru dev | v3b − v1 | +0.000 [−0.031, +0.044] | −0.035 [−0.074, +0.011] | −0.011 [−0.046, +0.010] | +0.000 / −0.035 |
| en-es dev | v3a − v1 | −0.000 [−0.023, +0.008] | −0.020 [−0.053, +0.049] | −0.042 [−0.084, +0.133] | −0.001 / −0.016 |
| en-es dev | v3b − v1 | −0.003 [−0.036, +0.007] | −0.001 [−0.045, +0.040] | −0.021 [−0.089, +0.112] | −0.006 / +0.000 |

On en-ru the rule checks barely fire on dev, so the app's score equals the model's to three
decimals.

### Applying the rule

| Candidate | ΔPR-AUC en-ru | ΔPR-AUC en-es | ΔAUROC en-ru | ΔAUROC en-es | Eligible | Mean ΔPR-AUC |
|---|---:|---:|---:|---:|---|---:|
| v3a | **−0.0357** | **−0.0196** | **−0.0105** | −0.0001 | no: fails 3 of 4 conditions | −0.0277 |
| v3b | **−0.0353** | −0.0007 | +0.0004 | −0.0028 | no: fails en-ru PR-AUC | −0.0180 |
| app-score reading | v3a −0.0357, v3b −0.0352 | v3a −0.0163, v3b +0.0003 | v3a −0.0105, v3b +0.0004 | v3a −0.0013, v3b −0.0059 | no, no | −0.0260, −0.0174 |

Bold: a condition that fails (below −0.01). No candidate is eligible, and neither mean ΔPR-AUC is
above 0, so **v1 is kept**.

- **en-ru PR-AUC decides it.** It is the larger dev split (803 Majors), and the drop is the same
  for both candidates. For v3a the CI excludes 0; for v3b it just includes 0.
- **v3b ties v1 on ranking.** Its AUROC is equal on en-ru and its en-es PR-AUC is within 0.001.
  It loses at the top of the en-ru ranking, which is what PR-AUC measures.
- **The 200-row curve during training pointed the same way:** v3a 0.728 / 0.482 and v3b 0.733 /
  0.506, against v1's 0.739 / 0.528.

### Review thresholds (threshold rule, dev only)

The rule from "Review operating point": the highest threshold whose dev recall of Major is
≥ 0.60 on the app's score, rounded down to 4 decimals. Recall, precision and flag rate are at
that fixed threshold, with document-bootstrap CIs (1,000 replicates, seed 0 for en-ru and 1 for
en-es). v1's values reproduce `secondread/thresholds.json` exactly.

| System | Pair | Review threshold | Recall [95% CI] | Precision [95% CI] | Flag rate [95% CI] | Flagged / Majors caught | Threshold re-picked per replicate |
|---|---|---:|---|---|---|---|---|
| **v1** | en-ru | **0.1857** | 0.600 [0.466, 0.705] | **0.469** [0.330, 0.575] | **34.6%** [26.7%, 42.2%] | 1,027 / 482 of 803 | [0.096, 0.314] |
| v3a | en-ru | 0.1763 | 0.600 [0.473, 0.719] | 0.440 [0.305, 0.551] | 36.8% [30.7%, 43.1%] | 1,095 / 482 | [0.104, 0.346] |
| v3b | en-ru | 0.3205 | 0.600 [0.483, 0.716] | 0.462 [0.337, 0.572] | 35.1% [28.6%, 42.0%] | 1,043 / 482 | [0.175, 0.470] |
| **v1** | en-es | **0.4984** | 0.600 [0.500, 0.673] | **0.341** [0.312, 0.433] | **10.1%** [6.8%, 18.4%] | 167 / 57 of 95 | [0.425, 0.533] |
| v3a | en-es | 0.5941 | 0.600 [0.508, 0.659] | 0.297 [0.239, 0.403] | 11.6% [9.7%, 18.1%] | 192 / 57 | [0.548, 0.681] |
| v3b | en-es | 0.6457 | 0.600 [0.470, 0.657] | 0.315 [0.252, 0.493] | 11.0% [8.5%, 18.6%] | 181 / 57 | [0.499, 0.674] |

- **Same recall, more flags.** At the review line, both candidates need more flags than v1 for the
  same 60% of dev Majors: en-ru 1,095 and 1,043 against 1,027, and en-es 192 and 181 against 167.
- **Pooled threshold:** not triggered. The widest recall CI is ±0.123 (v3a en-ru).
- **The thresholds differ, the rule does not.** v3b's en-ru line is 0.3205 against v1's 0.1857
  because v3b's risks are higher overall. The number on its own means nothing.
- **In the app:** `scripts/pick_thresholds.py --target-recall 0.60 --adapters v3a v3b` wrote these
  as `review` blocks for v3a and v3b in `secondread/thresholds.json`. The backend now recognises
  both GGUFs by sha256. Base, v1, v2 and the operating-point metadata are unchanged, and the
  default stays `review`. The candidates have no `f1` block, because the F1 mode also reads test
  for information. If one were served at `f1`, it would get the 0.5 fallback.

### Real mistranslations on dev (descriptive)

The point of v3 was wrong-sense errors, which are most of the real Majors. The dev Majors are
split by whether they carry a major or critical mistranslation (or `wrong_term`) span. For each
split: the share in the riskiest 10% of all dev rows, and the share flagged at each system's own
review line (app score).

| Pair | System | Mistranslation Majors in top 10% | at the review line | Other Majors in top 10% | at the review line |
|---|---|---:|---:|---:|---:|
| en-ru (696 + 107) | v1 | **0.254** | 0.598 | 0.140 | **0.617** |
| | v3a | 0.246 | 0.601 | 0.168 | 0.598 |
| | v3b | 0.236 | **0.608** | **0.178** | 0.551 |
| en-es (73 + 22) | v1 | 0.630 (46) | 0.630 (46) | **0.455** (10) | **0.500** (11) |
| | v3a | 0.630 (46) | **0.671** (49) | 0.364 (8) | 0.364 (8) |
| | v3b | 0.630 (46) | **0.671** (49) | 0.364 (8) | 0.364 (8) |

- **The real mistranslations do not move up the ranking.** In the riskiest 10%, en-ru goes from
  0.254 to 0.246 and 0.236, and en-es stays at 46 of 73 for all three systems.
- **At the review line, en-es swaps majors rather than gaining them.** Recall is fixed at 57 of
  95, so the 3 extra mistranslations (46 → 49) are paid for with 3 of the 22 other Majors
  (11 → 8). en-es dev has 12 documents, so this is about 3 units.

### Held-out synthetic sense set (descriptive)

- **The set:** `data/synth/sense.test.jsonl`, 420 rows. The same generator, prompts and
  validators as the v3 training rows made it, from clean units of the **test** documents. It is
  in-distribution for v3a and v3b and out-of-distribution for v1. No train-split sentence is in
  it.
- **Columns:**
  - **flagged:** the share flagged at each system's own review threshold. For the Major types
    that is recall; for `synonym_ok`, a meaning-preserving edit, it is the false-flag rate.
  - **AUROC vs parents:** the edited rows against their own clean parents. The parents are the
    unedited units the edits were made from (380 distinct, all gold OK), scored by the same
    system. For `synonym_ok`, 0.5 is ideal.
  - **All three:** on the app's score. The model-only AUROCs agree to within 0.01.
- **CIs:** 1,000 document-bootstrap replicates, for the pooled Major row only.
- **Check:** v1's re-scored parents match its earlier test-split scores to within 0.0013.

en-ru (review thresholds: v1 0.1857, v3a 0.1763, v3b 0.3205):

| Type | n | flagged: v1 | v3a | v3b | AUROC vs parents: v1 | v3a | v3b |
|---|---:|---:|---:|---:|---:|---:|---:|
| antonym_term | 66 | 0.94 | 0.97 | **1.00** | 0.91 | 0.96 | **0.98** |
| false_friend | 31 | 0.74 | **0.84** | **0.84** | 0.84 | **0.88** | 0.86 |
| wrong_sense | 65 | 0.91 | **0.95** | **0.95** | 0.88 | **0.92** | **0.92** |
| **all Major types** | 162 | 0.89 | 0.94 | **0.95** | 0.87 [0.83, 0.92] | **0.93** [0.89, 0.96] | 0.92 [0.89, 0.95] |
| synonym_ok (OK: false-flag rate) | 64 | 0.44 | 0.41 | 0.38 | 0.50 | 0.52 | 0.51 |
| clean parents (false-flag rate) | 207 | 0.42 | 0.43 | 0.36 | | | |

en-es (review thresholds: v1 0.4984, v3a 0.5941, v3b 0.6457):

| Type | n | flagged: v1 | v3a | v3b | AUROC vs parents: v1 | v3a | v3b |
|---|---:|---:|---:|---:|---:|---:|---:|
| antonym_term | 58 | 0.71 | 0.91 | **0.93** | 0.92 | 0.95 | **0.97** |
| false_friend | 22 | 0.27 | 0.50 | **0.59** | 0.88 | **0.90** | **0.90** |
| wrong_sense | 60 | 0.52 | **0.77** | 0.75 | 0.85 | 0.90 | **0.91** |
| **all Major types** | 140 | 0.56 | 0.79 | **0.80** | 0.87 [0.81, 0.91] | 0.91 [0.87, 0.94] | **0.92** [0.87, 0.95] |
| synonym_ok (OK: false-flag rate) | 54 | 0.09 | 0.13 | 0.15 | 0.52 | 0.52 | 0.51 |
| clean parents (false-flag rate) | 173 | 0.08 | 0.12 | 0.11 | | | |

- **v3 learned the synthetic types.** At the review line, en-es recall on the sense errors rises
  from 0.56 to 0.79–0.80, and false friends from 0.27 to 0.50–0.59. Against their parents, AUROC
  rises from 0.87 to 0.91–0.93 on both pairs.
- **It did not learn an "edited text" shortcut.** Synonym edits stay at 0.51–0.52 against their
  parents.
- **The cost on clean text is small but real.** On en-es the clean parents are flagged at
  0.11–0.12 instead of 0.08. On dev the same shows up as 181–192 flags instead of 167.
- **The gain is in-distribution.** The training rows and these test rows come from one generator.
  On human-labelled dev the candidates are lower (the tables above), so the synthetic gain does
  not carry over to the real wrong-sense errors in WMT MQM. v2 showed the same for omissions and
  additions.

### What this means for the Ship phase

- **v1 is the only final candidate.** Its test splits are scored once at its review line
  (0.1857 / 0.4984) and reported whatever the outcome (integrity rule 3).
- **v3a and v3b are not final candidates,** so their test splits are not scored. Their GGUFs and
  review thresholds stay on disk and in `secondread/thresholds.json`, as v2's do.
- **Real documents:** v3 was not scored on them. A later look would be evaluation after the
  decision. It could not change the decision and would have to be reported as post-selection.

### Spend of this phase

$0. Local GPU only, 22:54–23:26 CEST: llama-servers on `:9970` (v3a), `:9971` (v3b) and `:9972`
(v1), all with `-cram 0 -np 4` and all stopped. 11,644 rows were scored: 2 × 4,622 dev, 3 × 420
sense and 3 × 380 parents. `:9931` (v1, PID 22074) was not touched.

### Files

- **Scores** (`results/`; ids, labels and scores only):
  - `local_{v3a,v3b}_{enru,enes}_dev.jsonl`;
  - `local_{v1,v3a,v3b}_synth_sense_test.jsonl`;
  - `local_{v1,v3a,v3b}_synth_sense_parents.jsonl`.
- **Parents file:** `data/synth/sense.test.parents.jsonl`, the 380 clean parents (git-ignored).
- **Analysis:** `scripts/select_v3.py`. It holds the rule, reads no test split, and writes every
  number above to `runs/v3_select/select.json` (git-ignored).
- **Threshold script:** `scripts/pick_thresholds.py`, with `v3a` and `v3b` added to `SYSTEMS`.
- **Thresholds file:** `secondread/thresholds.json`, with new `review` blocks for v3a and v3b.

### Reproduce

```sh
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
# one server per adapter, own port, -cram 0; stop it afterwards
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora-v3a.gguf -c 16384 -np 4 -cram 0 \
   --host 127.0.0.1 --port 9970 &
for s in data/splits/en-ru.dev:enru_dev data/splits/en-es.dev:enes_dev \
         data/synth/sense.test:synth_sense_test data/synth/sense.test.parents:synth_sense_parents; do
  py scripts/probe_local.py ${s%%:*}.jsonl 100000 --workers 4 --lora-scale 1 --port 9970 --out results/local_v3a_${s#*:}.jsonl
done
# (same for v3b on another port; v1 needs only the two sense files, its dev scores exist)
# sense.test.parents.jsonl = the units of data/splits/{en-ru,en-es}.test.jsonl whose id is a synth_parent
# of data/synth/sense.test.jsonl (380 rows, all OK)
py scripts/select_v3.py                                         # dev selection + descriptive tables
py scripts/pick_thresholds.py --target-recall 0.60 --adapters v3a v3b   # review blocks; others kept
py tests/test_thresholds.py
```

---

## v3 — test and real documents (2026-10-14, night)

**Verdict: v1 stays.** The v3 dev selection kept v1, so v1 is the only final candidate. Its test
splits were evaluated once, at both operating points. v3a and v3b were scored on neither the test
splits nor the real documents.

- **en-ru test, review line (0.1857):** v1 finds 74% of the Majors, against 60% on dev. It flags
  38% of the segments, at precision 0.39. That is a little stricter than the F1 line (0.1609):
  2.5 points less recall for 3.4 points fewer flags.
- **en-es test, review line (0.4984): it misses the target set on dev.** It finds 48% of the
  Majors [0.42, 0.58], below the 0.60 it was set for on dev. It flags 8.2% of the segments, at
  precision 0.54.
  - Against the F1 line (0.7019): +18 points of recall [+11, +26] for +4.6 points of flags.
    Precision falls from 0.77 to 0.54. F1 rises from 0.436 to 0.511, though the CI of that
    difference includes 0.
- **Against Gemma 4 31B direct:** the same picture as the main table.
  - Ranking is tied on en-ru. On en-es, v1 is ahead on PR-AUC by +0.203.
  - At the review lines on en-es, v1 flags 8% of the segments and Gemma 32%. v1's recall is
    0.27 lower and its precision 0.33 higher.
- **Real documents, review line (first pass, superseded by the gold set in `docs/REALDOCS_GOLD.md`,
  section 8):** v1 catches 3 of the 9 verified majors in the three public OPUS-MT documents (2 at
  the F1 line). It raises 3 false flags, all on minor errors (1 at the F1 line), and flags 2.9%
  of the segments.
  - **Official Spanish (control):** 0 flags on the same three pages. Over all five documents it
    raises 2 flags in 348 segments, both on correct translations. The F1 line raised none.
  - **Wrong-sense misses:** tax "return", "flush" → "lavar" and "paperless" → "impresa" stay well
    below both lines (0.05–0.28).

### What was evaluated, and what was reused

- **No test row was re-scored.**
  - **v1:** `results/local_v1_{enru,enes}_test.jsonl`, from the v1 GGUF (sha256 `5337b2b7…`, the
    file `:9931` serves). The v3 phase re-scored 380 gold-OK test units with v1 for the sense
    set, and they match these files within 0.0013.
  - **Gemma:** `results/gemma_direct_{enru,enes}_test.jsonl`.
  - **Check:** `scripts/final_test.py` confirms that each file has exactly the ids and gold labels
    of the current split.
  - **App score:** computed from these files with the shipped rule calibration (`fbc71fcd…`), as
    the backend does.
- **First test run of the review line.** This is the only time it has met test. The F1-line test
  numbers were already reported (see "Rule calibration", "Test, run once after the choice") and
  reproduce exactly here.
- **Real documents: re-run end to end through the backend at its default (review) line.**
  - Command: `scripts/realdocs_score.py --no-gemma`, on its own llama-server with v1 on `:9973`
    (`-cram 0`, 23:33–23:38 CEST, then stopped).
  - Result: 1,104 segments, 57 flagged. Every risk equals the 19:58 snapshot
    `data/realdocs/scored.jsonl` (largest difference 0.0).
  - Gemma's real-document scores come from that snapshot. No Gemma call was made.

### Test, main table

1,000 document-bootstrap replicates (seed 0) over the rows sorted by id, the same replicates as
`scripts/report.py`. So the model-only rows reproduce the main table at the top of this file. The
replicates are shared within a pair, so every difference is paired.

**System names:**
- "v1, app" is the score the backend ranks and flags by: model log-odds plus the calibrated
  rule weights.
- "v1, model" is the adapter alone.

| Pair, split | System | AUROC [95% CI] | PR-AUC [95% CI] | recall@10% [95% CI] |
|---|---|---|---|---|
| **en-ru test** (3,573 rows, 33 docs, 723 Major) | **v1, app** | 0.799 [0.742, 0.848] | 0.533 [0.404, 0.633] | 0.303 [0.246, 0.379] |
| | v1, model | 0.801 [0.743, 0.851] | 0.537 [0.407, 0.635] | 0.303 [0.246, 0.380] |
| | Gemma direct | 0.794 [0.754, 0.840] | 0.494 [0.390, 0.588] | 0.307 [0.254, 0.385] |
| **en-es test** (1,815 rows, 25 docs, 168 Major) | **v1, app** | 0.813 [0.775, 0.851] | 0.514 [0.439, 0.590] | 0.524 [0.425, 0.596] |
| | v1, model | 0.815 [0.776, 0.853] | 0.517 [0.442, 0.591] | 0.518 [0.437, 0.602] |
| | Gemma direct | 0.807 [0.767, 0.843] | 0.311 [0.282, 0.368] | 0.458 [0.367, 0.558] |

At each system's operating points. Every threshold was fixed on dev before this run.

| Pair | System | Line | Threshold | Flagged (Major) | Recall [95% CI] | Precision [95% CI] | Flag rate [95% CI] | F1 [95% CI] |
|---|---|---|---:|---:|---|---|---|---|
| en-ru | **v1, app** | **review** | 0.1857 | 1,373 (538) | **0.744** [0.626, 0.835] | 0.392 [0.307, 0.466] | 38.4% [27.6%, 48.1%] | 0.513 [0.427, 0.580] |
| | v1, app | f1 | 0.1609 | 1,494 (556) | 0.769 [0.643, 0.862] | 0.372 [0.293, 0.443] | 41.8% [30.6%, 51.7%] | 0.502 [0.415, 0.572] |
| | Gemma direct | review ¹ | 0.1212 | 1,328 (523) | 0.723 [0.605, 0.821] | 0.394 [0.301, 0.485] | 37.2% [29.7%, 44.3%] | 0.510 [0.417, 0.582] |
| | Gemma direct | f1 ¹ | 0.9714 | 1,163 (489) | 0.676 [0.565, 0.776] | 0.420 [0.323, 0.513] | 32.5% [25.8%, 39.3%] | 0.519 [0.428, 0.589] |
| en-es | **v1, app** | **review** | 0.4984 | 149 (81) | **0.482** [0.421, 0.581] | 0.544 [0.404, 0.664] | 8.2% [6.0%, 12.4%] | 0.511 [0.437, 0.563] |
| | v1, app | f1 | 0.7019 | 66 (51) | 0.304 [0.240, 0.375] | 0.773 [0.611, 0.907] | 3.6% [2.7%, 5.0%] | 0.436 [0.355, 0.508] |
| | Gemma direct | review ¹ | 0.1212 | 580 (126) | 0.750 [0.653, 0.836] | 0.217 [0.174, 0.278] | 32.0% [27.6%, 37.6%] | 0.337 [0.284, 0.402] |
| | Gemma direct | f1 ¹ | 0.9714 | 488 (119) | 0.708 [0.621, 0.796] | 0.244 [0.200, 0.307] | 26.9% [22.7%, 32.0%] | 0.363 [0.315, 0.420] |

1. **Where Gemma's lines come from:** its en-ru dev run (2,972 rows), by the same two rules,
   with no rule fusion.
   - The F1 line is the 0.971 of the main table.
   - The review line is the highest threshold with dev recall ≥ 0.60 (`recall_threshold`, as in
     `pick_thresholds.py`). The review rule was written for the adapters, so applying it to
     Gemma is this pass's choice. It uses dev only.
   - Gemma has no en-es dev run, so the en-ru dev values serve en-es too.
   - Gemma's risks are saturated, so its two lines flag similar sets.

Paired differences (95% CI excluding 0 in bold). At the lines, each system uses its own threshold.

| Pair | A − B | ΔAUROC | ΔPR-AUC | Δrecall@10% |
|---|---|---|---|---|
| en-ru | v1 app − Gemma direct | +0.004 [−0.055, +0.046] | +0.039 [−0.052, +0.104] | −0.004 [−0.056, +0.040] |
| en-ru | v1 app − v1 model | **−0.002** [−0.004, −0.000] | **−0.004** [−0.008, −0.000] | +0.000 [−0.006, +0.004] |
| en-es | v1 app − Gemma direct | +0.006 [−0.017, +0.040] | **+0.203** [+0.099, +0.280] | +0.065 [−0.017, +0.124] |
| en-es | v1 app − v1 model | −0.002 [−0.004, +0.001] | −0.003 [−0.013, +0.006] | +0.006 [−0.030, +0.016] |

| Pair | A − B | Line | ΔRecall | ΔPrecision | ΔFlag rate | ΔF1 |
|---|---|---|---|---|---|---|
| en-ru | v1 app − Gemma direct | review | +0.021 [−0.135, +0.145] | −0.002 [−0.046, +0.047] | +1.3 pt [−6.3, +8.5] | +0.003 [−0.054, +0.057] |
| en-ru | v1 app − Gemma direct | f1 | +0.093 [−0.052, +0.207] | **−0.048** [−0.092, −0.001] | **+9.3 pt** [+1.9, +16.2] | −0.017 [−0.077, +0.036] |
| en-ru | v1 app, review − f1 | | **−0.025** [−0.042, −0.013] | **+0.020** [+0.008, +0.034] | **−3.4 pt** [−4.7, −2.2] | **+0.012** [+0.001, +0.024] |
| en-es | v1 app − Gemma direct | review | **−0.268** [−0.357, −0.153] | **+0.326** [+0.194, +0.454] | **−23.7 pt** [−29.4, −17.0] | **+0.174** [+0.088, +0.244] |
| en-es | v1 app − Gemma direct | f1 | **−0.405** [−0.509, −0.313] | **+0.529** [+0.345, +0.682] | **−23.3 pt** [−28.5, −18.4] | +0.073 [−0.045, +0.176] |
| en-es | v1 app, review − f1 | | **+0.179** [+0.110, +0.260] | **−0.229** [−0.315, −0.135] | **+4.6 pt** [+3.1, +7.6] | +0.075 [−0.016, +0.159] |

- **No "vs v1" deltas:** the selected model is v1, so they are 0 by construction. v3a and v3b
  have no test numbers, because they were not final candidates (integrity rule 3).
- **What changed for the app:** the rule weights cost 0.002 AUROC, as reported in "Rule
  calibration". Nothing else moved.

### The review line, dev against test

| Pair | Threshold | Recall, dev → test | Precision, dev → test | Flag rate, dev → test | Major rate, dev / test |
|---|---:|---|---|---|---|
| en-ru | 0.1857 | 0.600 → 0.744 [0.626, 0.835] | 0.469 → 0.392 | 34.6% → 38.4% | 0.270 / 0.202 |
| en-es | 0.4984 | 0.600 → **0.482** [0.421, 0.581] | 0.341 → 0.544 | 10.1% → 8.2% | 0.058 / 0.093 |

- **en-es falls short of the target, and the whole CI is below 0.60.** The dev section warned
  about this:
  - en-es dev has 12 documents and 95 Majors;
  - the threshold itself is uncertain: re-picked on each dev replicate, it spans [0.425, 0.533].
  - The test split has more Majors (9.3% against 5.8%), which is why precision is higher there
    at the same line.
- **en-ru does the opposite:** test recall exceeds the target.
- **The line is not moved.** Changing it now would be a choice made on test.
- **What it means for the app:** on en-es the default line finds 18 points more of the Majors
  than the F1 line (0.30 → 0.48) and still flags under 10% of a job. A reader who works through the flags gets about half
  of the en-es Majors. The riskiest-10% view gets 52%. The README's "60% of the dev Majors" holds
  on dev only. On en-es test it is 48%.

### Real documents (evaluation only)

> **Superseded as the main real-document result** by `docs/REALDOCS_GOLD.md`, section 8: 10
> public documents, 1,440 units (OPUS-MT and DeepSeek-V3.1), 84 Majors, labelled blind and scored
> once on 2026-10-15. There v1 gets AUROC 0.910, and its review line finds 20% of the Majors at
> precision 0.77. Gemma 4 31B direct gets AUROC 0.940 and finds 82% at its review line. The
> counts below are the first pass of `docs/REALDOCS.md`.

Per-document counts are in `docs/REALDOCS.md`, which this pass updated. `scripts/realdocs_eval.py`
applies both v1 lines from `secondread/thresholds.json` to the fused risks. The labels are those of
`demo/answer_key.en-es.json`: our adjudication against the official Spanish for the three public
documents, MQM for WMT24.

| Set | Segments | Major | v1, F1 line (0.7019): caught / false / flag rate | **v1, review line (0.4984) = the selected model**: caught / false / flag rate | Major in the riskiest 10% | Gemma ≥ 0.971: caught / flagged |
|---|---:|---:|---|---|---:|---|
| 3 public OPUS-MT documents (IRS, FDA, EU 261) | 204 | 9 | 2 / 1 / 1.5% | **3** / 3 / 2.9% | 3 of 20 | 8 / 48 |
| Official Spanish of the same 3 pages (control) | 204 | – | 0 flags | **0 flags** | – | not run |
| Official Spanish of all 5 documents (control) | 348 | – | 0 flags | 2 flags (0.6%) | – | not run |
| WMT24 *Stormfall* (MQM labels) | 31 | 21 | 14 / 0 / 45% | 16 / 2 / 58% | 3 of 3 | 19 / 24 |
| WMT24 *Forever Snow* (MQM labels) | 29 | 18 | 8 / 0 / 28% | 12 / 1 / 45% | 3 of 3 | 18 / 27 |

- **Gained at the review line:** FDA-022, "flush list" → "lista de color", at 0.635.
- **New false flags:** FDA-005 (0.555) and EU-073 (0.599), both minor errors in our reading:
  "drogas" for medicines, and "una ciudad, ciudad o región".
- **Official-Spanish flags:** both are NIDDK, at 0.539 and 0.608. Both translations are correct;
  one uses "la persona" for "you".
- **Still missed:** the tax "return" errors (0.08–0.28), "flush" → "lavar" (0.159) and
  "paperless" → "impresa" (0.054). These wrong-sense errors were the target of v3, and v3 did not
  carry over to human labels on dev.

### Spend of this phase

$0. No Tinker or DigitalOcean call was made. Local GPU only: llama-server on `:9973` (v1, `-cram 0
-np 4`) for the 1,104 real-document segments, 23:33–23:38 CEST, then stopped. `:9931` (PID 22074)
was not touched.

### Files

- `scripts/final_test.py` (new): the test run above. It reuses and verifies the score files, takes
  Gemma's lines from en-ru dev, and writes `runs/ship_test/test.json` (git-ignored).
- `scripts/realdocs_eval.py` (new): per-document counts at fixed lines. It writes
  `runs/realdocs_review/eval.json` (git-ignored).
- `runs/realdocs_review/scored.jsonl` and `scored_summary.json` (git-ignored): the end-to-end
  backend re-run at the review line, without Gemma fields. `data/realdocs/scored.jsonl` is
  unchanged. It is the F1-line snapshot, with Gemma, and its risks are identical.

### Ship (2026-10-14, 23:45–23:56 CEST)

- **Served adapter:** `models/secondread-lora.gguf` is v1 (sha256 `5337b2b7…`, byte-identical to
  `models/secondread-lora-v1.gguf`); the v2, v3a and v3b GGUFs are kept next to it. `:9931` was
  restarted with it (`-c 16384 -np 4 -cram 0`). `/lora-adapters` lists it at scale 1.0.
- **Check against the selected model's dev scores:** 40 en-ru dev rows (`probe_local.py … 40
  --workers 4`) match `results/local_v1_enru_dev.jsonl` to within 0.00044 (mean |difference|
  0.00007).
- **Backend:** `/api/health` reports `adapter_version: v1`, `operating_point: review` (source
  `file`), thresholds en-es 0.4984 and en-ru 0.1857 (`dev_review`), and the `f1` point at 0.7019 /
  0.1609.
- **Static demo** (`ui/public/demo-results.json`, rebuilt through the backend): the FDA page, 39
  segments, **3 flagged at 0.4984** (segments 8 and 22, both verified majors, and 5, a minor
  error). At the F1 line the same page had 1 flag. Every risk is within 0.0007 of the F1-line
  build. The riskiest 10% (4 segments) are 8, 22, 5 and 36. The third major, segment 30
  ("flush" → "lavar", 0.159), is not flagged.
- **Checks:** `tests/test_xliff.py` 20/20, `tests/test_thresholds.py` 7/7, `canary_check.py` PASS
  (the canary is in 0 of 12 envelopes, and in 4 with the scrubbers off).
- **Hosted demo:** App Platform rebuilt from commit `f765871` (deployment ACTIVE at 23:54 CEST). Its
  `demo-results.json` is byte-identical to the repo's: 3 of 39 flagged at 0.4984, `dev_review`.

### Reproduce

```sh
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
py scripts/final_test.py                                    # test tables above; scores nothing new
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora-v1.gguf -c 16384 -np 4 -cram 0 \
   --host 127.0.0.1 --port 9973 &
SECONDREAD_LLAMA_URL=http://127.0.0.1:9973 py scripts/realdocs_score.py --no-gemma \
   --out runs/realdocs_review/scored.jsonl --summary runs/realdocs_review/scored_summary.json
kill %1
py scripts/realdocs_eval.py                                 # both lines on data/realdocs/scored.jsonl
py scripts/realdocs_eval.py --scored runs/realdocs_review/scored.jsonl --out runs/realdocs_review/eval_rescored.json
```
