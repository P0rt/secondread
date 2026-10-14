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
  locally; doing so is free and would make the F1 column steadier.
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
