# SecondRead

An offline checker for translators who post-edit machine translation (MTPE). It reads an
XLIFF job and ranks segments by how likely they are to contain a **major meaning error**:
a dropped or added negation, a wrong number or unit, an omitted clause, a mistranslation.
Style, grammar and word order are out of scope; other QA tools handle those.

The goal is a shorter re-read: the reviewer starts with the riskiest 10% of the job instead of
reading every segment twice.

> **Status (2026-10-15):** the app serves LoRA v1, trained on Tinker on human labels
> only, as a 36 MB GGUF. By default it flags at the `review` line: per language pair, the highest
> threshold that finds 60% of the dev Majors.
> - **On test, measured once:** the review line finds 74% of the en-ru Majors (precision 0.39,
>   38% of segments flagged). On en-es it finds 48% [0.42, 0.58], below the 60% it was set for
>   on dev (precision 0.54, 8% flagged). The line was not moved after seeing test.
> - **Not shipped:** v2 (5,095 synthetic rows) and v3a/v3b (wrong-sense synthetic rows). Each is
>   better on its own kind of synthetic error, not on human-labelled ones. v3a and v3b lose about
>   0.035 PR-AUC on en-ru dev, so the selection rule, fixed before their scores existed, kept v1.
> - Scores against the base model, v2, v3 and two Gemma 4 31B baselines are in
>   [docs/RESULTS.md](docs/RESULTS.md).
> - **Real public documents** ([docs/REALDOCS_GOLD.md](docs/REALDOCS_GOLD.md)): a gold set of 10
>   documents with an official Spanish version (IRS, FDA, EU 261/2004, CDC, USCIS, SSA and
>   others), 720 segments machine-translated by OPUS-MT and DeepSeek-V3.1, 1,440 units labelled
>   blind by a three-lens AI panel and frozen before scoring, with 84 Majors (80 of them
>   OPUS-MT's). v1 ranks well there (AUROC 0.910; 62% of the Majors in its riskiest 10%), but its
>   en-es review line finds only 20% of them (precision 0.77, 1.5% of units flagged). Gemma 4 31B
>   ranks better (AUROC 0.940) and finds 82% at its line, flagging 12%. The earlier 3-document
>   pass, [docs/REALDOCS.md](docs/REALDOCS.md), is superseded.
> - The rule checks' weights in the app's score are measured on the train split. They no longer
>   pull the ranking below the model's (AUROC −0.002 against the model alone, was −0.010).
> - Done, in-flight and next items are in [docs/STATUS.md](docs/STATUS.md).

## Why offline and open

- **Client texts stay on the translator's machine.** Many agency terms and NDAs restrict
  cloud AI for client content. At run time SecondRead makes no outbound calls:
  - the model runs in a local `llama-server`;
  - the API listens on `127.0.0.1` only;
  - the tokenizer loads with `HF_HUB_OFFLINE=1`;
  - the UI loads nothing from remote hosts;
  - Sentry is off unless you set `SENTRY_DSN`. When it is on, it gets codes, counts and
    timings, never segment text (checked with a canary test, see below).
- **Small and inspectable.** A 4B model in 2.7 GB (Q4_K_M GGUF) plus a 36 MB LoRA runs on a
  laptop: about 0.16 s per segment on an Apple Silicon Mac with 4 parallel slots. The prompt,
  the labels, the training-data recipe and the evaluation are all in this repo.
- **Only public text goes to the cloud.** Training (Tinker) and synthetic-data generation
  (Gemma 4 on DigitalOcean) see public WMT test sets only.

## How it works

```
XLIFF 1.2 / 2.0 / SDLXLIFF ─► parse_xliff ─► number_check ─► negation_check ─► lora_score ─► aggregate
                               (lxml)        (rules: numbers, units,   (llama-server,      (model log-odds
                                              dates, times; negation    first-token        + train-measured
                                              counts for en→ru/en→es)   P(Major) vs P(OK))  rule weights)
```

- **Model:** `Qwen/Qwen3.5-4B` with an MLP-only LoRA (rank 16), trained on
  [Tinker](https://thinkingmachines.ai/tinker/) to answer `Major` or `OK` for one source/MT pair.
- **Score:** both labels are single tokens (`Major` = 33331, `OK` = 3793), so one forward pass
  gives `risk = p(Major) / (p(Major) + p(OK))`.
- **Prompt:** `secondread/prompt.py` is the only definition, used for training, Tinker evaluation
  and local inference.
- **Rule checks:** they add reasons and highlight spans for the UI. Each finding also adds a
  small weight to the model's log-odds, measured on the train split
  (`scripts/calibrate_rules.py` → `secondread/rule_calibration.json`). A finding is a real Major
  error in only 10–50% of hits, so the weights are small. The ranking is close to the model
  alone, but every finding is still shown (RESULTS.md, "Rule calibration").
- **Flag thresholds:** one per adapter, language pair and operating point in
  `secondread/thresholds.json` (`scripts/pick_thresholds.py`), all picked on the fused risk of the
  full dev split:
  - **`review`, the default:** the highest threshold that still finds 60% of the dev Majors
    (`--target-recall 0.60`). For v1 that is en-ru 0.1857 (35% of dev segments flagged, precision
    0.47) and en-es 0.4984 (10% flagged, precision 0.34). The 60% holds on dev only: on test the
    same lines find 74% of the en-ru Majors and 48% of the en-es ones (RESULTS.md, "v3 — test and
    real documents");
  - **`f1`:** the F1-optimal threshold, en-ru 0.1609 and en-es 0.7019 for v1. On en-es it flags 3%
    of dev and finds 35% of the Majors (30% on test).

  `SECONDREAD_OPERATING_POINT=f1`, or `"operating_point"` in a request, switches the point. The
  backend identifies the loaded adapter by its sha256 and reports the version and the active
  operating point in `/api/health`.
- **Write-back:** `secondread/xliff.py` can write the scores into the XLIFF: a Trados comment
  in SDLXLIFF, `<note>` in XLIFF 1.2, notes plus metadata in XLIFF 2.0.
- **Tracing:** each review is one Sentry AI-agent trace (`invoke_agent` → `execute_tool` ×4 → `chat`).

## Quickstart (macOS, Apple Silicon)

Requirements: [uv](https://docs.astral.sh/uv/), Node 22+, Homebrew, about 6 GB of free disk.

```sh
# 1. Tools and Python environment
brew install llama.cpp            # tested with llama-server build 11429 (commit d81235049)
uv sync

# 2. Model files: the Q4 GGUF (2.7 GB) and the tokenizer (13 MB). Never the bf16 weights (9.3 GB).
mkdir -p models
curl -L -o models/Qwen3.5-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
uv run python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-4B', \
  allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'])"

# 3. Run the checker: three terminals
#    -cram 0 turns off llama-server's RAM prompt cache. Every prompt here is unique, and the cache
#    grows the server to 5-8 GB, which macOS swaps out between jobs: the first job after a few idle
#    minutes then takes 16-33 s instead of ~9 s for the 51-segment demo
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9931 \
  # add  --lora models/secondread-lora.gguf  for the shipped LoRA v1 (build it with scripts/export_adapter.sh)
HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python -m secondread.server      # API on 127.0.0.1:8787
cd ui && npm install && npm run dev                                   # UI on http://127.0.0.1:5173
```

Open the UI and drop in `demo/secondread-demo.en-ru.sdlxliff`: 51 segments of public WMT22
text with known errors. `demo/answer_key.json` has the human labels. Or call the API
directly:

```sh
curl -F xliff=@demo/secondread-demo.en-ru.xliff20.xlf -F lp=en-ru http://127.0.0.1:8787/api/review
curl -H 'content-type: application/json' http://127.0.0.1:8787/api/review \
  -d '{"lp":"en-ru","segments":[{"src":"The fee is 5 dollars.","mt":"Плата составляет 7 долларов."}]}'
```

The API contract is in [ui/API.md](ui/API.md). `npm run build:demo` builds a static demo
that needs no backend. It loads `ui/public/demo-results.json`: the FDA medicine-disposal page
(39 segments, OPUS-MT en-es) run through the backend at the review line. It flags 3 segments,
2 of them verified majors, and misses the third major (docs/REALDOCS.md, example 5).

## Reproducing the data and the model

```sh
# Data: Google's MQM annotations (TSV). Not committed; see "Data and attribution".
mkdir -p data/raw && cd data/raw
curl -LO https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/generalMT2022/enru/mqm_generalMT2022_enru.tsv
curl -LO https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/generalMT2024/mqm_generalMT2024_enes.tsv
cd ../..
python3 -I scripts/convert.py data/raw/mqm_generalMT2022_enru.tsv data/enru_wmt22.jsonl en-ru 2022
python3 -I scripts/convert.py data/raw/mqm_generalMT2024_enes.tsv data/enes_wmt24.jsonl en-es 2024
# Document-level split (hash of the doc id): these ratios reproduce the committed data/splits/*.docs.json
python3 -I scripts/split.py data/enru_wmt22.jsonl 0.7 0.15 0.15
python3 -I scripts/split.py data/enes_wmt24.jsonl 0.7 0.1 0.2

py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python "$@"; }   # works in zsh and bash
py scripts/synth_rules.py                    # number/date/unit corruptions + OK variants -> data/synth/
py scripts/synth_llm.py --per-type 20        # Gemma 4 negation/omission/paraphrase edits (DO_MODEL_KEY)
py scripts/build_sft.py                      # v1, human labels only -> data/sft_v1.jsonl
py scripts/build_sft.py --synthetic data/synth/rules.train.jsonl data/synth/llm.train.jsonl \
   --out data/sft_v2.jsonl                   # v2 (not shipped); drops the en-ru neg_insert questions by default
py scripts/train_tinker.py --data data/sft_v1.jsonl --log-path runs/v1 --dry-run  # checks batches, no key
py scripts/train_tinker.py --data data/sft_v1.jsonl --log-path runs/v1 --lr 2e-4  # TINKER_API_KEY; see docs/RUNBOOK.md
py scripts/fetch_adapter.py tinker://<run-id>/weights/final --out ckpt           # adapter files, see below

# Gemma 4 31B baselines on DigitalOcean (DO_MODEL_KEY; cached, budget-capped)
py scripts/baseline_gemma.py --method direct data/splits/en-ru.test.jsonl --out results/gemma_direct_enru_test.jsonl
py scripts/baseline_gemma.py --method gemba data/splits/en-ru.test.jsonl --sample 500 --seed 0 \
   --out results/gemma_gemba_enru_test500.jsonl
```

`tinker checkpoint download` uses a checkpoint-archive endpoint that tinker 0.33.1 marks
deprecated, and it hung for over 10 minutes on 2026-10-14. `fetch_adapter.py` uses
`save_weights_external` instead.

Export the adapter to GGUF. This never downloads the base weights: it converts against
`base35/`, a 172 KB stub of the HF checkpoint that holds only the safetensors headers
(sparse files; do not copy them with Finder, zip or tar without `-S`).

```sh
mkdir -p base35
for f in config.json model.safetensors.index.json; do
  curl -L -o base35/$f https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/$f; done
python3 -I scripts/mkstub.py Qwen/Qwen3.5-4B base35 \
  model.safetensors-00001-of-00002.safetensors=5329398688 model.safetensors-00002-of-00002.safetensors=3990429408
# converter at the tag of the brew llama-server (b11429 = v0.6.0 = commit d81235049)
git clone --depth 1 --branch b11429 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp tools/llama.cpp
git -C tools/llama.cpp sparse-checkout set conversion gguf-py requirements

scripts/export_adapter.sh ckpt               # -> models/secondread-lora.gguf, checked in a real llama-server
py scripts/eval_tinker.py data/splits/en-ru.test.jsonl --model-path tinker://... --out results/lora_enru_test.jsonl
# local scoring: N larger than the split scores every row; the LoRA server must be started with --lora
py scripts/probe_local.py data/splits/en-ru.test.jsonl 100000 --workers 4 --lora-scale 1 --port 9936 \
   --out results/local_v1_enru_test.jsonl
py scripts/report.py base=results/local_base_enru_test.jsonl,results/local_base_enru_dev.jsonl \
   v1=results/local_v1_enru_test.jsonl,results/local_v1_enru_dev.jsonl --n-boot 1000
py scripts/calibrate_rules.py                # rule weights from the train split -> secondread/rule_calibration.json
py scripts/pick_thresholds.py                # F1 flag thresholds (dev-calibrated) -> secondread/thresholds.json
py scripts/pick_thresholds.py --target-recall 0.60   # review thresholds (dev recall >= 0.60), dev only
```

Tests and checks that need no API keys (using the `py` helper above):

```sh
py tests/test_xliff.py              # XLIFF parser/writer, 20 tests
py tests/test_thresholds.py         # operating points: threshold rule, thresholds file, backend choice
py scripts/canary_check.py          # Sentry privacy: a canary in the segment text must reach no envelope
py scripts/bench_review.py data/splits/en-ru.dev.jsonl 100   # latency + ranking via the running API
```

## API keys

All keys go in `.env`, which is git-ignored. The checker itself needs none.

| Key | Used by | Needed for |
|---|---|---|
| `TINKER_API_KEY` | `train_tinker.py`, `eval_tinker.py`, `fetch_adapter.py` | Training the LoRA, exact-logprob evaluation on Tinker, the base-model baseline on Tinker, downloading the adapter |
| `DO_MODEL_KEY` | `synth_llm.py`, `baseline_gemma.py` | Gemma 4 31B (`gemma-4-31B-it`, DigitalOcean serverless inference) for negation, omission and paraphrase synthetic data and for the two Gemma baselines. Use a model-access key limited to that model |
| `SENTRY_DSN` | backend (`secondread/telemetry.py`) | Optional error and agent tracing. Off by default |
| `VITE_SENTRY_DSN` | UI build | Optional browser error reporting. Off by default, and the SDK is not even bundled |

## Data and attribution

No WMT or MQM data is committed except the small demo slice in `demo/` and
`ui/public/demo-results.json`. `.gitignore` keeps `data/` out apart from the split manifests.

- **Source texts and MT outputs:** WMT22 and WMT24 General Machine Translation test sets.
  WMT releases them for research purposes; for any other use, ask the original data owners.
  - Kocmi et al., 2022, *Findings of the 2022 Conference on Machine Translation (WMT22)*.
  - Kocmi et al., 2024, *Findings of the WMT24 General Machine Translation Shared Task*.
- **MQM error annotations:** Google's
  [wmt-mqm-human-evaluation](https://github.com/google/wmt-mqm-human-evaluation), Apache-2.0.
  - Freitag et al., 2021, *Experts, Errors, and Context: A Large-Scale Study of Human
    Evaluation for Machine Translation*, TACL.
  - Freitag et al., 2022, *Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural
    Metrics Are Better and More Robust*. The WMT22 en-ru annotations were made by Unbabel.
- **Label rule:** `Major` means at least one major or critical error in the meaning family
  (accuracy and terminology categories). Each unit has one annotator, so expect label noise.
- **Synthetic data:** derived from the train-split documents only; `build_sft.py` refuses dev or
  test documents.
- **GEMBA-MQM:** the Gemma baseline uses the GEMBA-MQM prompt (Kocmi & Federmann, 2023,
  [MicrosoftTranslator/GEMBA](https://github.com/MicrosoftTranslator/GEMBA), commit
  `ea90068f`), which is **CC BY-SA 4.0**. The adapted prompt is
  [`prompts/gemba_mqm.txt`](prompts/gemba_mqm.txt); its header carries the attribution, the
  license and the list of changes. That file is not under the code license.
- **Gemma 4 outputs:** Gemma 4 is Apache-2.0, and neither Google's nor DigitalOcean's terms
  restrict training on its outputs (reading of 2026-10-14 in
  [docs/LICENSES-NOTES.md](docs/LICENSES-NOTES.md)). Synthetic rows keep their
  `synth_generator`/`synth_model` provenance fields; if the data is ever released, label it as
  AI-generated.
- **Evaluation-only sets with non-commercial licenses:** ACES (CC BY-NC-SA 4.0) and HalOmi
  (CC BY-NC 4.0). If used, they stay out of training and out of git.
- **Base model:** Qwen3.5-4B by the Qwen team. The GGUF is the `unsloth/Qwen3.5-4B-GGUF`
  conversion.
