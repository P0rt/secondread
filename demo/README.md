# SecondRead demo job (WMT22 en-ru)

A small post-editing job for trying SecondRead: 51 machine-translated segments from five public
WMT22 test documents, as the three file types a translator is likely to receive.

| File | Format |
|---|---|
| `secondread-demo.en-ru.xliff12.xlf` | XLIFF 1.2 (one `<file>` per document, `state-qualifier="leveraged-mt"`) |
| `secondread-demo.en-ru.xliff20.xlf` | XLIFF 2.0 (one `<unit>`/`<segment>` per WMT segment) |
| `secondread-demo.en-ru.sdlxliff` | Trados SDLXLIFF (UTF-8 with BOM, `seg-source`/`mrk mtype="seg"`, `sdl:seg conf="Draft" origin="mt"`) |
| `answer_key.json` | Gold labels: `Major`/`OK` per segment, the MQM error spans, and the segment key in each format |

All three files contain the same segments in the same order. `answer_key.json` maps them
(`segments[i].keys.xliff12` / `.xliff20` / `.sdlxliff`) to the keys `secondread.xliff` produces.

## What is in it

- **Language pair:** English to Russian.
- **MT system:** eTranslation, the WMT22 submission. One system for the whole job, as in a
  real job.
- **Documents:** all 10 or 11 annotated segments of each, in document order. WMT22 MQM
  documents are 10-segment excerpts, so a job of about 50 segments needs five:

| Document | Domain | Segments | Major |
|---|---|---|---|
| `ecommerce_en_36` (watch dial listing) | ecommerce | 10 | 6 |
| `ecommerce_en_13` (dog supplement listing) | ecommerce | 10 | 4 |
| `en_fr_CLIENT-02_default_2020-12-20-16_doc06` (e-reader support chat) | conversation | 11 | 1 |
| `brisbanetimes.com.au.47160` | news | 10 | 3 |
| `guardian.44758` | news | 10 | 0 |

- **Labels:** 14 of 51 segments are `Major` (27.5%). For comparison, eTranslation scores
  26.7% over the whole en-ru test split.
- **Label rule:** the one used in training (`scripts/convert.py`). `Major` means at least one
  major or critical MQM error in the meaning family: accuracy or terminology, which covers
  mistranslation, omission, addition, untranslated text, wrong named entity and
  number/date/unit format errors.
- **Annotators:** each segment has exactly one MQM annotator, so expect some label noise.
- **Held out from training:** every document is in the test split
  (`data/splits/en-ru.docs.json`), so the model never saw them.
- **Text:** the WMT text is unchanged. The only exception is one segment, where CSV quoting
  left in the TSV (`""keen…""`) was undone, and only because no gold span on that side
  would move. `#PRS_ORG#` is WMT's own anonymisation placeholder.

## Rebuild

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/make_demo_xliff.py
```

The build is deterministic: ids are UUIDv5 values, so rebuilding gives byte-identical files.
Change the system or the documents with `--system` and `--docs`.

## Sources, attribution and terms of use

- **Source texts and MT output:** WMT22 General Machine Translation Task test set.
  - Kocmi et al., 2022, *Findings of the 2022 Conference on Machine Translation (WMT22)*.
  - <https://www.statmt.org/wmt22/translation-task.html>
  - WMT releases its test data for research purposes. For any other use, ask the original
    data owners.
- **Error annotations (MQM):** Google's
  [`wmt-mqm-human-evaluation`](https://github.com/google/wmt-mqm-human-evaluation),
  Apache-2.0.
  - The en-ru annotations were done by Unbabel for the WMT22 Metrics Shared Task.
  - Freitag et al., 2022, *Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural
    Metrics Are Better and More Robust*.
  - The repository asks users to cite Freitag et al., 2021, *Experts, Errors, and Context: A
    Large-Scale Study of Human Evaluation for Machine Translation* (TACL).
- **Terms:** this folder is a research and demo excerpt (51 segments). It is not licensed
  for commercial redistribution of the WMT texts.
