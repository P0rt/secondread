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
- **Terms:** the en-ru files are a research and demo excerpt (51 segments). They are not
  licensed for commercial redistribution of the WMT texts.

# English to Spanish: real documents (en-es)

Four post-editing jobs, one document each. Three are real public documents that have an
official Spanish version; we machine-translated the English with an openly licensed MT system,
so every error in them is a real error of a real MT system on a real text. The fourth is a WMT24
test document with human MQM labels, for comparison with the en-ru job above.

| File | Document | Segments | Major | Labels |
|---|---|---:|---:|---|
| `secondread-demo.en-es.irs-failure-to-file-penalty.xlf` and `.sdlxliff` | IRS, *Failure to File Penalty* | 52 | 5 | our adjudication against the official Spanish |
| `secondread-demo.en-es.fda-medicine-disposal.xlf` | FDA, *Where and How to Dispose of Unused Medicines* | 39 | 3 | our adjudication against the official Spanish |
| `secondread-demo.en-es.eu-air-passenger-rights.xlf` | EU Regulation (EC) No 261/2004 on air passenger rights, Articles 1-19 | 113 | 1 | our adjudication against the official Spanish |
| `secondread-demo.en-es.wmt24_the_other_side_stormfall.xlf` | WMT24 literary document *The Other Side: Stormfall* | 31 | 21 | WMT24 human MQM |
| `answer_key.en-es.json` | labels, one-line explanations, the official Spanish, and each segment's key in each file | 235 | 30 | |

Results, the best examples and the limits are in [`docs/REALDOCS.md`](../docs/REALDOCS.md).

## Files

- **XLIFF 1.2:** one `<file>` per job (`source-language` en-US or en-GB, `target-language`
  es-US, es-ES or es-MX), one `<trans-unit>` per segment with the segment id as `id`, the MT
  in `<target state="needs-review-translation" state-qualifier="leveraged-mt">`, and the
  page section in `<note from="secondread-demo" annotates="source">`. The header `<note>`
  carries the attribution.
- **SDLXLIFF** (IRS only): what Trados Studio writes for a pre-translated file. UTF-8 with BOM,
  `seg-source`/`mrk mtype="seg"`, `sdl:seg conf="Draft" origin="mt"
  origin-system="Helsinki-NLP/opus-mt-en-es"`.
- `answer_key.en-es.json`: `documents[]` holds each job's sources, licence, attribution, MT
  system and label rule. `segments[].keys.xliff12` / `.sdlxliff` are the keys
  `secondread.xliff` gives the segment. Every file is parsed back by `secondread.xliff` at
  build time and must give the same segments in the same order.

## What is in it

- **Segments are aligned sentence pairs, not the whole page.** `scripts/realdocs_align.py`
  pairs each English sentence with its official Spanish sentence and keeps a pair only when both
  sides carry the same numbers. It leaves out tables, link lists, and the pairs where the
  official Spanish says something different from the English (4 IRS, 4 FDA, 1 EU). Every kept
  pair was read by hand.
- **MT for the three real documents:** OPUS-MT en-es, `Helsinki-NLP/opus-mt-en-es` at
  revision `5bc4493d`, Apache-2.0. It ran locally with beam 4, one sentence per input
  (`scripts/realdocs_mt.py`). One system per job, as in a real job.
- **MT for the WMT24 job:** MSLC, a WMT24 submission (NRC Canada). Its output ships with the
  WMT24 test data.
- **Not in it:** output of NLLB-200 (`facebook/nllb-200-distilled-600M`, CC-BY-NC-4.0). We
  scored that output only as a research comparison (`docs/REALDOCS.md`).
- **The Spanish in the job files is machine translation, not the official text.** It
  contains real meaning errors on purpose, because that is what the demo is for. The
  publishers' official Spanish is in `answer_key.en-es.json` (`official_es`). Do not quote the
  job files as the IRS's, the FDA's or the EU's Spanish.
- **Labels, real documents:** our adjudication against the official Spanish translation. A
  reader (Claude, comparing the MT with the publisher's own Spanish) labelled each segment
  `major_error`, `minor_error` or `ok`. A second pass tried to refute every major call, and kept
  it only if a reader of the MT would be misled about who must do what, when, how much, or
  what something is. `Major` = `major_error`. These are not professional human annotations.
  `label_pass` says which pass gave the final label: `a` is the first pass, `b` the refutation
  pass, and `c` covers the segments outside the sampled review pool. Pass `c` read the rest of
  each document, so every segment of the three jobs is labelled.
- **Labels, WMT24 job:** WMT24 human MQM, one Unbabel rater for the whole document. `Major`
  means at least one major or critical error in the meaning family, as in the en-ru job. Our
  own reading of the same segments is in `our_reading`, for comparison. It is stricter: 7
  major, against MQM's 21.
- **Held out:** SecondRead never trained on the three real documents (they were fetched on
  2026-10-14). The WMT24 document is in the en-es test split (`data/splits/en-es.docs.json`).

## Try it

```sh
curl -F xliff=@demo/secondread-demo.en-es.irs-failure-to-file-penalty.sdlxliff -F lp=en-es \
     http://127.0.0.1:8787/api/review
```

## Rebuild

```sh
PYTHONPATH=. uv run --frozen python scripts/make_realdocs_demo.py
```

It reads `data/realdocs/` (git-ignored: scores, adjudication, WMT24 rows), checks that every
job uses publicly displayable MT, and writes the files above. The build is deterministic.

## Sources, attribution and terms of use (checked 2026-10-14)

Each job file repeats its attribution line in the header `<note>`, and so does
`answer_key.en-es.json` (`documents[].attribution`). Show that line with any excerpt.

- **IRS, "Failure to File Penalty"**
  - EN <https://www.irs.gov/payments/failure-to-file-penalty>, official ES
    <https://www.irs.gov/es/payments/failure-to-file-penalty>
  - Licence: U.S. federal government work (17 U.S.C. 105). IRS.gov: content made by federal
    employees "is not subject to copyright and may be freely copied. Credit is requested."
    (<https://www.irs.gov/about-irs/use-of-content-from-irsgov>)
  - Attribution: *Source: Internal Revenue Service, "Failure to File Penalty" / "Multa por
    incumplimiento de presentar la declaración de impuestos", IRS.gov (retrieved 2026-10-14).
    U.S. government work. Segmented and aligned by SecondRead; no IRS seal or logo is reused and
    no endorsement is implied.*
- **FDA, "Where and How to Dispose of Unused Medicines"**
  - EN <https://www.fda.gov/consumers/consumer-updates/where-and-how-dispose-unused-medicines>,
    official ES
    <https://www.fda.gov/consumers/articulos-para-el-consumidor-en-espanol/como-y-donde-deshacerse-de-los-medicamentos-sin-utilizar>
  - Licence: public domain. FDA Website Policies: the site's contents "are in the public domain
    and may be republished, reprinted" without permission. Credit to FDA is appreciated.
    (<https://www.fda.gov/about-fda/about-website/website-policies>)
  - Attribution: *Source: U.S. Food and Drug Administration, "Where and How to Dispose of
    Unused Medicines" / "Cómo y dónde deshacerse de los medicamentos sin utilizar" (retrieved
    2026-10-14). Segmented and aligned by SecondRead; no FDA logo is reused and no endorsement
    is implied.*
- **EU, Regulation (EC) No 261/2004 (OJ L 46, 17.2.2004, p. 1), original text, Articles 1-19**
  - EN <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32004R0261>, official
    ES <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32004R0261>
  - Licence: the EUR-Lex legal notice says "The Commission's document reuse policy is based on
    Decision 2011/833/EU. Unless otherwise specified, you can re-use the legal documents
    published in EUR-Lex for commercial or non-commercial purposes."
    (<https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>, read 2026-10-14).
    Under Decision 2011/833/EU, Article 6, the reuse conditions may include acknowledging the
    source and not distorting the original meaning. So the job files say that their Spanish
    is machine translation, and the answer key gives the official Spanish next to it.
  - Attribution: *Source: EUR-Lex, Regulation (EC) No 261/2004, OJ L 46, 17.2.2004, p. 1.
    © European Union, https://eur-lex.europa.eu, 1998-2026. Segmented and aligned by
    SecondRead; only European Union documents published in the Official Journal are deemed
    authentic.*
- **MT system for the three documents above:** Helsinki-NLP OPUS-MT en-es (Tiedemann and
  Thottingal, 2020, *OPUS-MT – Building open translation services for the World*, EAMT),
  <https://huggingface.co/Helsinki-NLP/opus-mt-en-es>, Apache-2.0 (model card at revision
  `5bc4493d`).
- **WMT24 job, *The Other Side: Stormfall***
  - Source text, MT (system MSLC) and reference: WMT24 General MT test set, en-es (Kocmi et
    al., 2024, *Findings of the WMT24 General Machine Translation Shared Task*,
    <https://www2.statmt.org/wmt24/translation-task.html>). The test data "can be freely used
    for research purposes", with a citation of the findings paper. The English story comes from
    Archive of Our Own, and its author keeps the copyright. **This job is not openly licensed:**
    it is a research and demo excerpt, the same footing as the en-ru job, and must not be used in
    commercial material.
  - MT system: Larkin, Lo and Knowles, 2024, *MSLC24 Submissions to the General Machine
    Translation Task*, WMT 2024.
  - MQM labels: [`google/wmt-mqm-human-evaluation`](https://github.com/google/wmt-mqm-human-evaluation),
    `generalMT2024/mqm_generalMT2024_enes.tsv`, Apache-2.0. Unbabel raters, Freitag et al.,
    2024, *Are LLMs Breaking MT Metrics? Results of the WMT24 Metrics Shared Task*.
