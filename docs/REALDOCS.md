# Real documents, English to Spanish (2026-10-14)

The WMT test sets tell us how SecondRead ranks errors in news, social and literary text. A
post-editor works on other things, such as tax rules, medicine leaflets and regulations. So we
took public documents that already have an official Spanish version, machine-translated the
English, and checked every MT segment against the publisher's own Spanish. The demo jobs built
from them are in [`demo/`](../demo/README.md) (`secondread-demo.en-es.*`,
`answer_key.en-es.json`).

**Short version.** The three public documents give 204 segments with 9 major errors. At its
en-es threshold, SecondRead raises 3 flags. 2 of them are major errors and the third is a
minor one. It misses 7 of the 9 majors, including the most instructive ones: tax "return"
rendered as "refund" and fentanyl patches to be "washed". It raises no flags at all on the
publishers' own Spanish (348 segments). Gemma 4 31B catches 8 of the 9, but flags 48 of the 204
segments to do it. On the WMT24 document, all 14 SecondRead flags are human-labelled Major,
and they cover 14 of its 21 Majors. SecondRead v1 for en-es is precise and quiet, and its recall
is low. This matches the WMT test numbers: precision 0.77 and recall 0.30 at this threshold
(`secondread/thresholds.json`).

## The documents

| Job | Publisher | Licence (checked 2026-10-14) | MT in the job | Segments |
|---|---|---|---|---:|
| `irs-failure-to-file-penalty`: *Failure to File Penalty* | IRS | U.S. government work: "is not subject to copyright and may be freely copied. Credit is requested." ([IRS](https://www.irs.gov/about-irs/use-of-content-from-irsgov)) | OPUS-MT en-es | 52 |
| `fda-medicine-disposal`: *Where and How to Dispose of Unused Medicines* | FDA | public domain: "may be republished, reprinted" without permission ([FDA](https://www.fda.gov/about-fda/about-website/website-policies)) | OPUS-MT en-es | 39 |
| `eu-air-passenger-rights`: Regulation (EC) No 261/2004, Articles 1-19 | EU (EUR-Lex) | "you can re-use the legal documents published in EUR-Lex for commercial or non-commercial purposes", under Decision 2011/833/EU, with the source acknowledged ([legal notice](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html)) | OPUS-MT en-es | 113 |
| `wmt24_the_other_side_stormfall`: *The Other Side: Stormfall* | WMT24 test set (story from AO3) | **research use only** (WMT24 terms); MQM labels Apache-2.0 | MSLC (WMT24 submission) | 31 |

- **Official Spanish:** <https://www.irs.gov/es/payments/failure-to-file-penalty>,
  <https://www.fda.gov/consumers/articulos-para-el-consumidor-en-espanol/como-y-donde-deshacerse-de-los-medicamentos-sin-utilizar>,
  and <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32004R0261>.
- **Fetching:** each document's `data/realdocs/<doc>/SOURCE.md` records the fetched files and
  their sha256, the alignment, and the pairs set aside because the official Spanish is not
  equivalent.
- **Licence checks:** we re-read the IRS and FDA statements from the saved policy pages
  (`data/realdocs/_licence/`) and the EUR-Lex legal notice in a browser. The text of Decision
  2011/833/EU came from the Publications Office (CELLAR, sha256 `2d5bc877…`). Its Article 6
  lets the reuse conditions include acknowledging the source and not distorting the meaning,
  so the job files say that their Spanish is machine translation and not the official text.
- **MT:** OPUS-MT en-es is `Helsinki-NLP/opus-mt-en-es` at revision `5bc4493d`, Apache-2.0
  (model card front matter, checked in the local HF cache). It is a 2020 MarianMT
  transformer-base, run locally with beam 4, one sentence per input.
- **Why these three:** they come from five candidates (also NIDDK *Low Blood Glucose* and
  Ready.gov *Hurricanes*). IRS and FDA had the most verified majors in the publishable MT. EU
  261/2004 is the best-known text, and its MT is almost clean, so it tests whether SecondRead
  stays quiet. NIDDK had no verified major in the OPUS-MT output, and Ready.gov had one.
- **Why this WMT24 document:** *Stormfall* beat the other WMT24 candidate, *Forever Snow* (8
  of 18 Majors caught, also no false flags), on recall.

## Results per document

**Columns:**

- **Major:** verified major meaning errors, labelled by our adjudication against the
  official Spanish. For the WMT24 row they are WMT24 human MQM labels.
- **SecondRead:** flagged at the v1 en-es threshold 0.7019. "False" means a flag on a segment
  that is not Major.
- **Gemma:** Gemma 4 31B direct (same classifier prompt, `scripts/baseline_gemma.py`) at 0.971,
  the repo's Gemma threshold. That threshold is F1-optimal on en-ru dev, because there is no
  en-es dev run.
- **top 10%:** the Majors among the riskiest 10% of the document's segments by SecondRead
  risk. A post-editor with a 10% review budget would read those.

| Job | Segments | Major | SecondRead caught | SecondRead false | top 10% | Gemma caught | Gemma flags (false) |
|---|---:|---:|---:|---:|---:|---:|---:|
| IRS, failure to file penalty | 52 | 5 | **1** | 1 (a minor error) | 1 of 5 | 4 | 24 (20) |
| FDA, medicine disposal | 39 | 3 | **1** | 0 | 2 of 4 | 3 | 16 (13) |
| EU 261/2004, Articles 1-19 | 113 | 1 | 0 | 0 | 0 of 11 | 1 | 8 (7) |
| **3 public documents** | **204** | **9** | **2** | **1** | 3 | **8** | **48 (40)** |
| Official Spanish of the same 3 pages (control) | 204 | – | – | **0 flags** | – | not run | – |
| WMT24 *Stormfall*, MQM labels | 31 | 21 | **14** | **0** | 3 of 3 | 19 | 24 (5) |
| WMT24 *Stormfall*, our labels (28 read) | 28 | 7 | 6 | 8 (all minor) | – | 7 | – |

- **SecondRead:** 3 flags on 204 public-document segments (1.5%), and 0 on all 348 segments
  of the five publishers' official Spanish. Most misses sit well below the threshold:
  0.05-0.32. The exception is FDA "lista de color" at 0.635.
- **Gemma at 0.971:** it flags 24% of the public-document segments. 34 of its 40 extra flags
  are real but minor errors (calques, "pena" for "multa", word order), and 6 are fine. It also
  misses one major, IRS-019, where Gemma's risk is 0.010.
- **Rule checks** (numbers, negation) fired on 2 EU segments, both negation hits on correct MT,
  and on no segment of the IRS or FDA jobs. OPUS-MT kept every rate, deadline and amount
  (5%, 25%, 60 days, 12 months, EUR 250/400/600, 1500 km, 30/50/75%). So no number errors
  exist here to catch.
- **Speed:** the same jobs through the backend took about 0.16-0.18 s per segment on this Mac
  (llama-server, LoRA v1, 4 slots).

## Five examples

"Official ES" is the publisher's translation, or the WMT24 human reference for the WMT row.
"Risk" is SecondRead's fused risk (threshold 0.7019) / Gemma's risk.

**1. Caught. IRS-022: business partner becomes romantic partner.**

- **EN:** Each partner is either an individual (excluding nonresident aliens), or the estate
  of a deceased partner.
- **MT:** Cada **pareja** es una persona (excluidos los extranjeros no residentes) o la
  herencia de una **pareja fallecida**.
- **Official ES:** Cada **socio** es un individuo (excluidos los extranjeros no residentes), o
  el caudal hereditario (patrimonio) de un **socio fallecido**.
- **Why it matters:** this is an eligibility condition for small-partnership penalty relief.
  The MT makes it a rule about couples.
- **Risk:** **0.858** / 1.000.

**2. Caught. FDA-008: the DEA's Take Back Day becomes unrecognisable.**

- **EN:** The U.S. Drug Enforcement Administration sponsors National Prescription Drug Take
  Back Day in communities nationwide.
- **MT:** La Administración Antidrogas de los Estados Unidos patrocina el **Día Nacional de
  Retraso de las Drogas Recluidas** en comunidades de todo el país.
- **Official ES:** … auspicia el **Día Nacional de Recolección de Medicamentos de Prescripción
  Médica** …
- **Why it matters:** "National Day of the Delay of Confined Drugs". A reader cannot find the
  take-back event, and the page calls take-back programs the best way to dispose of most
  medicines.
- **Risk:** **0.800** / 1.000.

**3. Caught (WMT24). Stormfall-24: a character dies by mistranslation.**

- **EN:** "Shit. Its getting bad on the front," Nemic muttered. …
- **MT:** "Shit. **Se quedó mal** en el frente," Nemic **murió**. …
- **Official ES (reference):** "Mierda. La cosa se está poniendo fea en el frente", murmuró
  Nemic. …
- **Why it matters:** "muttered" became "died", while Nemic keeps talking in the same
  paragraph.
  The MQM rater marked only "Se quedó mal" (it ended up bad). Our reading adds "murió".
- **Risk:** **0.871** / 1.000.

**4. Missed. IRS-006: tax "return" becomes "refund".**

- **EN:** The penalty is 5% of the tax due … for each month or partial month **the return is
  late**.
- **MT:** … por cada mes o mes parcial **la devolución es tardía**.
- **Official ES:** … por cada mes o mes parcial **que la declaración se retrasa**.
- **Why it matters:** the 5%-a-month penalty now runs while the taxpayer's refund is late.
  The same false friend appears in IRS-013 (the 60-day minimum penalty) and IRS-019 (how the
  penalty is calculated). SecondRead misses all three. Gemma catches two and misses 019 (risk
  0.010).
- **Risk:** 0.276 / 1.000 (013: 0.083 / 0.999; 019: 0.113 / 0.010).

**5. Missed. FDA-030: fentanyl patches to be "washed".**

- **EN:** That's why the drug comes with instructions to **flush** used or leftover patches.
- **MT:** … instrucciones para **lavar** los parches usados o sobrantes.
- **Official ES:** … instrucciones para **eliminar** los parches usados o sobrantes.
- **Why it matters:** this is a safety instruction for a powerful opioid. "Wash the patches"
  suggests keeping and handling them, which is the opposite of disposal.
- **Risk:** 0.159 / 1.000.

**Also worth showing:**

- **EU-011, missed:** an antonym in a legal definition. In the definition of "ticket",
  "something equivalent in **paperless** form" became "en forma **impresa**" (printed form).
  The official Spanish has "en forma no impresa". Risk 0.054 / 1.000.
- **FDA-022, near miss:** "don't flush your medicine unless it is on the flush list" became
  "… a menos que esté en la **lista de color**" (colour list). Risk 0.635, just under 0.7019.
- **IRS-015, the one false flag:** "la falta de presentación de la pena será máximo" (risk
  0.836). It garbles both penalty names, but the contrast survives, so we call it minor.
- **WMT24 Stormfall-12, caught:** "You make a good point, Specialist" became "**Hacemos un
  buen especialista**" (we make a good specialist). Risk 0.953.
- **WMT24 Stormfall-09 and -16, missed:** short untranslated greetings ("Morning", "Morning
  everyone"). SecondRead scores them 0.033 and 0.022, and Gemma at 0.971 misses them too. They
  are MQM Major, but a Spanish reader understands them, so our reading calls them minor.

## Research comparison, not in the demo

- **NLLB-200** (`facebook/nllb-200-distilled-600M`, CC-BY-NC-4.0, research only, never in
  `demo/`). We translated the same five documents with NLLB-200 and labelled the review pool
  (below). Over the five documents that pool holds 15 verified majors. SecondRead caught 3 of
  them, and these are its most striking real-document catches:
  - "denied boarding" → "aborto negado" (denied abortion), EU-015, 0.757
  - "not low-calorie or reduced-sugar soda" fused into "sodano baja en calorías o reducida
    en azúcar", NIDDK-070, 0.919, read as advice to treat a low with diet soda
  - "De-clutter drains" → "Desordenar los desagües" (make the drains messy), Ready-023,
    0.920

  It missed all 7 IRS majors in the NLLB output. Its one false flag was a minor error.
- **The two documents not chosen (OPUS-MT):** NIDDK *Low Blood Glucose* had 0 majors in the
  pool, 0 SecondRead flags and 5 Gemma flags. Ready.gov *Hurricanes* had 1 major ("storm
  surge" → "la oleada de tormentas", risk 0.144, missed), 0 SecondRead flags and 7 Gemma
  flags.
- **WMT24 *Forever Snow*:** 29 segments, 18 MQM Majors. SecondRead flagged 8, all Major.
  Gemma flagged 27.

## Limitations

- **Our adjudication, not professional annotators:** the labels of the three public documents
  come from Claude, reading the MT against the publisher's official Spanish.
  - **Pass A** labelled the review pool: every SecondRead flag, every Gemma risk ≥ 0.5, every
    rule hit, and a seeded 20% random sample.
  - **Pass B** tried to refute every major or unsure call. It kept "major" only if a reader
    of the MT would be misled about who must do what, when, how much, or what something is.
  - **Pass C** read the 117 remaining OPUS-MT segments of the three chosen documents. It found
    one more major, IRS-019, which both systems missed. So pool-only numbers (NLLB, NIDDK,
    Ready.gov) overstate recall, for SecondRead and for Gemma alike.
  - **Stricter than MQM:** on *Stormfall*, our reading finds 7 majors where the MQM rater
    marked 21. The UI's "Human label" wording should not be used for these labels (see below).
- **The official Spanish is a reference, not ground truth.** For the three chosen documents we
  set aside 9 pairs where the official Spanish says something different from the English: 4
  IRS, 4 FDA, 1 EU. Tables were left out entirely, the IRS penalty tables among them, because
  the Spanish tables lag a year behind.
- **Small n:** 9 verified majors in the publishable MT. Read every rate here as anecdotal;
  confidence intervals would span almost everything.
- **One older MT system:** OPUS-MT (2020) makes word-sense and false-friend errors that modern
  commercial MT and LLM translation make less often, so the error mix does not transfer.
  OPUS training data very likely includes EU legislation (JRC-Acquis, DGT), which would
  explain why its EU 261/2004 output is so close to the official text.
- **Thresholds:**
  - SecondRead: the en-es threshold (0.7019) was picked for F1 on WMT dev, where Majors are
    6% of segments. At that operating point, precision is high and recall is low. A
    review-budget reading (top 10% by risk) catches 3 of the 9 majors.
  - Gemma: its 0.971 is an en-ru number.
- **Segment-level, no context:** the MT ran one sentence at a time, and SecondRead scores one
  segment at a time.
- **Snapshot:**
  - **Source of the numbers:** `data/realdocs/scored.jsonl`, scored 2026-10-14 19:58 with
    LoRA v1, rule calibration `fbc71fcd…` and threshold 0.7019. No new Gemma calls were made
    for this page ($0 DigitalOcean spend).
  - **End-to-end check:** the five job files went through the backend (`secondread.agent`)
    at about 20:45 and gave the same flags. Every risk was within 0.02 of the snapshot.
  - **If the rules or thresholds change:** re-run the commands below and update this page.

## Reproduce and publish

```sh
# 1. (only if agent/checks/thresholds changed) re-score; every Gemma row is cached, so expect $0 new spend
set -a; . ./.env; set +a
HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python scripts/realdocs_score.py --gemma-cap 0.05
# 2. rebuild the demo jobs and the answer key; prints the per-document numbers in the table above
PYTHONPATH=. uv run --frozen python scripts/make_realdocs_demo.py
# 3. static demo: precompute one whole document through the backend (FDA recommended: 1 true flag,
#    0 false flags, and two visible misses with their labels)
HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python -m secondread.server &        # :8787
PYTHONPATH=. uv run python scripts/make_demo_results.py --api http://127.0.0.1:8787 \
    --answer-key demo/answer_key.en-es.json --doc fda-medicine-disposal --n 0
(cd ui && npm run build:demo)
```

`make_demo_results.py --answer-key` carries the document's attribution into `meta` and its
labels into `gold`, with `source: "checked against the official Spanish"`. The UI prints
`gold` as "Human label (…)", which is wrong for these labels. Change that wording, or pass the
WMT24 document instead (`--doc wmt24_the_other_side_stormfall`), whose labels are human MQM.
