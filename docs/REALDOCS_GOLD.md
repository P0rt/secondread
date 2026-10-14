# Real-document gold set, English to Spanish (gold v1, 2026-10-14)

This set labels every machine-translated segment of ten public English documents as **Major**
or **OK**. Each document has an official Spanish version. Major means that a Spanish-only reader
would be misled about the content, or could not get it: a wrong, missing, added or
unintelligible action, actor, object, amount, time, condition, negation or name. There are 720
source segments, and each has two MT outputs, so the set has **1,440 units**. It is a test set
for Major/OK detectors such as SecondRead. Nothing in this step scored SecondRead, Gemma or any
other detector. Section 6 says how to do it.

**Short version.**

- **Majors.** 84 of the 1,440 units are Major (5.8%; 95% document-bootstrap CI 3.5-8.9%), and
  3 of them are critical.
- **By MT system.**
  - OPUS-MT (Marian, 2020) has 80 Majors in 720 units (11.1%; CI 6.5-17.3%). Every document
    has at least 2.
  - DeepSeek-V3.1 has 4 in 720 (0.6%; CI 0.0-1.1%). 7 of the 10 documents have none.
- **Panel agreement.** Three blind AI annotator "lenses" labelled every unit. Fleiss' kappa on
  Major is 0.876 (CI 0.828-0.915). 98.0% of the units are unanimous.
- **Adjudication.** 16 units (1.1%) went to adjudication. The adjudicator kept the majority
  vote on all 16: the 3 critical units stayed Major, and the 13 lone Major votes became OK.
  So the primary labelling and the pre-registered labelling are the same.
- **Earlier labels.** The earlier real-document labels cover 238 of the OPUS-MT units. All 10
  of their Majors are Major here too. The new gold has 24 more Majors, all of which the earlier
  pass had called "minor". Cohen's kappa is 0.42, and raw agreement is 90%.
- **Main caveat.** The annotators are AI and they saw the official Spanish. Almost all the
  Majors come from one older MT system.

## 1. Documents

The list in `data/realdocs/gold/DOC_LIST.json` was frozen at 2026-10-14 21:53 (+02:00), before any
MT output of the five added documents existed. No document can be dropped, replaced or added
because of results. Each document's segments are fixed by the sha256 of its `aligned.jsonl`.

**Acceptance rules:**

- public, high-stakes instructions or rules for the general public;
- an English original;
- reuse clearly allowed;
- an official Spanish counterpart written by people on the agency's own site (machine-translation
  widgets do not count);
- HTML on both sides that aligns as one document, with at least 40 candidate pairs;
- at most one added document per publisher.

Where the official Spanish of a pair is not equivalent to the English, the pair was set aside
during the hand check (23 pairs, listed in each document's `ref_deviations.jsonl`). That check
judged the reference only, never an MT output.

| Document | Publisher | Segments | Set aside | Licence |
|---|---|---:|---:|---|
| `irs-failure-to-file-penalty` | IRS | 52 | 4 | U.S. federal work, free to copy |
| `fda-medicine-disposal` | FDA | 39 | 4 | U.S. federal work, public domain |
| `eu-air-passenger-rights` (Reg. 261/2004) | EU (EUR-Lex) | 113 | 1 | reuse under Decision 2011/833/EU |
| `niddk-hypoglycemia` | NIDDK (NIH) | 91 | 3 | U.S. federal work, copyright free |
| `ready-hurricanes` | FEMA, Ready.gov | 53 | 1 | U.S. federal work, public domain |
| `uscis-common-scams` | USCIS | 88 | 5 | U.S. federal work, public information |
| `ssa-disability-qualify` | SSA | 81 | 1 | U.S. federal work, public domain |
| `cdc-carbon-monoxide` | CDC | 43 | 0 | U.S. federal work, public domain |
| `ready-home-fires` | FEMA, Ready.gov | 83 | 2 | U.S. federal work, public domain |
| `medlineplus-pt-inr` | MedlinePlus (NLM) | 77 | 2 | U.S. federal work, public domain |
| **Total** | | **720** | **23** | |

The first five are the original real documents (`docs/REALDOCS.md`), and the last five were added
for the gold set. The licence evidence pages are saved under `data/realdocs/_licence/` and
`data/realdocs/gold/_licence/`. URLs and retrieval dates are in `DOC_LIST.json`.

## 2. MT systems

The details are in `data/realdocs/gold/mt/MT_META.json`.

| System key | Model | Licence | How it ran |
|---|---|---|---|
| `marian` | `Helsinki-NLP/opus-mt-en-es`, revision `5bc4493d` (MarianMT transformer-base, OPUS 2020) | Apache-2.0 | Local (MPS, float32). Beam search with 4 beams, no sampling, one sentence per input with outputs joined by a space, no context. The first five documents reuse `data/realdocs/<doc>/mt_marian.jsonl`. |
| `deepseek-v3.1` | `deepseek-ai/DeepSeek-V3.1`, HF revision `c0781d03` (671B MoE, non-thinking chat mode) | MIT | Tinker sampling API. Temperature 0, seed 0. Prompt: "professional translator … neutral Spanish … translation only". The two previous source segments are given as context. Whole segment per call. Estimated cost $0.28. |

Both licences allow the outputs to be shown in the public demo with attribution.

## 3. Method

### 3.1 Blind three-lens panel

The protocol is `docs/ANNOTATION_PROTOCOL.md` v2, frozen 2026-10-14. Its lens texts are pinned by
sha256 in section 2 of that file, and they are unchanged.

- **Unit.** A unit is `{uid, doc_id, seg_index, context_src, src, mt, ref_es}`.
  - `context_src` holds the two previous English segments.
  - `ref_es` is the publisher's Spanish. It is supporting evidence only, never the gold.
  - Units are shuffled within each document (seed 0) and split into 13 batches of about 110
    (`batches/MANIFEST.json`).
- **Blinding.** Annotators never saw the MT system (the key is `key_systems.json`), any
  automatic score, SecondRead or Gemma output, or the earlier labels. They never saw each
  other's lens.
- **Three lenses.** Each lens annotated every unit, with its own procedure:
  - **MQM rater:** inventories the error spans first.
  - **Post-editor:** sorts its own minimal edits into MEANING and FORM edits.
  - **Spanish-only reader:** reads the Spanish first, writes "Read as: …", and only then reads
    the English.
- **Shared rules.** All three share one severity scale (none / minor / major / critical), one
  MQM category list and rules R1 to R5. Two rules matter most:
  - **R1:** a literal rendering whose ordinary Spanish sense differs, or a phrase too broken to
    say anything, is major when it carries content. "The gist is recoverable" is no reason for
    minor.
  - **R2:** truncation and dropped content are major.
- **Coverage.** All 39 lens-batch files are complete: 1,440 records per lens, with no missing,
  duplicate or unknown uid.

### 3.2 Calibration against WMT24 human MQM

Before the freeze, the lenses (v1 texts) labelled 247 WMT24 en-es units that professional MQM
raters had already labelled. The set had 22 whole source chunks, 14 systems and 44 human Majors.
The human label is Major when the unit has any major or critical span in the meaning family.
CIs come from a cluster bootstrap over the chunks. Full numbers are in
`data/realdocs/gold/calib/RESULTS.md`.

| Rater (v1) | Precision | Recall | F1 (95% CI) | Cohen's kappa vs human (95% CI) |
|---|---:|---:|---|---|
| MQM lens | 0.67 | 0.75 | 0.71 (0.59-0.80) | 0.64 (0.51-0.75) |
| Post-editor lens | 0.62 | 0.68 | 0.65 (0.57-0.75) | 0.57 (0.47-0.69) |
| Reader lens | 0.64 | 0.61 | 0.63 (0.53-0.73) | 0.55 (0.44-0.67) |
| **Majority (2 of 3)** | 0.65 | 0.68 | **0.67 (0.57-0.75)** | **0.59 (0.48-0.69)** |

- **Lens agreement.** The lenses agreed with each other far more than with the humans:
  Fleiss' kappa 0.90 (0.82-0.96).
- **Human-human agreement.** This cannot be measured on WMT24 en-es, which has one rater per
  segment. The published MQM figures are lower: segment-level Cohen's kappa 0.21-0.23 on WMT20/21
  en-de (arXiv 2407.03658), so a kappa of 0.59 against one human is at or above that range. The
  metrics differ, so the comparison is only indicative.
- **Misses.** Half of the misses (7 of 14) were literal renderings that change the Spanish sense,
  which the lenses called minor ('buena llamada' for "good call"). This led to rule R1.
- **False alarms.** Most false alarms (10 of 16) looked like real errors that the single human
  left unmarked.
- **Not re-measured.** The v2 texts were not re-run on the calibration set, because that would
  tune on it.

### 3.3 Aggregation

- **Vote.** A lens votes Major when it says major or critical. A unit is Major when at least 2
  of the 3 lenses vote Major.
- **Queue.** These units go to adjudication:
  - units where any lens says critical (pre-registered);
  - units with a single Major vote at confidence 0.7 or more (pre-registered);
  - every other unit with a single Major vote (amendment A1, added because calibration had 4
    human Majors among 6 lone Major votes, all at confidence 0.50-0.55).
- **Not queued.** A 2-1 split whose majority is Major is not adjudicated.
- **Final label.** The adjudicated label where one exists, else the majority.
  `label_preregistered` rebuilds the labelling that ignores A1.

### 3.4 Adjudication

The queue holds 16 units (1.11% of 1,440). 3 are critical and 13 are lone Major votes at
confidence 0.5-0.6. No lone Major vote reached 0.7, so the pre-registered rules alone would have
queued only the 3 critical units. All 16 are OPUS-MT outputs, which was learned only after the key
was joined.

**Blindness.** The senior adjudicator (Claude) read src, context, MT, `ref_es` and the three
annotations of each unit, and did not see the system, any score or the old labels. To keep one
standard, it also looked at the panel's labels on the same phrase elsewhere in the set (for
example 'CAD' and 'Asignaciones por Compasión'). It wrote its decisions to `adjudication.jsonl`
at 2026-10-14 22:31:05 +02:00 (sha256 `44b6fe02…`). Only after that did `scripts/gold_finalize.py`
read `key_systems.json` and the old labels.

**Standard.** The adjudicator applied the protocol's scale and R1 to R4 to one question: what does
a Spanish-only reader most likely understand?

- **Major:** the reading is coherent but different from the English, or a piece of content is
  missing or unintelligible.
- **OK:** the Spanish is odd or partly broken, but the reader's most likely reading is the
  intended content.

| Outcome | Units | Decision |
|---|---:|---|
| critical, panel 3/3 Major | 3 | Major (critical): g0006 IRS drops "timely or complete" (late filers read "no penalty"); g0112 FDA "lavar los parches" (wash fentanyl patches) for "flush"; g1232 Ready "Si hace calor" (if the weather is hot) for "if the doorknob or door is hot" |
| lone Major vote, panel 2/3 OK | 13 | OK, minor. 'el fracaso' for "the failure [to file]" (4 IRS units); garbled 'intereses de suma'; 'hayan confirmado una reserva'; 'partidarios'; 'que se requiere para completar'; 'Subsidios compasivos'; this output's own acronym 'CAD'; a verbless clause; exclamative 'lo bien que'; 'Latidos cardíacos rápidos y respiración'. Each line in `adjudication.jsonl` gives its reason. |

The adjudicator did not overturn the majority anywhere, so the primary and the pre-registered
labellings are identical (84 Majors in each).

### 3.5 Non-binding audit

The protocol does not adjudicate 2-1 splits with a Major majority, or unanimous units. To see how
firm the Major boundary is, the adjudicator also read, before the key was joined, three groups:
the 16 not-queued 2-1 Majors, a seeded sample (seed 0) of 40 unanimous OK units, and a seeded
sample of 20 unanimous Major units. Its opinions are in `adjudicator_audit.jsonl`, marked
`binding: false`. **They change no label.**

| Group | Units | Agrees with gold | Would call OK |
|---|---:|---:|---|
| 2-1 split, majority Major (not queued) | 16 | 10 | g0105 'disponer de', g0285 missing verb, g0394 'abordo denegado', g0829 loose attachment, g1164 and g1170 'salida(s)' for outlet(s) |
| Unanimous OK, sample | 40 | 40 | none |
| Unanimous Major, sample | 20 | 18 | g1184 'utilidades', g0158 English heading (both borderline) |

**What this means:** "OK" is solid. The Major class has a soft edge: literal renderings that the
context lets a reader repair. The panel, following R1, calls them Major; the adjudicator's
reading test leans OK. Calibration showed that professional WMT raters mark this kind of
rendering Major more often than an AI reading for the gist (S1). So the adjudicator's lean may
be the bias here, not the panel's. Treat about 6 to 10 of the 84 Majors as contestable.

## 4. Statistics

Every number below is in `data/realdocs/gold/gold_stats.json`. CIs are percentile 95% CIs from a
document-level cluster bootstrap (10,000 replicates, seed 0, resampling the 10 documents).

### 4.1 Labels

| | Units | Major | Major rate (95% CI) | Critical |
|---|---:|---:|---|---:|
| All | 1,440 | 84 | 5.8% (3.5-8.9%) | 3 |
| `marian` (OPUS-MT) | 720 | 80 | 11.1% (6.5-17.3%) | 3 |
| `deepseek-v3.1` | 720 | 4 | 0.6% (0.0-1.1%) | 0 |

- **Final severities:** 822 none, 534 minor, 81 major and 3 critical.
- **Categories of the 84 Majors:** accuracy/mistranslation 70, accuracy/omission 7, terminology
  5, accuracy/untranslated 2.
- **Segments:** 82 segments have a Major in at least one system, and 2 have one in both.
- **DeepSeek's 4 Majors:**
  - g0212: 'al menos un día después' for "the day after", which narrows an EU 261 entitlement;
  - g0724 and g0768: 'libertad condicional' (criminal parole) for immigration parole;
  - g0918: a 10-year eligibility window that shifts by one year.

  Only g0918 is unanimous.

| Document | Segments | `marian` Major | `deepseek-v3.1` Major | Total | Adjudicated |
|---|---:|---:|---:|---:|---:|
| irs-failure-to-file-penalty | 52 | 14 (27%) | 0 | 14 | 6 |
| fda-medicine-disposal | 39 | 10 (26%) | 0 | 10 | 1 |
| eu-air-passenger-rights | 113 | 4 (4%) | 1 | 5 | 1 |
| niddk-hypoglycemia | 91 | 2 (2%) | 0 | 2 | 0 |
| ready-hurricanes | 53 | 6 (11%) | 0 | 6 | 0 |
| uscis-common-scams | 88 | 12 (14%) | 2 | 14 | 2 |
| ssa-disability-qualify | 81 | 5 (6%) | 1 | 6 | 2 |
| cdc-carbon-monoxide | 43 | 12 (28%) | 0 | 12 | 1 |
| ready-home-fires | 83 | 13 (16%) | 0 | 13 | 1 |
| medlineplus-pt-inr | 77 | 2 (3%) | 0 | 2 | 2 |
| **Total** | **720** | **80** | **4** | **84** | **16** |

### 4.2 Agreement among the lenses

| Measure | Value |
|---|---|
| Fleiss' kappa, binary Major (95% CI) | **0.876** (0.828-0.915) |
| Fleiss' kappa, 3-class none / minor / major+ (95% CI) | 0.758 (0.694-0.820) |
| Pairwise Cohen's kappa, binary | MQM / post-editor 0.908, MQM / reader 0.832, post-editor / reader 0.888 |
| Pairwise Cohen's kappa, 3-class | 0.784, 0.771, 0.722 |
| Major votes (of 3) per unit | 0: 1,343; 1: 13; 2: 16; 3: 68 |
| Unanimous on Major / on 3-class | 98.0% / 80.6% |
| Major votes per lens | MQM 88, post-editor 85, reader 76 (critical: 3, 2, 3) |
| Fleiss' kappa by system, binary | `marian` 0.878 (0.832-0.914); `deepseek-v3.1` 0.665 (0.497-1.000, with only 4 majority Majors) |

- **Compared with calibration.** The lenses are a little less collapsed than in calibration
  (0.876 against 0.90), and every lens now says critical at least twice (calibration: 2, 0, 0).
  The reader lens is still the most lenient.

### 4.3 Adjudication share

| | Units | Share of 1,440 |
|---|---:|---:|
| Adjudicated, all reasons | 16 | 1.11% |
| `critical` | 3 | 0.21% |
| `minority_major_conf>=0.7` | 0 | 0% |
| `minority_major_low_conf` (amendment A1) | 13 | 0.90% |
| Pre-registered queue only | 3 | 0.21% |
| Majority overturned | 0 | 0% |

### 4.4 Agreement with the earlier real-document labels

The earlier labels (`data/realdocs/adjudication_final.jsonl` and `adjudication_fill.jsonl`, passes
A, B and C in `docs/REALDOCS.md`) overlap the gold on 238 OPUS-MT units. The MT text is identical
(up to whitespace) on all 238. Coverage differs by document:

- IRS, FDA and EU: every OPUS-MT segment, 204 units.
- NIDDK and Ready hurricanes: only the earlier review pool, 34 units.

An earlier `major_error` counts as Major; `minor_error` and `ok` count as OK.

| | Earlier: Major | Earlier: OK | Total |
|---|---:|---:|---:|
| **Gold: Major** | 10 | 24 | 34 |
| **Gold: OK** | 0 | 204 | 204 |
| **Total** | 10 | 228 | 238 |

- **Agreement.** Cohen's kappa is 0.42 and raw agreement is 89.9%.
  - Whole-document part: 9 / 19 / 0 / 176.
  - Review-pool part: 1 / 5 / 0 / 28.
- **Severity cross-tab.** The gold's 2 critical units are earlier `major_error`s. Its 24 extra
  Majors are all earlier `minor_error`s. Of the 172 earlier `ok`s, the gold calls 108 none and
  64 minor.
- **The difference is one of definition, not noise.** Every earlier Major is a gold Major, so
  the earlier Majors are a strict subset.
  - 21 of the 24 extra Majors are unanimous on the panel. 12 of the 24 are calls that the earlier
    pass B re-read and downgraded to minor.
  - **Typical cases:**
    - IRS: the garbled compound 'la falta de presentación de la sanción' for "failure to file
      penalty" (6 units);
    - FDA: 'coleccionista' (hobby collector) for an authorized collector, 'mediante el lavado'
      for disposal by flushing, 'granos de café usados' (coffee beans) for used coffee grounds,
      and an English heading left in;
    - EU: 'deniegue el embarque o la cancelación de un vuelo';
    - Ready: 'copia de seguridad' (data backup) for backup power, and 'Desclutter'.
  - Pass B's test was whether a reader would act differently about who, what, when or how much.
    Rule R1 in v2 was written because WMT professional raters mark such renderings Major.
- **Not comparable with earlier numbers.** Numbers scored against the earlier labels
  (`docs/REALDOCS.md`) do not carry over to this gold.

## 5. Limitations

- **AI annotators.**
  - Three lenses and the adjudicator are AI annotators. The lens model is not recorded in the
    annotation files. The same workflow ran the lenses and the adjudicator, so treat all four as
    one kind of annotator.
  - The lenses are highly correlated (Fleiss' kappa 0.876), and only 29 units split, so shared
    blind spots survive the vote.
  - No professional human labelled any unit of this set.
  - Calibration against humans is for the v1 texts, on other domains (social, speech, news, no
    reference). v2 was never measured.
- **Reference-aided.** Annotators saw the official Spanish. It helps with terminology (for
  example 'permiso de permanencia temporal' for parole), but it may also pull judgements
  toward "differs from the official text". The protocol says that a difference from `ref_es` is
  not an error by itself. A reference-free detector solves a harder task than the annotators
  did. The 23 set-aside pairs, where the reference itself deviates, are not in the set.
- **Segment level.** Each unit is judged alone, with two English context sentences. The
  annotators did not see the rest of the MT document. So errors that only show across
  segments are not labelled as such:
  - inconsistent terms ('DAC' in one segment and 'CAD' in the next);
  - a heading that contradicts its body;
  - a term defined one way and used another.

  OPUS-MT had no context at all, and DeepSeek had the two previous source segments.
- **One system carries the set.** 80 of the 84 Majors are OPUS-MT outputs, so the set mostly
  measures how a detector finds 2020-era NMT errors: word sense, false friends and garbled
  compounds.
  - DeepSeek-V3.1 has 4 Majors in 3 documents. In that stratum, recall and AUROC are not
    interpretable.
  - The DeepSeek stratum is useful for false flags on near-clean modern MT (716 OK units).
- **Small n, few clusters.** There are 84 Majors and 10 documents. Document-bootstrap CIs are
  wide, and with 10 clusters they are approximate.
  - In a stratum with very few positives, the bootstrap understates the uncertainty. For
    example, uniform random risks (seed 1) got AUROC 0.84 with a "CI" of 0.81-0.87 on the 4
    DeepSeek Majors.
- **Soft Major edge.** The edge of Major is soft for literal renderings that a reader can
  repair. The queue adjudicates only lone Major votes, never lone OK votes, so such units stay
  Major when 2 lenses say so. Section 3.5 puts the contestable share at about 6 to 10 of the 84.
- **High-stakes domain.** The documents are high-stakes public information. "Critical" (harm,
  financial loss or legal risk) was judged by the annotators, and no domain expert reviewed
  it.

## 6. How to score a system on this set

A "system" here is a detector that gives each (source, MT) unit a risk of Major. These rules are
fixed now, before any detector has been scored. `scripts/gold_score.py` implements them.

**1. Inputs.** Give the system only the blind input file:

```
PYTHONPATH=. uv run --frozen python -I scripts/gold_score.py --export-inputs /path/gold_inputs.jsonl
```

There is one line per unit, sorted by uid:

```json
{"uid": "g0001", "doc_id": "irs-failure-to-file-penalty", "seg_index": 30,
 "context_src": ["…", "…"], "src": "For more information about …", "mt": "Para obtener más información …"}
```

- **No reference by default.** The file has no `ref_es`, no MT-system name and no label. A
  reference-based metric may use `--with-ref`, but must report that it did. SecondRead is
  reference-free and must not get `ref_es`.
- **Context is optional.** A segment-level system may ignore `context_src`. Say whether it was
  used.
- **Off limits.** The system, its prompts and its thresholds must not use `labels.jsonl`,
  `annotations*`, `adjudication*.jsonl`, `key_systems.json`, `gold_stats.json` or the earlier
  labels. Nothing may be tuned or selected on this set.

**2. Predictions.** A JSONL file with exactly one line per gold uid (1,440 lines):

```json
{"uid": "g0001", "risk": 0.0832}
```

- `risk` is any finite number, and higher means more likely Major. The scorer refuses a file
  with a missing, duplicate or unknown uid, or a risk that is not finite. Extra fields are
  ignored, so the file may also carry `flag` or a rationale for error analysis.
- A binary system writes 1.0 or 0.0 and is scored at `--threshold NAME=0.5`.

**3. Thresholds.** Flag metrics need a threshold fixed before scoring.

- For SecondRead, take it from `secondread/thresholds.json` (picked on WMT dev) for the adapter
  being scored: the `review` line and/or the F1 line. Record the file's sha256, because the file
  changes when a new adapter ships.
- Never pick a threshold on this set. F1 at a best-on-test threshold is not a valid number
  here.
- Ranking metrics need no threshold.

**4. Which units and which labels.**

- **Headline.** All 1,440 units against `label` (the primary labelling) in
  `data/realdocs/gold/labels.jsonl`. Pin it by sha256: `d0401948bf145ec05e61765022b8de37652d2392833b44ff0db90d4f166e5273`.
- **Do not drop units.** Keep adjudicated, split, critical and multi-sentence units.
- **Also report:**
  - `--labelling preregistered`, identical to `label` in gold v1, but report it so later gold
    versions stay comparable;
  - the strata `mt=marian`, where recall is measured (80 Majors), and `mt=deepseek-v3.1`, where
    false flags are measured (716 OK units; ignore its recall and AUROC);
  - the per-document flag and true-positive counts.

**5. Metrics** (definitions from `secondread/metrics.py`, the same as for the WMT numbers in `docs/RESULTS.md`):

- **Ranking:** AUROC (a tied pair counts 1/2), PR-AUC as average precision, and recall@10%. The
  last is the share of Majors in the top 10% of units by risk, with a tie group at the cut
  counted fractionally. It is the re-read budget in the README.
- **At the fixed threshold** (flag when risk >= t): precision, recall, F1, flag rate, the false
  flag rate on OK units, and TP / FP / FN / TN.

**6. Confidence intervals.** Report a percentile 95% CI from a document-level cluster bootstrap
(`--n-boot 2000 --seed 0` by default).

- Each replicate draws 10 documents with replacement, and every unit carries its document's
  count. The two MT outputs of a segment always move together.
- Within a stratum, every system uses the same weight matrix. So when two prediction files are
  given, the report also has paired differences (A minus B) with their CI and the share of
  replicates in which A is better. Compare systems on those, not on overlapping single CIs.
- A metric that is undefined in a replicate (no Major drawn) is left out, and
  `n_boot_defined` says how many replicates remain.

**7. Command and output:**

```
PYTHONPATH=. uv run --frozen python -I scripts/gold_score.py \
  --pred secondread_v1=preds_v1.jsonl --threshold secondread_v1=<review threshold from thresholds.json> \
  --pred baseline=preds_base.jsonl --threshold baseline=0.5 --out report.json
```

`report.json` contains:

- `gold`: path, labelling and the sha256 of the labels;
- `pred_files`: path and sha256 of each prediction file;
- `bootstrap` and `thresholds`;
- `strata`: `all`, `mt=marian` and `mt=deepseek-v3.1`, each with `n`, `n_docs`, `major`,
  `major_rate`, every system's `{metric: {point, ci95, n_boot_defined}}`, the confusion counts
  and the `paired` differences;
- `per_doc`.

`--selftest` checks the scorer on synthetic oracle, constant and random predictions. It scores
no real system.

**8. Errors found later.** Error analysis will turn up gold errors. Log them in a separate
errata file and do not edit `labels.jsonl`. A corrected set is a new gold version, re-adjudicated
blind, and results are reported on both versions.

## 7. Files and reproduction

All data files are in `data/realdocs/gold/`, which is git-ignored.

| File | Content |
|---|---|
| `DOC_LIST.json` | the frozen document list, its rules and the rejected candidates |
| `units.jsonl`, `batches/` | the 1,440 blind units and the 13 annotation batches (sha256 in `MANIFEST.json`) |
| `mt/MT_META.json`, `mt/<system>/<doc>.jsonl` | the MT runs |
| `key_systems.json` | uid to `{system, segment_id}`; never shown to annotators or the adjudicator |
| `annotations/<lens>_batch_XX.jsonl` | the v2 lens annotations, verbatim (39 files) |
| `annotations_raw.jsonl`, `to_adjudicate.jsonl`, `merge_summary.json` | the merge (`scripts/gold_merge.py`) |
| `adjudication.jsonl` | the 16 adjudicated decisions, with label, severity, category, span, reason and confidence |
| `adjudicator_audit.jsonl` | the non-binding audit of section 3.5 |
| **`labels.jsonl`** | **the gold**: `{uid, doc_id, seg_index, system, label, severity, category, span_mt, reason, votes, adjudicated, segment_id, n_major, adjudication_reason, label_preregistered}` |
| `gold_stats.json` | every number on this page, plus the list of disagreements with the earlier labels |
| `calib/` | the calibration set, annotations, metrics and `RESULTS.md` |

How the label fields are filled:

- `votes` maps each lens to its severity.
- On a unit that was not adjudicated:
  - `severity` is the median of the three lens severities;
  - `category` and `span_mt` come from the first lens (MQM, then post-editor, then reader)
    whose severity equals that median;
  - `reason` is "panel k/3 Major; lens: explanation".

**Reproduce:**

```
PYTHONPATH=. uv run --frozen python -I scripts/gold_units.py      # units and batches from the frozen list
PYTHONPATH=. uv run --frozen python -I scripts/gold_merge.py      # vote, queue, agreement (no key, no scores)
PYTHONPATH=. uv run --frozen python -I scripts/gold_finalize.py   # adjudication + key join -> labels.jsonl, gold_stats.json
PYTHONPATH=. uv run --frozen python -I scripts/gold_score.py --selftest
```

The adjudication, the audit, the finalisation and the scorer made no API calls ($0).
