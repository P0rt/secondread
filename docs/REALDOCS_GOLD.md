# Real-document gold set, English to Spanish (gold v1, 2026-10-14)

This set labels every machine-translated segment of ten public English documents as **Major**
or **OK**. Each document has an official Spanish version. Major means that a Spanish-only reader
would be misled about the content, or could not get it: a wrong, missing, added or
unintelligible action, actor, object, amount, time, condition, negation or name. There are 720
source segments, and each has two MT outputs, so the set has **1,440 units**. It is a test set
for Major/OK detectors such as SecondRead. Nothing in this step scored SecondRead, Gemma or any
other detector. Section 6 says how to do it. Section 8 has the scores, computed once on 2026-10-15
under those rules.

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

## 8. Scores on the gold set (2026-10-15)

The set was scored once, by the rules of section 6. Every prediction file was written from the
blind inputs and frozen before `scripts/gold_score.py` first read `labels.jsonl` (sha256 of each
file in `runs/gold_eval/PREDICTIONS_MANIFEST.json`, 00:37 CEST). No label, threshold, prompt,
rule weight or model was changed or chosen on this set. Every number below is in `runs/gold_eval/`
(git-ignored): `report_primary.json` is the scorer's report, and `analysis.json` holds the
per-document, per-category and control counts. Copies of the two scorer reports and the manifest
are committed in `results/gold_eval/` (metrics, counts and sha256 only, no text and no labels).

**Short version.**

- **Ranking, all 1,440 units.** SecondRead v1 (the app score) gets AUROC 0.910, PR-AUC 0.520
  and recall@10% 0.619.
  - It ranks clearly better than the base model: AUROC +0.071 [+0.036, +0.119], PR-AUC +0.120.
  - It ranks worse than Gemma 4 31B direct on all three measures: AUROC −0.030 [−0.051, −0.009],
    PR-AUC −0.107, recall@10% −0.131. All three CIs exclude 0.
  - On the OPUS-MT outputs alone, which hold 80 of the 84 Majors, the v1 − Gemma gaps are small
    and within noise (AUROC −0.013 [−0.051, +0.017]). Most of Gemma's overall lead comes from
    ranking DeepSeek's near-clean output below OPUS-MT's errors.
- **The review line is far too quiet on these documents.** At 0.4984, v1 flags 22 of the 1,440
  units (1.5%). 17 of them are Major, so precision is 0.77.
  - But recall is **0.20** [0.14, 0.30]. The line was set for 0.60 on WMT en-es dev, and it
    reached 0.48 on WMT en-es test.
  - The F1 line (0.7019) flags 4 units, all Major (recall 0.05).
- **Gemma, at its review line (0.1212),** flags 12.2% of the units and finds 82% of the Majors,
  at precision 0.39. **The base model, at its review line (0.6183),** flags 7.8% and finds 49%,
  at precision 0.36.
- **The three critical units** ('lavar los parches', 'Si hace calor', the dropped "timely or
  complete"): v1 misses all three at both lines (risk 0.10 to 0.32). Gemma flags all three.
- **v1's false flags are rare and mild.** All 5 at the review line are on units the gold calls
  minor. On the 716 OK DeepSeek units it raises 2, and none at the F1 line. Gemma raises 16
  there, and the base model 25.
- **DeepSeek's 4 Majors** (counts only): v1 flags none of them at either line, Gemma flags 1 and
  the base model 0.
- **Control, the official Spanish scored as the MT** (720 segments): v1 flags 3 (0.4%) at the
  review line and none at the F1 line. Gemma flags 79 (11.0%) and 62 (8.6%).
- **Where the lines come from.** None was picked on this set.
  - v1 and base: WMT en-es dev, `secondread/thresholds.json` (sha256 `dd03044a…`).
  - Gemma: its en-ru dev run, because Gemma has no en-es dev run (`docs/RESULTS.md`, "v3 — test
    and real documents"). Both of its lines were set on en-ru, not en-es.

### 8.1 Systems and how they ran

| Name in the report | System | Line(s) |
|---|---|---|
| `v1_app_review`, `v1_app_f1` | **SecondRead v1, scored as the app scores it.** `secondread.agent.ReviewAgent.review` (the backend pipeline) ran in-process against the shipped llama-server on `:9931`: Q4_K_M plus LoRA v1 (sha256 `5337b2b7…`) at scale 1. The risk is logit(model risk) plus the train-calibrated rule weights (`secondread/rule_calibration.json`, `fbc71fcd…`), unrounded. That is the number the backend compares with the threshold | review 0.4984, F1 0.7019 (en-es dev) |
| `v1_model` | the model risk of the same run, p(Major) / (p(Major) + p(OK)), without the rules | ranking only |
| `base_app_review` | the same pipeline at `lora_scale` 0, which is the base Qwen3.5-4B Q4_K_M. It ran on its own llama-server on `:9980` (`-cram 0`), which was stopped afterwards | review 0.6183 (base, en-es dev) |
| `gemma_review`, `gemma_f1` | `gemma-4-31B-it` on DigitalOcean with the SecondRead classifier prompt: the direct method of `scripts/baseline_gemma.py` (temperature 0, top-10 logprobs) | review 0.1212, F1 0.9714 (en-ru dev) |

- **Inputs.** Every system read only `runs/gold_eval/gold_inputs.jsonl`, from `--export-inputs`.
  It has no `ref_es`, no system name and no label. No system used `context_src`: all of them
  score one segment at a time.
- **Checks before scoring.** None of these needs a label.
  - For v1 and base, the backend's own flag equals `risk >= threshold` on every unit. The
    recomputed fused risk rounds to the value the backend returns, also on every unit.
  - No model call failed and no Gemma reply was unparsed. Gemma's top-10 lists held both labels
    in 1,390 units, only Major in 42 and only OK in 8.
  - 12 en-es dev rows went through the same two servers. They match the stored dev scores
    (`results/local_{v1,base}_enes_dev.jsonl`) to within 0.0002 for v1 and 0.0008 for base.
  - 387 gold units have the same document, source and MT as a row of the earlier snapshot
    `data/realdocs/scored.jsonl`. Their v1 risks match it to within 0.0011.
  - Gemma's exact F1 line is 0.97144 (`runs/ship_test/test.json`). No Gemma risk lies between
    0.9714 and that value, so the two give the same flags.
- **Labelling.** `--labelling preregistered` gives a report identical to the primary one, in
  every stratum and every document (`report_preregistered.json`). This is expected in gold v1.
- **Rule checks.** 16 units have a finding: negation_added 11, negation_dropped 3 and
  number_changed 2. None of the 16 is Major, and the weights change no flag at either line. They
  cost 0.001 AUROC (v1 app − v1 model, [−0.001, −0.000]).
- **Spend.**
  - DigitalOcean: $0.029 at list price, under the $0.15 cap. That is 1,059 new calls for the gold
    units and 680 for the control. 421 replies came from byte-identical requests already in the
    cache: 390 sent for `docs/REALDOCS.md` (OPUS-MT and NLLB outputs), and 31 control segments
    whose official Spanish equals a gold unit's MT, scored minutes before.
  - Tinker: $0.
  - `:9931` served the v1 calls. It was not restarted.

### 8.2 Overall and by MT system

There were 2,000 document-bootstrap replicates (seed 0). recall@10% reads the riskiest 10% of
the stratum: 144 units overall, and 72 within one MT system. The DeepSeek stratum has 4 Majors,
so its ranking metrics are not read as findings and are left out here (they are in
`report_primary.json`); section 8.6 gives its counts.

| Stratum | System | AUROC [95% CI] | PR-AUC [95% CI] | recall@10% [95% CI] |
|---|---|---|---|---|
| All (1,440 units, 84 Major) | **v1, app** | 0.910 [0.884, 0.928] | 0.520 [0.393, 0.631] | 0.619 [0.536, 0.676] |
|  | v1, model only | 0.911 [0.885, 0.928] | 0.520 [0.394, 0.632] | 0.619 [0.537, 0.677] |
|  | base, app | 0.839 [0.770, 0.885] | 0.400 [0.254, 0.523] | 0.571 [0.459, 0.671] |
|  | Gemma direct | 0.940 [0.911, 0.957] | 0.627 [0.564, 0.681] | 0.750 [0.645, 0.833] |
| OPUS-MT (720 units, 80 Major) | **v1, app** | 0.906 [0.875, 0.925] | 0.639 [0.509, 0.742] | 0.537 [0.423, 0.642] |
|  | v1, model only | 0.906 [0.876, 0.925] | 0.639 [0.510, 0.742] | 0.537 [0.423, 0.642] |
|  | base, app | 0.827 [0.758, 0.877] | 0.488 [0.326, 0.623] | 0.450 [0.336, 0.543] |
|  | Gemma direct | 0.918 [0.885, 0.941] | 0.655 [0.605, 0.701] | 0.613 [0.420, 0.716] |

At the lines. A threshold is applied as risk >= t. "False flags on OK" is the share of the OK
units that are flagged, with their count in brackets.

| Stratum | System | Line | Threshold | Flags (Major) | Recall [95% CI] | Precision [95% CI] | Flag rate [95% CI] | False flags on OK [95% CI] |
|---|---|---|---:|---:|---|---|---|---|
| All (1,440 units, 84 Major) | **v1, app** | review | 0.4984 | 22 (17) | 0.202 [0.139, 0.296] | 0.773 [0.611, 0.905] | 1.5% [1.0, 2.3] | 0.4% [0.1, 0.6] (5) |
|  | v1, app | F1 | 0.7019 | 4 (4) | 0.048 [0.000, 0.086] | 4 of 4 ² | 0.3% [0.0, 0.6] | 0.0% [0.0, 0.0] (0) |
|  | base, app | review | 0.6183 | 113 (41) | 0.488 [0.350, 0.640] | 0.363 [0.235, 0.472] | 7.8% [6.0, 10.7] | 5.3% [4.5, 6.5] (72) |
|  | Gemma direct | review ¹ | 0.1212 | 175 (69) | 0.821 [0.712, 0.894] | 0.394 [0.282, 0.484] | 12.2% [8.6, 16.7] | 7.8% [5.8, 10.6] (106) |
|  | Gemma direct | F1 ¹ | 0.9714 | 145 (63) | 0.750 [0.656, 0.824] | 0.434 [0.333, 0.515] | 10.1% [6.6, 14.6] | 6.0% [4.1, 8.6] (82) |
| OPUS-MT (720 units, 80 Major) | **v1, app** | review | 0.4984 | 20 (17) | 0.212 [0.148, 0.310] | 0.850 [0.688, 1.000] | 2.8% [1.7, 4.3] | 0.5% [0.0, 1.1] (3) |
|  | v1, app | F1 | 0.7019 | 4 (4) | 0.050 [0.000, 0.090] | 4 of 4 ² | 0.6% [0.0, 1.3] | 0.0% [0.0, 0.0] (0) |
|  | base, app | review | 0.6183 | 88 (41) | 0.512 [0.362, 0.664] | 0.466 [0.317, 0.582] | 12.2% [8.9, 17.2] | 7.3% [6.2, 9.1] (47) |
|  | Gemma direct | review ¹ | 0.1212 | 158 (68) | 0.850 [0.753, 0.917] | 0.430 [0.321, 0.514] | 21.9% [15.5, 30.3] | 14.1% [10.6, 19.1] (90) |
|  | Gemma direct | F1 ¹ | 0.9714 | 136 (62) | 0.775 [0.685, 0.839] | 0.456 [0.372, 0.519] | 18.9% [12.4, 27.6] | 11.6% [8.0, 17.0] (74) |
| DeepSeek-V3.1 (720 units, 4 Major) | **v1, app** | review | 0.4984 | 2 (0) | 0 of 4 | 0 of 2 | 0.3% [0.0, 0.7] | 0.3% [0.0, 0.7] (2) |
|  | v1, app | F1 | 0.7019 | 0 (0) | 0 of 4 | – | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] (0) |
|  | base, app | review | 0.6183 | 25 (0) | 0 of 4 | 0 of 25 | 3.5% [2.6, 4.5] | 3.5% [2.7, 4.6] (25) |
|  | Gemma direct | review ¹ | 0.1212 | 17 (1) | 1 of 4 | 1 of 17 | 2.4% [1.1, 4.1] | 2.2% [1.0, 4.0] (16) |
|  | Gemma direct | F1 ¹ | 0.9714 | 9 (1) | 1 of 4 | 1 of 9 | 1.2% [0.1, 3.0] | 1.1% [0.1, 2.9] (8) |

1. **Gemma's lines** come from its en-ru dev run (2,972 rows). F1 is the F1-optimal threshold,
   and review is the highest threshold with dev recall ≥ 0.60. Both were set on en-ru, not en-es.
2. **The F1-line precision of v1 has no usable CI.** v1 raises 4 flags at that line, in 3
   documents. A replicate that draws none of those documents has no flag, and
   `metrics.prf_at` then sets precision to 0. So the CI is [0.000, 1.000], which says nothing.
   The point value is 4 of 4, and the paired precision difference at that line is left out for
   the same reason.

### 8.3 Paired differences

The deltas use the same replicates as section 8.2. At the lines, each system uses its own
threshold.

| Stratum | A − B | ΔAUROC | ΔPR-AUC | Δrecall@10% |
|---|---|---|---|---|
| All | v1 app − Gemma | −0.030 [−0.051, −0.009] | −0.107 [−0.232, −0.008] | −0.131 [−0.194, −0.043] |
|  | v1 app − base app | +0.071 [+0.036, +0.119] | +0.120 [+0.064, +0.194] | +0.048 [−0.030, +0.136] |
|  | v1 app − v1 model | −0.001 [−0.001, −0.000] | −0.001 [−0.001, −0.000] | +0.000 [+0.000, +0.000] |
| OPUS-MT | v1 app − Gemma | −0.013 [−0.051, +0.017] | −0.017 [−0.157, +0.085] | −0.075 [−0.155, +0.067] |
|  | v1 app − base app | +0.078 [+0.042, +0.122] | +0.150 [+0.095, +0.222] | +0.087 [+0.047, +0.170] |
|  | v1 app − v1 model | −0.001 [−0.001, −0.000] | −0.001 [−0.001, −0.000] | +0.000 [+0.000, +0.000] |

| Stratum | A − B | Line | ΔRecall | ΔPrecision | ΔFlag rate (points) | ΔFalse flags on OK (points) | ΔF1 |
|---|---|---|---|---|---|---|---|
| All | v1 app − Gemma | review | −0.619 [−0.731, −0.444] | +0.378 [+0.236, +0.499] | −10.6 [−14.7, −7.4] | −7.4 [−10.1, −5.5] | −0.212 [−0.349, −0.008] |
|  | v1 app − Gemma | F1 | −0.702 [−0.771, −0.617] | – ² | −9.8 [−14.0, −6.5] | −6.0 [−8.6, −4.1] | −0.459 [−0.560, −0.357] |
|  | v1 app − base app | review | −0.286 [−0.449, −0.117] | +0.410 [+0.282, +0.560] | −6.3 [−8.7, −4.8] | −4.9 [−6.1, −4.1] | −0.095 [−0.242, +0.088] |
| OPUS-MT | v1 app − Gemma | review | −0.637 [−0.748, −0.466] | +0.420 [+0.252, +0.594] | −19.2 [−26.5, −13.4] | −13.6 [−18.3, −10.2] | −0.231 [−0.369, −0.013] |
|  | v1 app − Gemma | F1 | −0.725 [−0.791, −0.643] | – ² | −18.3 [−26.4, −12.3] | −11.6 [−17.0, −8.0] | −0.479 [−0.573, −0.383] |
|  | v1 app − base app | review | −0.300 [−0.469, −0.123] | +0.384 [+0.236, +0.558] | −9.4 [−13.6, −6.8] | −6.9 [−8.4, −5.8] | −0.148 [−0.305, +0.047] |
| DeepSeek-V3.1 | v1 app − Gemma | review | – | – | −2.1 [−3.7, −0.9] | −2.0 [−3.6, −0.8] | – |
|  | v1 app − Gemma | F1 | – | – | −1.2 [−3.0, −0.1] | −1.1 [−2.9, −0.1] | – |
|  | v1 app − base app | review | – | – | −3.2 [−4.4, −2.3] | −3.2 [−4.4, −2.3] | – |

- **v1 against the base model.**
  - v1 ranks better on AUROC and PR-AUC. On recall@10% it is better only on OPUS-MT:
    +0.087 [+0.047, +0.170].
  - At their review lines the base model finds more Majors, +0.29 in recall, at less than half
    of v1's precision. The F1 difference includes 0.
- **v1 against Gemma, ranking.** Gemma is ahead overall. On OPUS-MT the difference is within
  noise.
  - AUROC by pair type (`analysis.json`). Ranking an OPUS-MT Major above an OPUS-MT OK unit: v1
    0.906, Gemma 0.918. Ranking an OPUS-MT Major above a DeepSeek OK unit: v1 0.915, Gemma 0.967.
  - Gemma's lead comes from the second kind of pair: it puts DeepSeek's clean output further
    below OPUS-MT's errors.
- **v1 against Gemma, at the lines.**
  - v1's recall is 0.62 lower at the review line, its precision 0.38 higher, and it flags 10.6
    points fewer units. F1 is lower: −0.212 [−0.349, −0.008].
  - This reverses WMT en-es test. There v1 tied Gemma on AUROC (+0.006) and led it on PR-AUC
    (+0.203).

### 8.4 Per document

These are point values only, because a document-cluster bootstrap gives no CI inside one
document. NIDDK and MedlinePlus have 2 Majors each, so their AUROC and recall@10% rest on two
units. Both MT systems count in each row.

| Document | Units | Major (OPUS-MT / DeepSeek) | v1 app: AUROC / PR-AUC / recall@10% | Gemma: the same three | base app: the same three |
|---|---:|---|---|---|---|
| `irs-failure-to-file-penalty` | 104 | 14 (14 / 0) | 0.95 / 0.82 / 0.57 | 0.95 / 0.69 / 0.57 | 0.93 / 0.68 / 0.43 |
| `fda-medicine-disposal` | 78 | 10 (10 / 0) | 0.86 / 0.51 / 0.30 | 0.89 / 0.66 / 0.50 | 0.78 / 0.44 / 0.30 |
| `eu-air-passenger-rights` | 226 | 5 (4 / 1) | 0.83 / 0.14 / 0.60 | 0.95 / 0.68 / 0.80 | 0.46 / 0.04 / 0.20 |
| `niddk-hypoglycemia` | 182 | 2 (2 / 0) | 0.81 / 0.51 / 0.50 | 0.71 / 0.51 / 0.50 | 0.68 / 0.04 / 0.50 |
| `ready-hurricanes` | 106 | 6 (6 / 0) | 0.93 / 0.65 / 0.50 | 0.91 / 0.58 / 0.67 | 0.79 / 0.43 / 0.67 |
| `uscis-common-scams` | 176 | 14 (12 / 2) | 0.93 / 0.70 / 0.64 | 0.96 / 0.73 / 0.71 | 0.86 / 0.61 / 0.64 |
| `ssa-disability-qualify` | 162 | 6 (5 / 1) | 0.90 / 0.68 / 0.67 | 0.92 / 0.49 / 0.50 | 0.85 / 0.43 / 0.67 |
| `cdc-carbon-monoxide` | 86 | 12 (12 / 0) | 0.90 / 0.65 / 0.50 | 0.96 / 0.76 / 0.50 | 0.79 / 0.44 / 0.42 |
| `ready-home-fires` | 166 | 13 (13 / 0) | 0.92 / 0.57 / 0.62 | 0.96 / 0.71 / 0.77 | 0.90 / 0.50 / 0.54 |
| `medlineplus-pt-inr` | 154 | 2 (2 / 0) | 0.79 / 0.27 / 0.50 | 0.73 / 0.51 / 0.50 | 0.97 / 0.60 / 1.00 |

Flags / caught Majors / false flags on OK units, at each line:

| Document | Major | v1 review | v1 F1 | base review | Gemma review | Gemma F1 |
|---|---:|---|---|---|---|---|
| `irs-failure-to-file-penalty` | 14 | 2 / 2 / 0.0% | 2 / 2 / 0.0% | 19 / 12 / 7.8% | 25 / 13 / 13.3% | 24 / 13 / 12.2% |
| `fda-medicine-disposal` | 10 | 3 / 2 / 1.5% | 1 / 1 / 0.0% | 6 / 2 / 5.9% | 19 / 7 / 17.6% | 17 / 7 / 14.7% |
| `eu-air-passenger-rights` | 5 | 1 / 0 / 0.5% | 0 / 0 / 0.0% | 9 / 1 / 3.6% | 12 / 3 / 4.1% | 8 / 3 / 2.3% |
| `niddk-hypoglycemia` | 2 | 1 / 1 / 0.0% | 0 / 0 / 0.0% | 9 / 0 / 5.0% | 7 / 1 / 3.3% | 5 / 1 / 2.2% |
| `ready-hurricanes` | 6 | 3 / 3 / 0.0% | 0 / 0 / 0.0% | 10 / 3 / 7.0% | 8 / 4 / 4.0% | 7 / 3 / 4.0% |
| `uscis-common-scams` | 14 | 2 / 2 / 0.0% | 1 / 1 / 0.0% | 15 / 7 / 4.9% | 24 / 12 / 7.4% | 21 / 10 / 6.8% |
| `ssa-disability-qualify` | 6 | 4 / 3 / 0.6% | 0 / 0 / 0.0% | 12 / 4 / 5.1% | 23 / 4 / 12.2% | 20 / 4 / 10.3% |
| `cdc-carbon-monoxide` | 12 | 3 / 2 / 1.4% | 0 / 0 / 0.0% | 14 / 6 / 10.8% | 21 / 12 / 12.2% | 16 / 10 / 8.1% |
| `ready-home-fires` | 13 | 2 / 2 / 0.0% | 0 / 0 / 0.0% | 10 / 5 / 3.3% | 24 / 12 / 7.8% | 21 / 11 / 6.5% |
| `medlineplus-pt-inr` | 2 | 1 / 0 / 0.7% | 0 / 0 / 0.0% | 9 / 1 / 5.3% | 12 / 1 / 7.2% | 6 / 1 / 3.3% |
| **Total** | **84** | **22 / 17 / 0.4%** | **4 / 4 / 0.0%** | **113 / 41 / 5.3%** | **175 / 69 / 7.8%** | **145 / 63 / 6.0%** |

- **v1's flags.** It flags at most 4 units in any document at the review line, and at the F1
  line it flags nothing in 7 of the 10 documents.
- **The three documents of `docs/REALDOCS.md`** (IRS, FDA, EU 261).
  - v1's review-line flags on OPUS-MT are the same 6 segments as there: IRS-015, IRS-022, FDA-005,
    FDA-008, FDA-022 and EU-073.
  - Under the gold, 4 of the 6 are Major, against 3 before. IRS-015 garbles both penalty names
    ('la falta de presentación de la pena'). The earlier pass called it minor, and the gold calls
    it Major (section 4.4).

### 8.5 Which Majors v1 catches and misses

Categories are those of `labels.jsonl`. "Riskiest 10%" means the 144 riskiest of all 1,440 units
by each system's risk.

| Gold category | Majors | v1 review | v1 F1 | v1 riskiest 10% | base review | Gemma review | Gemma F1 | Gemma riskiest 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accuracy/mistranslation | 70 | 16 | 4 | 44 | 35 | 59 | 55 | 55 |
| accuracy/omission | 7 | 0 | 0 | 5 | 5 | 5 | 4 | 4 |
| terminology | 5 | 0 | 0 | 1 | 1 | 3 | 2 | 2 |
| accuracy/untranslated | 2 | 1 | 0 | 2 | 0 | 2 | 2 | 2 |
| **All Majors** | **84** | **17** | **4** | **52** | **41** | **69** | **63** | **63** |
| of which critical | 3 | 0 | 0 | 1 | 1 | 3 | 3 | 3 |

- **Caught at the review line (17).** Most are garbled names and compounds whose Spanish reads
  as nonsense or as something else:
  - 'Cada pareja' for a business partner;
  - 'Día Nacional de Retraso de las Drogas Recluidas';
  - 'lista de color';
  - 'hijastro, nieto o hijastro';
  - 'Desclutter';
  - 'cada primavera y caída';
  - the IRS-015 penalty names.
- **Missed (67).**
  - **A wrong sense in fluent Spanish:** 'lavar los parches' (risk 0.159), 'Si hace calor'
    (0.101), 'granos de café usados' (0.137), 'dispositivos de carga de copia de seguridad'
    (0.111), 'en forma impresa' (0.054), 'su bolso' (0.035), 'las utilidades' (0.045) and
    'prestar juramentos' (0.044).
  - **Immigration parole as 'libertad condicional':** 0.146 and 0.385 in OPUS-MT, 0.327 and
    0.137 in DeepSeek.
  - **The IRS false friend 'devolución'** for the tax return (0.08 to 0.28), and most of the
    garbled IRS penalty names (0.12 to 0.48).
  - **All 7 omissions** (0.02 to 0.495). Even so, 5 of them are in v1's riskiest 10%.
- **What v1's risks look like here.**
  - On the Majors, the median risk is 0.26. 21 Majors sit between 0.30 and the review line, and
    46 sit below 0.30.
  - On the OK units, the median is 0.024 and the 99th percentile 0.40.
  - The riskiest 10% of all units starts at 0.165.
  - So v1 ranks the Majors high (52 of the 84 are in its riskiest 10%). But its scores on these
    documents are low compared with a line calibrated on WMT dev.
- **Overlap with Gemma at the review lines.** Gemma flags 53 Majors that v1 does not. v1 flags 1
  that Gemma does not: g0883 (SSA, 'haya funcionado nunca', Gemma risk 0.011).
- **The 8 Majors that the non-binding audit leans OK on** (section 3.5): v1 flags none of them,
  and Gemma flags 5 at its review line. So these contestable labels do not explain v1's low
  recall. The labels stay as they are.

### 8.6 DeepSeek-V3.1 stratum: counts, not rates

Rank 1 is the riskiest of the 720 DeepSeek units.

| Unit | Error | v1 app risk (rank) | Gemma risk (rank) | base app risk (rank) |
|---|---|---|---|---|
| g0212 | 'al menos un día después' for "the day after" (EU 261) | 0.103 (80) | 0.000 (154) | 0.244 (392) |
| g0724 | 'libertad condicional' for immigration parole (USCIS) | 0.327 (14) | 1.000 (5) | 0.599 (38) |
| g0768 | 'período de libertad condicional', the same (USCIS) | 0.137 (55) | 0.101 (18) | 0.602 (36) |
| g0918 | a 10-year window shifted by one year (SSA) | 0.087 (99) | 0.001 (64) | 0.517 (73) |

- **Majors.** No system flags g0212, g0768 or g0918 at any line. Gemma flags g0724 at both of its
  lines. v1 ranks g0724 14th of 720, but its risk (0.327) is under the line.
- **False flags on the 716 OK units.**
  - v1: 2 at the review line, g1308 (MedlinePlus) and g1024 (SSA), both minor errors in the gold.
    None at the F1 line.
  - Gemma: 16 at the review line (10 on minor errors, 6 on units with no error) and 8 at the F1
    line.
  - Base: 25 (12 on minor errors, 13 on units with no error).
  - The rates and their CIs are in the table in section 8.2.

### 8.7 Control: the official Spanish scored as the MT

Each of the 720 segments was scored again, with the publisher's Spanish in place of the MT. These
pairs passed the hand check of section 1, where the 23 pairs that are not equivalent were set
aside. So they are human translations judged equivalent. Nobody labelled them Major or OK, and a
flag here is most likely a false alarm.

| Text scored | Units | v1 review | v1 F1 | Gemma review | Gemma F1 |
|---|---:|---|---|---|---|
| Official Spanish (`ref_es`) | 720 | 3 (0.4%) | 0 | 79 (11.0%) | 62 (8.6%) |
| For comparison: the OK MT units, false flags | 1,356 | 5 (0.4%) | 0 | 106 (7.8%) | 82 (6.0%) |

- **v1's 3 flags.**
  - NIDDK segments 22 and 31 (0.539 and 0.608): "la persona" for "you", and 'hipoglucemia
    inconsciente'. They are NIDDK-023 and NIDDK-032, the two flagged in `docs/REALDOCS.md`.
  - CDC segment 31 (0.547): "Nunca queme carbón adentro" for "Never burn charcoal indoors".
  - All three read as correct translations.
- **Gemma's flags** cluster in EU 261 (16), SSA (13) and NIDDK (11). Gemma flags the human
  Spanish more often than the OK machine output, probably because the official translations are
  freer than MT. This was not checked unit by unit.
- **The base model** was not run on the control.

### 8.8 What this means

- **v1's ranking transfers to these documents, and its flag line does not.**
  - v1's AUROC here (0.910) is higher than on WMT en-es test (0.813), and 62% of the Majors are
    in its riskiest 10%.
  - The review line, picked on 12 WMT en-es dev documents, flags 1.5% of these units and finds
    20% of the Majors.
  - A post-editor who worked through the default flag list on documents like these would miss 4
    in 5 of the Majors. The riskiest-10% view would find 3 in 5.
- **The line stays where it is.** Section 6 forbids picking a threshold on this set. A new line
  needs new dev data, for example held-out real public documents, and it must be reported as
  informed by these results.
- **Gemma 4 31B is the stronger detector on this set.** On WMT en-es test, by contrast, v1 tied
  it on AUROC and led it on PR-AUC. Gemma's lines were set on en-ru, and here they still find 75%
  to 82% of the Majors, with 10% to 12% of the units flagged.
- **Caveats for this comparison.**
  - **The gold comes from AI annotators** who read the same pair plus the official Spanish. An
    LLM judge such as Gemma may share their idea of Major more closely than a 4B classifier
    trained on one-rater WMT MQM labels does. This set cannot separate that effect from
    detection skill.
  - **Few Majors, few clusters.** 80 of the 84 Majors are OPUS-MT errors, in 10 documents.
    CIs from 10 clusters are approximate, and the DeepSeek stratum gives counts only.
  - **The F1-line precision CI** of v1 says nothing (footnote 2 of section 8.2).

### 8.9 Files and reproduction

These files are in `runs/gold_eval/`, which is git-ignored. `report_primary.json`,
`report_preregistered.json` and `PREDICTIONS_MANIFEST.json` are also committed, byte for byte, in
`results/gold_eval/`.

| File | Content |
|---|---|
| `gold_inputs.jsonl`, `gold_inputs_with_ref.jsonl` | the blind inputs. The `--with-ref` export is used only for the control |
| `preds_v1_app.jsonl`, `preds_v1_model.jsonl`, `preds_base_app.jsonl`, `preds_gemma_direct.jsonl` | one line per uid with `risk`. The v1 and base files also hold the backend's rounded risk and flag, the model risk, and the rule codes and weights. The Gemma file holds the Gemma status and cache key |
| `preds_control_ref_es_v1_app.jsonl`, `preds_control_ref_es_gemma.jsonl` | the control, one line per source segment |
| `meta_*.json` | for each run: server health, the thresholds and their source, timings, the consistency checks and the Gemma spend |
| `PREDICTIONS_MANIFEST.json` | the sha256 of every prediction file and input, written before scoring |
| `report_primary.json`, `report_preregistered.json`, `score_stdout.txt` | the scorer's output |
| `analysis.json`, `analysis_stdout.txt` | per document, per category and severity, the unit lists and the control |

New scripts:

- `scripts/gold_predict.py` writes the predictions. It reads only the blind inputs, and it refuses
  to run with `SECONDREAD_THRESHOLD` or `SECONDREAD_OPERATING_POINT` set.
- `scripts/gold_analyze.py` writes the error analysis. It refuses prediction files that changed
  after they were frozen.

```sh
py() { HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen python "$@"; }
D=runs/gold_eval
py -I scripts/gold_score.py --export-inputs $D/gold_inputs.jsonl
py -I scripts/gold_score.py --export-inputs $D/gold_inputs_with_ref.jsonl --with-ref
# v1 on the shipped :9931; base on its own server, at LoRA scale 0 per request
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --lora models/secondread-lora.gguf --lora-init-without-apply \
  -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9980 &
py scripts/gold_predict.py v1 --llama-url http://127.0.0.1:9931
py scripts/gold_predict.py control --inputs $D/gold_inputs_with_ref.jsonl --llama-url http://127.0.0.1:9931
py scripts/gold_predict.py base --llama-url http://127.0.0.1:9980
kill %1
set -a; . ./.env; set +a
py scripts/gold_predict.py gemma --cap 0.15            # cached: a re-run makes no call
py scripts/gold_predict.py gemma-control --inputs $D/gold_inputs_with_ref.jsonl --cap 0.13
py scripts/gold_predict.py freeze                      # before anything reads labels.jsonl (this run wrote
                                                       # the manifest with the same code, inline, at 00:37)
for L in primary preregistered; do
  py -I scripts/gold_score.py --labelling $L --out $D/report_$L.json \
    --pred v1_app_review=$D/preds_v1_app.jsonl --threshold v1_app_review=0.4984 \
    --pred v1_app_f1=$D/preds_v1_app.jsonl --threshold v1_app_f1=0.7019 \
    --pred v1_model=$D/preds_v1_model.jsonl \
    --pred base_app_review=$D/preds_base_app.jsonl --threshold base_app_review=0.6183 \
    --pred gemma_review=$D/preds_gemma_direct.jsonl --threshold gemma_review=0.1212 \
    --pred gemma_f1=$D/preds_gemma_direct.jsonl --threshold gemma_f1=0.9714
done
py -I scripts/gold_analyze.py
cp $D/report_primary.json $D/report_preregistered.json $D/PREDICTIONS_MANIFEST.json results/gold_eval/
```
