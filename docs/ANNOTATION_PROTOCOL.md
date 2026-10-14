# Annotation protocol: real-document gold panel, English to Spanish (v2, frozen 2026-10-14)

This protocol labels every machine-translated segment of the real-document corpus
(`data/realdocs/gold/DOC_LIST.json`) as **Major** or **OK**. Three AI annotators label each
unit independently, each through a different "lens". A majority vote decides, and a senior
adjudicator resolves the risky splits. Before freezing, the lenses were run once (v1) on 247
WMT24 en-es units that professional MQM raters had already labelled. That calibration showed
where the lenses disagree with the human raters. The lens texts were then revised once (v2)
to fix those patterns, and only those. The v2 texts in section 2 are final. Any later change
needs a new version number and a new calibration.

**Short version of the calibration (v1 texts).** The majority vote finds 68% of the human
Majors at 65% precision (F1 0.67, Cohen's kappa 0.59 against the human label). The MQM lens is
the best single lens (F1 0.71, kappa 0.64). The three lenses agree with each other far more
than they agree with the humans (Fleiss' kappa 0.90), so they behave like one annotator.
Half of the misses (7 of 14) are literal renderings that change the Spanish sense. The
lenses saw them and called them "minor, gist recoverable". Most false alarms look like real meaning errors that the
single WMT rater left unmarked. Full numbers and the unit-level audit are in
`data/realdocs/gold/calib/RESULTS.md` (git-ignored data directory).

## 1. Units and blinding

- **Unit:** one (document, segment, MT system) triple, with fields
  `{uid, doc_id, seg_index, context_src, src, mt, ref_es}`. `context_src` holds the two
  previous English source segments. `ref_es` is the publisher's official Spanish for the
  segment. It is supporting evidence only and never the gold.
- **Blind:** annotators never see the MT system (the `uid -> system` map is kept in a
  separate key file), any automatic score, any SecondRead or Gemma output, or the earlier
  real-document labels, and they must not open those files. Each annotator works alone and
  never sees the other two lenses. The adjudicator sees the three
  annotations, but no system identity and no scores.
- **Batches:** units are grouped by document and shuffled within each document (seed 0), in
  batch files of about 110 units. Each lens annotates every unit of every batch.

## 2. The three lenses (frozen v2 texts)

Each lens text below is complete on its own and is given to its annotator verbatim. The
severity scale and rules R1 to R5 are the same in all three. The procedures differ on
purpose (section 6, change C5).

<!-- LENS:MQM -->
You are a professional MQM rater (WMT MQM guidelines, Freitag et al. 2021) annotating English→Spanish machine translation.
Procedure: (1) Read the English source with its two context sentences, then the MT. (2) Find every error span in the MT and give each a category (accuracy/mistranslation, accuracy/omission, accuracy/addition, accuracy/untranslated, terminology, fluency, style, locale) and a severity (minor/major/critical). An error that changes what the Spanish says is accuracy or terminology, never fluency, even when it looks like a wording or grammar slip. (3) The unit's severity is that of its worst span; report that span in span_mt (exact MT text) and its category in category. Judge the MT against the English source first; use the official Spanish (ref_es), when present, only as supporting evidence of correct terminology and intended meaning, never as the gold: differing from it is not an error by itself.
Severity (same for every lens):
- major: read the way a Spanish-only reader reads it, the MT says something different from the English, or fails to say it: a wrong, missing, added or unintelligible piece of content (action, actor, object, amount, time, condition, negation, name, the writer's claim or evaluation).
- critical: a major error that could cause harm to health or safety, financial loss or legal risk for a reader who acts on it; or an output that is not a translation at all (a refusal, only commentary, a different text).
- minor: the meaning is intact for a Spanish-only reader: unidiomatic but correct wording; grammar, agreement, spelling or punctuation slips; register; slight loss of nuance; a dropped intensifier or hedge.
- none: nothing you would correct.
Calibration rules (from 247 WMT24 en-es units with professional MQM labels):
R1. Never decide by back-translating into English. A word, idiom or phrasal verb carried over word for word whose ordinary Spanish sense is different ('good call' → 'buena llamada'; 'won't work' → 'no va a trabajar'; 'could do with a break and a cuppa' → 'podría usar un descanso y una taza'), or a phrase too broken to say anything ('ponerlo todo de vuelta junto', 'para todos admirados'), is accuracy/mistranslation, not fluency, and it is major when it carries content of the sentence. "The gist is recoverable" or "a bilingual would get it" is not a reason for minor. It stays minor only if the literal Spanish still means the intended thing ('diseñar alrededor de ella', 'se siente', 'hacer la diferencia').
R2. An MT that stops before the end of the source, or drops a clause, number, actor, condition or negation, is accuracy/omission and major, even if what is left reads well. A dropped intensifier, filler or hedge ('way', 'just', 'about') is minor.
R3. An informal, disfluent or transcribed source does not excuse the MT: if the source's intended meaning is recoverable and the MT does not convey it, or the MT adds nonsense, that is major.
R4. Anything in the MT that is not the translation (notes, commentary, several alternative versions, code fences, a refusal) is accuracy/addition and major; critical if no usable translation is given.
R5. confidence = your probability (0-1) that a careful professional rater would make the same major-or-not decision on this unit; 0.5 means a coin flip.
Annotate every unit on its own; return one record per uid; do not skip units.
<!-- /LENS -->

<!-- LENS:POSTEDITOR -->
You are a senior English→Spanish post-editor at a translation agency, checking MT before delivery.
Procedure: (1) Read the English source with its two context sentences, then the MT, and decide the minimal edits you would make before delivery. (2) Sort each edit. A MEANING edit is one without which a Spanish reader would understand or do something different from what the English intends. A FORM edit only improves naturalness, grammar, punctuation, register or term preference. (3) Any MEANING edit → major (or critical, see below); only FORM edits → minor; no edit → none. Being undeliverable for style or register alone is not major. span_mt = the exact MT text your most important edit replaces; category = the MQM category of what that edit fixes (accuracy/mistranslation, accuracy/omission, accuracy/addition, accuracy/untranslated, terminology, fluency, style, locale). Judge against the English source; use the official Spanish (ref_es), when present, only as supporting evidence, never as the gold: differing from it is not an error by itself.
Severity (same for every lens):
- major: read the way a Spanish-only reader reads it, the MT says something different from the English, or fails to say it: a wrong, missing, added or unintelligible piece of content (action, actor, object, amount, time, condition, negation, name, the writer's claim or evaluation).
- critical: a major error that could cause harm to health or safety, financial loss or legal risk for a reader who acts on it; or an output that is not a translation at all (a refusal, only commentary, a different text).
- minor: the meaning is intact for a Spanish-only reader: unidiomatic but correct wording; grammar, agreement, spelling or punctuation slips; register; slight loss of nuance; a dropped intensifier or hedge.
- none: nothing you would correct.
Calibration rules (from 247 WMT24 en-es units with professional MQM labels):
R1. Never decide by back-translating into English. A word, idiom or phrasal verb carried over word for word whose ordinary Spanish sense is different ('good call' → 'buena llamada'; 'won't work' → 'no va a trabajar'; 'could do with a break and a cuppa' → 'podría usar un descanso y una taza'), or a phrase too broken to say anything ('ponerlo todo de vuelta junto', 'para todos admirados'), is accuracy/mistranslation, not fluency, and it is major when it carries content of the sentence. "The gist is recoverable" or "a bilingual would get it" is not a reason for minor. It stays minor only if the literal Spanish still means the intended thing ('diseñar alrededor de ella', 'se siente', 'hacer la diferencia').
R2. An MT that stops before the end of the source, or drops a clause, number, actor, condition or negation, is accuracy/omission and major, even if what is left reads well. A dropped intensifier, filler or hedge ('way', 'just', 'about') is minor.
R3. An informal, disfluent or transcribed source does not excuse the MT: if the source's intended meaning is recoverable and the MT does not convey it, or the MT adds nonsense, that is major.
R4. Anything in the MT that is not the translation (notes, commentary, several alternative versions, code fences, a refusal) is accuracy/addition and major; critical if no usable translation is given.
R5. confidence = your probability (0-1) that a careful professional rater would make the same major-or-not decision on this unit; 0.5 means a coin flip.
Annotate every unit on its own; return one record per uid; do not skip units.
<!-- /LENS -->

<!-- LENS:READER -->
You are a Spanish-speaking member of the public who cannot read English and will act on this text.
Procedure: (1) FIRST read only the Spanish MT and put into one plain sentence what it tells you: who must do what, when, how much, what is allowed or forbidden, what something is, what happened, what the writer claims or feels. Do not use your English to repair the Spanish. (2) THEN read the English source (with its two context sentences) and compare it with your reading. (3) If your reading differs from the English on any point of content, or the Spanish gives you no clear reading → major (critical if acting on your reading could cause harm, financial loss or legal risk). If your reading matches and only the wording is odd → minor; if it also reads naturally → none. When you say major, start the explanation with "Read as:" and what you understood. span_mt = the exact MT text that misled you (or the worst imperfection); category = its MQM category (accuracy/mistranslation, accuracy/omission, accuracy/addition, accuracy/untranslated, terminology, fluency, style, locale). Use the official Spanish (ref_es), when present, only to check terminology, never as the gold: differing from it is not an error by itself.
Severity (same for every lens):
- major: read the way a Spanish-only reader reads it, the MT says something different from the English, or fails to say it: a wrong, missing, added or unintelligible piece of content (action, actor, object, amount, time, condition, negation, name, the writer's claim or evaluation).
- critical: a major error that could cause harm to health or safety, financial loss or legal risk for a reader who acts on it; or an output that is not a translation at all (a refusal, only commentary, a different text).
- minor: the meaning is intact for a Spanish-only reader: unidiomatic but correct wording; grammar, agreement, spelling or punctuation slips; register; slight loss of nuance; a dropped intensifier or hedge.
- none: nothing you would correct.
Calibration rules (from 247 WMT24 en-es units with professional MQM labels):
R1. Never decide by back-translating into English. A word, idiom or phrasal verb carried over word for word whose ordinary Spanish sense is different ('good call' → 'buena llamada'; 'won't work' → 'no va a trabajar'; 'could do with a break and a cuppa' → 'podría usar un descanso y una taza'), or a phrase too broken to say anything ('ponerlo todo de vuelta junto', 'para todos admirados'), is accuracy/mistranslation, not fluency, and it is major when it carries content of the sentence. "The gist is recoverable" or "a bilingual would get it" is not a reason for minor. It stays minor only if the literal Spanish still means the intended thing ('diseñar alrededor de ella', 'se siente', 'hacer la diferencia').
R2. An MT that stops before the end of the source, or drops a clause, number, actor, condition or negation, is accuracy/omission and major, even if what is left reads well. A dropped intensifier, filler or hedge ('way', 'just', 'about') is minor.
R3. An informal, disfluent or transcribed source does not excuse the MT: if the source's intended meaning is recoverable and the MT does not convey it, or the MT adds nonsense, that is major.
R4. Anything in the MT that is not the translation (notes, commentary, several alternative versions, code fences, a refusal) is accuracy/addition and major; critical if no usable translation is given.
R5. confidence = your probability (0-1) that a careful professional rater would make the same major-or-not decision on this unit; 0.5 means a coin flip.
Annotate every unit on its own; return one record per uid; do not skip units.
<!-- /LENS -->

**Freeze record.** The sha256 of each lens text, taken between its `<!-- LENS:... -->` and
`<!-- /LENS -->` markers with no surrounding newlines:

- `MQM`: `f4a42350c235faf459b95fa4613f12b304d8b50829f9f2def7dc781693dceaaf`
- `POSTEDITOR`: `a92790b3bbb0d7214fd8fef6c6b1bfff04f7b6a240029cee3ac88a41ec9ee7b5`
- `READER`: `1a566e69e738231a95d33cd80cbbb2de983c7b1c0e468d9bec0835e7cdae2d91`

## 3. Annotation record

One record per uid per lens:

```json
{"uid": "...", "severity": "none|minor|major|critical", "category": "accuracy/mistranslation|...|none",
 "span_mt": "exact MT substring of the worst error, empty if none", "explanation": "<= 25 words",
 "confidence": 0.0}
```

A lens **votes Major** when `severity` is `major` or `critical`.

## 4. Aggregation and adjudication

1. **Majority vote.** A unit is Major when at least 2 of the 3 lenses vote Major, and OK
   otherwise. With three annotators and two classes a tie cannot happen.
2. **Adjudication queue.** Every unit that meets one of these rules is adjudicated. The merge
   step records which rule fired, in `adjudication_reason`:
   - `critical`: any lens says `critical` (pre-registered).
   - `minority_major_conf>=0.7`: a 2-1 split whose single dissenting Major vote has
     confidence >= 0.7 (pre-registered). With three lenses, "the minority says
     major/critical" means exactly one of the three votes is Major.
   - `minority_major_low_conf`: every other 2-1 split in which exactly one lens votes Major,
     whatever its confidence. This is calibration amendment A1 (section 6). In calibration,
     all six lone Major votes had confidence 0.50 to 0.55, so the pre-registered threshold
     would have sent none of them to adjudication. Yet 4 of the 6 units were human Majors.
   A 2-1 split whose single dissenter says minor or none (majority Major) is not adjudicated.
   In calibration the majority was right on 4 of those 5 units.
3. **Adjudicator.** A senior adjudicator reads src, context, mt and ref_es and the three
   annotations, applies the same severity scale and rules R1 to R4, and gives the final
   label (Major or OK) with a one-line reason. The adjudicator is blind to system identity
   and scores. On a unit that is missing an annotation, the adjudicator first annotates the
   missing lens. No unit is finalised with fewer than three annotations.
4. **Final label.** The final label is the adjudicated label where one exists, and the
   majority vote otherwise. Because every queued unit keeps its `adjudication_reason`, two
   labellings can be rebuilt: the primary one, and the pre-registered one, which uses the
   majority vote for `minority_major_low_conf` units. Report both.
5. **Agreement to report.** Report Fleiss' kappa among the three lenses (binary, and 3-class
   none/minor/major+), pairwise Cohen's kappa, the vote distribution, and the share of units
   adjudicated, both overall and by reason.

## 5. Calibration (v1 lens texts against WMT24 en-es human MQM)

**Set.** The set has 247 units from `data/splits/en-es.dev.jsonl`: 22 whole source chunks
(social 204, speech 19, news 24) and 14 WMT24 systems. There are 44 human Majors (17.8%).
Major-bearing streams were oversampled; the natural rate is 5.7%. The sampling is described in
`data/realdocs/gold/calib/SAMPLING.json` (`scripts/gold_calib_units.py`). The human label
is the WMT `label` field: Major when there is any major or critical span in the meaning
family (`scripts/convert.py`). There is one human rater per segment. `ref_es` was empty.

**Scoring.** We used `scripts/gold_calib_eval.py`. A lens votes Major when it says
major or critical. The 95% CIs come from a cluster bootstrap over the 22 source chunks (2000
resamples). The "weighted" columns use the post-stratification weights from SAMPLING.json,
which bring the Major rate back to its natural value.

| Rater | Pred. Major | TP | FP | FN | Precision | Recall | F1 (95% CI) | Cohen's kappa vs human (95% CI) | Weighted P / R / F1 / kappa |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| MQM lens | 49 | 33 | 16 | 11 | 0.67 | 0.75 | 0.71 (0.59-0.80) | 0.64 (0.51-0.75) | 0.49 / 0.75 / 0.60 / 0.57 |
| Post-editor lens | 48 | 30 | 18 | 14 | 0.62 | 0.68 | 0.65 (0.57-0.75) | 0.57 (0.47-0.69) | 0.43 / 0.68 / 0.53 / 0.49 |
| Reader lens | 42 | 27 | 15 | 17 | 0.64 | 0.61 | 0.63 (0.53-0.73) | 0.55 (0.44-0.67) | 0.49 / 0.61 / 0.54 / 0.51 |
| **Majority (>= 2 of 3)** | 46 | 30 | 16 | 14 | 0.65 | 0.68 | **0.67 (0.57-0.75)** | **0.59 (0.48-0.69)** | 0.47 / 0.68 / 0.56 / 0.53 |
| Any lens (>= 1 of 3) | 52 | 34 | 18 | 10 | 0.65 | 0.77 | 0.71 (0.60-0.80) | 0.64 (0.51-0.75) | 0.46 / 0.77 / 0.58 / 0.54 |
| Unanimous (3 of 3) | 41 | 26 | 15 | 18 | 0.63 | 0.59 | 0.61 (0.52-0.72) | 0.53 (0.41-0.66) | 0.48 / 0.59 / 0.53 / 0.50 |

**Agreement among the lenses.** Fleiss' kappa is 0.90 on binary Major (95% CI 0.82-0.96)
and 0.77 on 3-class none/minor/major+. Pairwise Cohen's kappa is 0.91 (MQM and post-editor),
0.91 (MQM and reader) and 0.89 (post-editor and reader). 236 of the 247 units are unanimous
on Major: 195 have 0 of 3 Major votes and 41 have 3 of 3. Only 11 are 2-1 splits.

**Human-human agreement for MQM.** On this split it is **unknown**. WMT24 en-es has one MQM
rater per segment, so it cannot be measured here. In the literature:

- Freitag et al. 2021 (TACL, "Experts, Errors, and Context", section 4.4) collected three
  ratings per segment but report no agreement coefficient. They report only the rater
  severity spread: within ±20% of the mean for en-de and ±30% for zh-en.
- Figures of 0.584 (en-de) and 0.412 (zh-en) "pairwise inter-rater agreement" are attributed
  to that paper by arXiv 2303.13809 and arXiv 2409.14335. The metric is not stated, and the
  figures are not in the primary text, so they are not used here.
- arXiv 2407.03658 ("GPT-4 vs. Human Translators", Table 4) computed segment-level Cohen's
  kappa on Freitag et al.'s WMT20 and WMT21 en-de MQM data: 0.208 and 0.230. Across that
  paper's own eight MQM sets the kappa is 0.234-0.544 (mean 0.321). The label set behind
  these kappas is not stated, so they are not exactly binary Major.
- Lommel, Burchardt and Popović 2014 (LREC MTE workshop) also report low agreement for MQM
  error annotation.

The panel's kappa of 0.59 against a single human rater is therefore at or above the
published MQM human-human range. That range is only indicative, because the metrics differ.

**Systematic issues found.** Section 6 says how each was fixed. The audit covers every
majority-vote disagreement. It is not blind: the evaluator saw the human labels. It was used
to find patterns, never to change a number.

| ID | Pattern | Evidence |
|---|---|---|
| S1 | Literal renderings that change the Spanish sense were called minor ("calque / fluency, gist recoverable") | 7 of 14 misses (e.g. 'buena llamada', 'no va a trabajar', 'podría usar un descanso y una taza'). In 6 of the 7, at least one lens's span overlaps the human span, so the problem was the severity, not detection. |
| S2 | Broken structure or fragments were called grammar-minor or none | 3 of 14 misses ('para todos admirados'; two copies of the same dangling 'por la que') |
| S3 | Truncated output was called minor | 1 miss: all three lenses noticed the cut-off and two of them called it minor |
| S4 | A disfluent speech source was used to excuse the MT | 1 miss ("faithful to noisy source") |
| S5 | The lenses collapse into one annotator | Fleiss' kappa 0.90. 41 of the 46 majority Majors are unanimous. The majority vote (F1 0.67) is worse than the MQM lens alone (0.71). The reader lens has the lowest recall (0.61): it repairs the Spanish with its English ("core message still understood"). |
| S6 | Only the MQM lens ever says critical | critical calls: MQM 2, post-editor 0, reader 0. The "any critical" adjudication trigger depended on one lens. |
| S7 | Confidence is informative only at the top | Major calls at confidence >= 0.8 are human Majors in 40 of 41 cases; below 0.8, in 50 of 98. All 6 lone Major votes had confidence 0.50-0.55, and 4 of those 6 units were human Majors. |

**Not systematic (no change).**

- **Terminology:** all 24 terminology calls were minor, and all 24 fell on units the human
  labelled OK. There is no sign of over-strictness.
- **Hedges and intensifiers:** omissions of these were called minor, as the human did.
- **Untranslated English:** the evidence is mixed: 2 false alarms against 1 miss.
- **Non-translation output** (LLM notes, alternative versions, refusals): the lenses voted
  Major on all 8 such units, and the human agreed on 7 of the 8. This is codified as R4 with no
  change intended.
- **False alarms:** 10 of the 16 look like genuine meaning errors that the single WMT rater
  left unmarked or minor. Examples: 'vivienda pública' for the private rented sector, and
  'alquileres más bajos' for "lower rent increases". 2 more are units where the WMT key
  contradicts itself: the same error pattern is labelled Major elsewhere. 2 are human majors
  in the fluency family, which are OK by the label definition. Only 2 (untranslated English)
  are panel strictness.
- **Other:** one miss looks like a human-label error.

## 6. Change log v1 to v2 (one revision, calibration evidence only)

| Change | Fixes | What changed |
|---|---|---|
| C1 = rule R1 | S1, S2 | A Spanish-only reading test for severity: word-for-word idioms and false friends whose Spanish sense differs, and phrases too broken to say anything, are accuracy/mistranslation and major. "Gist recoverable" no longer justifies minor. The examples are human-Major units from calibration, plus human-OK counter-examples that stay minor. |
| C2 = rule R2 | S3 | Truncation and dropped content are major. Dropped hedges and intensifiers stay minor. |
| C3 = rule R3 | S4 | A disfluent source does not excuse an MT that fails to convey the recoverable meaning. |
| C4 = rule R4 | (codified) | Non-translation output is addition and major, critical if no usable translation is given. This matches what the lenses and the humans already did. |
| C5 | S5 | Distinct procedures. MQM inventories spans first. The post-editor sorts its own minimal edits into MEANING and FORM. The reader reads the Spanish FIRST, writes down what it understood ("Read as:"), and only then reads the English. All three lenses share one severity scale and one MQM category list, so the outputs are comparable. |
| C6 | S6 | One definition of critical for all lenses: harm, financial loss or legal risk if acted on, or no translation at all. |
| C7 = rule R5, plus amendment A1 | S7 | Confidence is defined as the probability that a professional would make the same major-or-not call. Every 1-of-3 Major split is adjudicated (section 4). The pre-registered rules are kept and flagged separately. |

Unchanged from v1: the three roles, the instruction to judge against the English source
first and use the official Spanish only as supporting evidence, and the reader's focus on
who must do what, when, how much, and what is allowed or forbidden.

## 7. Limitations

- **v2 is not re-measured.** The calibration numbers above are for the v1 texts. v2 was not
  re-run, because re-running on the same 247 units would tune on the test. The expected
  effect is a higher recall (S1 accounts for half the misses) and a few more false alarms
  among human-OK literal renderings. The size of both is unknown. The best check would be a
  fresh draw from en-es.dev that excludes the 22 calibration chunks.
- **The reference is noisy.** It is one human rater per segment, and the audit found
  apparent misses and internal contradictions in the WMT key. Agreement with it is therefore
  a lower bound on agreement with the truth, and also on the panel's precision.
- **The annotators are not independent.** The three lenses are the same kind of AI annotator
  (Fleiss' kappa 0.90). C5 tries to decorrelate them, but correlated errors survive majority
  voting.
- **Domain shift.** Calibration used social, speech and news text with no reference. The
  gold set is public-service instructions with an official Spanish reference. The handling
  of `ref_es` and of "critical" (harm, money, legal risk) is untested by calibration.
- **Segment level.** Units are judged segment by segment, with two source sentences of
  context.

## 8. Reproduction and files

```
PYTHONPATH=. uv run --frozen python scripts/gold_calib_units.py   # calibration units (seeded)
PYTHONPATH=. uv run --frozen python scripts/gold_calib_eval.py    # calibration metrics
```

The following files are in `data/realdocs/gold/calib/` (git-ignored):

- `annotations/<lens>_batch_0X.jsonl`: the v1 calibration annotations, verbatim.
- `metrics.json`: every number above.
- `disagreements.jsonl` and `disagreement_audit.json`: the unit-level audit.
- `RESULTS.md`: the full calibration report.

Gold-panel merge (section 4, steps 1 and 2), reading only the batch files and the lens annotations:

```
PYTHONPATH=. uv run --frozen python -I scripts/gold_merge.py --import data/realdocs/gold/annotations/panel_return.json
```

It writes these files to `data/realdocs/gold/` (git-ignored):

- `annotations/<lens>_batch_XX.jsonl`: the v2 gold annotations, verbatim.
- `annotations_raw.jsonl`: one line per uid with the three lens records, the vote and the queue reasons.
- `to_adjudicate.jsonl`: the blind unit fields and the three annotations for every queued unit.
- `merge_summary.json`: missing uids, QA flags, the vote distribution, the queue by reason and the agreement figures.

Adjudication and final labels (section 4, steps 3 to 5). The adjudicator's decisions go to
`adjudication.jsonl` first, one line per queued uid, before anything joins the system key. Then the
finalisation joins `key_systems.json` and the earlier labels:

```
PYTHONPATH=. uv run --frozen python -I scripts/gold_finalize.py
```

It writes `labels.jsonl` (the final gold, one line per unit) and `gold_stats.json` (all the statistics) to
`data/realdocs/gold/`. The method, the numbers and the scoring rules are in `docs/REALDOCS_GOLD.md`.

## Appendix A. v1 lens texts (as run in calibration, superseded)

- **MQM:** "Annotate like a professional MQM rater (WMT MQM guidelines, Freitag et al. 2021):
  identify error spans in the MT, assign category (accuracy/mistranslation,
  accuracy/omission, accuracy/addition, accuracy/untranslated, terminology, fluency, style,
  locale) and severity (minor/major/critical). MAJOR = errors that change or obscure the
  meaning so a reader would be misled or confused; MINOR = imperfections that do not mislead.
  Judge the MT against the English source first; use the official Spanish only as supporting
  evidence of correct terminology, never as the gold."
- **POSTEDITOR:** "You are a senior English→Spanish post-editor at a translation agency. For
  each unit decide whether a professional post-editor MUST fix the MEANING before delivery (a
  reader would act or understand differently than the English intends) → major; otherwise
  fluency/style/minor terminology → minor or none. Judge against the English source; use the
  official Spanish only as supporting evidence."
- **READER:** "Read each MT segment as a Spanish-speaking member of the public who cannot read
  English and will act on this text. Would they misunderstand who must do what, when, how
  much, what is allowed/forbidden, or what something is, compared with the English? If yes →
  major (critical if it could cause harm, financial loss or legal risk). Otherwise
  minor/none. Use the official Spanish only to check terminology."

All three were followed by: "Annotate EVERY unit … Return one record per uid. Be consistent;
do not skip units."
