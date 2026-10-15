"""Wrong-sense synthetic data from Gemma 4 on DigitalOcean: ONE word or short term of a clean MT
segment is replaced, the edit is validated by code, then (optionally) checked by a second, blind
Gemma call before it is kept.

usage: uv run python scripts/synth_sense.py [--plan "en-es:train:*=700,*:test:*=85"] [--phase all|gen|verify]
           [--total-cap 0.90] [--gen-cap 0.70] [--max-requests 20000] [--workers 6]
           [--dry-run] [--cache-only] [--self-test] [--out-dir data/synth]
The 2026-10-14 run: --phase gen --gen-cap 0.69 --workers 6, then --phase verify (DEFAULT_PLAN);
--cache-only rebuilds data/synth/sense.* byte-identically from data/synth/cache/sense_cache.jsonl.

Edit types (label):
  wrong_sense   translation of a DIFFERENT sense of a polysemous English word  (Major, mistranslation)
  false_friend  a target-language false friend of the English word            (Major, mistranslation)
  antonym_term  a near-antonym / opposite-direction term, no negation word    (Major, mistranslation)
  synonym_ok    a true synonym or equally correct translation                 (OK)

Gemma does not echo the translation. It answers {"source_word", "source_sense", "new_sense",
"old", "new"}, and the code replaces the one word-bounded occurrence of `old` with `new`. Code checks
(`validate`): `old` occurs exactly once at word boundaries; after trimming common tokens the change
spans 1-4 tokens on both sides (a replacement, not an insertion or deletion); digits, negation-marker
count, placeholders and script are unchanged; the change is not case/accent-only; `source_word`
occurs in the source; a false friend must look like the English word (phonetic similarity, with
transliteration for Russian); the result is not a human-rated MT of the same source; no
integrity-list word (below) appears in the result.

Verification (phase `verify`): accepted edits go to Gemma again in blind batches of up to 4
(one item per type, shuffled), which answers SAME or CHANGED for each. Rows whose verdict disagrees
with their label are moved to the rejects file (reason `verify_disagree`, full record kept), and
`synth_verify` is stored on every kept row.

Integrity (pre-registered, see the task notes in docs/STATUS.md): the real-document words of the
evaluation set must never reach a generation prompt. Parents whose source or MT contains any of
them are excluded, every prompt is asserted free of them, and outputs containing them are rejected.
Documents of data/realdocs and the demo answer keys (demo/answer_key.en-es.json and the locally
generated data/demo_wmt/answer_key.en-es.wmt24.json; and other chunks of the same WMT story)
are excluded too. Parents come from secondread.synth.source_pool(lp, split): train documents ->
sense.train, test documents -> held-out sense.test.

Selection is independent per (lp, split, type) and prefix-stable: raising a count only appends jobs,
so top-ups reuse every cached reply. Parents are interleaved round-robin across WMT domains, with
at most --max-per-seg-type edits per source segment and type; the second one asks for a word from
the other half of the sentence.

Cost: replies are cached in data/synth/cache/sense_cache.jsonl. --total-cap bounds the spend of
everything in that cache (this task's whole budget), at $0.18/M input and $0.50/M output.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import itertools
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from secondread.synth import (SYNTH_DIR, assert_hygiene, digit_string, error_span, make_record,  # noqa: E402
                              natural_mts, negation_count, read_jsonl, rng_for, segment_key, shift_spans,
                              source_pool, target_lang, write_jsonl)
from synth_llm import (DO_BASE_URL, DO_MODEL, LANG_NAME, PRICE_IN, PRICE_OUT, PROTECTED_RE, Cache, Job,  # noqa: E402
                       cost, est_tokens, run_jobs, script_ok)

GENERATOR = "llm-gemma4-sense-v1"
CACHE_PATH = SYNTH_DIR / "cache" / "sense_cache.jsonl"
TYPES = ("wrong_sense", "false_friend", "antonym_term", "synonym_ok")
LABEL = {t: ("OK" if t.endswith("_ok") else "Major") for t in TYPES}
EXPECT = {"Major": "CHANGED", "OK": "SAME"}
# The plan of the 2026-10-14 run (data/synth/sense.*.jsonl): job counts per (lp, split, type).
DEFAULT_PLAN = ('en-es:train:wrong_sense=908,en-es:train:false_friend=600,en-es:train:antonym_term=700,'
                'en-es:train:synonym_ok=650,en-ru:train:wrong_sense=740,en-ru:train:false_friend=480,'
                'en-ru:train:antonym_term=510,en-ru:train:synonym_ok=540,en-es:test:wrong_sense=120,'
                'en-es:test:false_friend=180,en-es:test:antonym_term=72,en-es:test:synonym_ok=68,'
                'en-ru:test:wrong_sense=110,en-ru:test:false_friend=180,en-ru:test:antonym_term=76,'
                'en-ru:test:synonym_ok=80')

# Integrity list: the words of the real-document misses/catches (docs/REALDOCS.md) and their
# inflections. Over-matching is harmless (it only excludes more parents).
FORBIDDEN_RE = re.compile(
    r"(?<!\w)(?:return\w*|devoluci\w*|flush\w*|lav(?:ar|o|a|as|an|ado|ada|ados|adas|ando|e|en|es|aron|ó|é|"
    r"arse|arlo|arla|arlos|arlas)|paperless|impres[ao]s?|partner\w*|parejas?|take[\s-]*back)(?!\w)", re.IGNORECASE)

# --------------------------------------------------------------------------- prompts

EXAMPLES = {
    "wrong_sense": {
        "es": ('"bank" (of a river) -> «banco» (finance); "fine" (penalty) -> «bien» (well); "charge" (fee) -> '
               '«carga» (load); "bill" (draft law) -> «factura» (invoice); "discharge" (from hospital) -> «descarga» '
               '(electric)'),
        "ru": ('"bank" (of a river) -> «банк» (finance); "fine" (penalty) -> «хорошо» (well); "charge" (fee) -> '
               '«заряд» (electric); "bill" (draft law) -> «счёт» (invoice); "discharge" (from hospital) -> «разряд» '
               '(electric)'),
    },
    "false_friend": {
        "es": ('"actually" -> «actualmente» (currently); "eventually" -> «eventualmente» (possibly); "sensible" -> '
               '«sensible» (sensitive); "embarrassed" -> «embarazada» (pregnant); "library" -> «librería» (bookshop)'),
        "ru": ('"magazine" -> «магазин» (shop); "actually" -> «актуально» (topical); "decade" -> «декада» (ten days); '
               '"sympathetic" -> «симпатичный» (good-looking); "accurate" -> «аккуратный» (tidy)'),
    },
}
INSTRUCTIONS = {
    "wrong_sense": (
        "Pick an English word of the source that has several distinct meanings and replace its translation with a "
        "{lang} translation of a DIFFERENT meaning of that word, wrong in this context, as if the translator "
        "confused two senses. E.g. {examples}."),
    "false_friend": (
        "Pick an English word of the source that has a {lang} false friend (a similar-looking word with another "
        "meaning) and replace the correct translation with it, adapting endings, so that the translation says "
        "something the source does not. E.g. {examples}. Use a real {lang} word; never invent one, and do not "
        "force a pair that is not a false friend."),
    "antonym_term": (
        "Replace one word or term of the translation with a near-antonym or an opposite-direction term "
        "(increase/decrease, buy/sell, before/after, maximum/minimum, arrive/leave, allowed/forbidden, early/late), "
        "so that it states the opposite of the source yet reads naturally. Do not add or remove a negation."),
    "synonym_ok": (
        "Replace one word or term of the translation with a true synonym or another correct translation of the "
        "same English word that keeps EXACTLY the meaning in this context (same sense and register), so that the "
        "translation stays fully correct. Avoid near-synonyms that shift, weaken or strengthen the meaning."),
}
SLOT_HINT = ("", " If possible, pick a word in the second half of the sentence.",
             " If possible, pick a word in the first half of the sentence.")
PROMPT = """Create test data for a checker of meaning errors in machine translation.

English: {src}
{lang}: {mt}

Task: {instruction}

Rules: change ONE word or short term (max 3 words) of the translation, plus an adjacent article or ending only if grammar requires it. Keep numbers, names, negations, URLs, tags and placeholders (e.g. PRS_ORG, #NAME#). The result must stay fluent {lang}.{hint} Check your choice silently before answering: the new text must differ from the old one and fit the task. If the sentence offers no suitable word, reply {{"skip": true}}.

Reply with ONE JSON object and nothing else: {{"source_word": "<English word targeted>", "source_sense": "<its meaning here, 2-4 words>", "new_sense": "<meaning of the replacement, 2-4 words>", "old": "<exact text replaced, copied from the translation>", "new": "<replacement>"}}"""

VERIFY_PROMPT = """You are reviewing edits to {lang} translations of English sentences. In each item, one word or term of a translation was replaced. Judge the edited translation against its English source.

Answer SAME if the edited translation still says exactly what the source says (the new word is a correct translation in this context). Answer CHANGED if the replacement changes the meaning so that a reader of the translation would be misled. Answer BROKEN if the new word is not a real {lang} word or makes the sentence ungrammatical.

{items}

Answer with JSON only, one verdict per item: {{{fmt}}}"""
VERIFY_ITEM = "Item {n}\nSource: {src}\nEdited translation: {mt}\nReplaced: «{old}» -> «{new}»"


def assert_clean_prompt(text: str) -> None:
    m = FORBIDDEN_RE.search(text)
    assert m is None, f"integrity: forbidden word {m.group(0)!r} in a prompt"


def build_messages(parent: dict, synth_type: str, slot: int) -> list[dict]:
    lang = target_lang(parent["lp"])
    examples = EXAMPLES.get(synth_type, {}).get(lang, "")
    instruction = INSTRUCTIONS[synth_type].format(lang=LANG_NAME[lang], examples=examples)
    text = PROMPT.format(src=parent["src"], mt=parent["mt"], lang=LANG_NAME[lang], instruction=instruction,
                         hint=SLOT_HINT[min(slot, len(SLOT_HINT) - 1)])
    assert_clean_prompt(text)
    return [{"role": "user", "content": text}]


# --------------------------------------------------------------------------- integrity: real documents

def realdoc_docs() -> set[str]:
    """WMT document ids used by the real-document evaluation (data/realdocs, demo answer keys)."""
    docs: set[str] = set()
    for f in Path("data/realdocs").glob("*/aligned.jsonl"):
        docs.update(r["doc"] for r in read_jsonl(f) if r.get("doc"))
    for key in (Path("demo/answer_key.en-es.json"), Path("data/demo_wmt/answer_key.en-es.wmt24.json")):
        if not key.exists():
            continue

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, str) and re.fullmatch(r"test-en-\S+", x):
                docs.add(x)
        walk(json.loads(key.read_text(encoding="utf-8")))
    return docs


def story(doc: str) -> str:
    return re.sub(r"_chunk_\d+.*$", "", doc)


# --------------------------------------------------------------------------- selection

# English words with a well-known false friend in the target language (stems, matched at the start
# of a source token). Only an ORDERING hint for false_friend jobs: parents whose source has one are
# asked first, the rest follow; the prompt never names the word. The integrity-list words are absent.
FF_STEMS = {  # "word*" matches any token starting with word; "word" matches it plus a plain suffix
    "es": "actual* eventual* sensible embarrass* librar* assist* attend* realiz* realis* support* career* college* "
          "carpet* exit* fabric* introduc* lecture* preservative* record* resume* success* constipat* demand* "
          "compromis* parent* argument* advertis* casual* commodit* conductor* contest* deception* disgust* educated "
          "exciting familiar large molest* notice* pretend* remove* rope sane sympath* terrific topic* ultimate* vase "
          "billion* collar* complexion* costume* diversion* editor* injur* mayor* misery notorious* policy policies "
          "presum* prevent* receipt* regular* resign* signature* soap stranger* suburb* tuna vicious* agenda* assum* "
          "attempt* discuss* content grocer* physician* bizarre fastidious jubilation ingenu* carton* balloon* "
          "intoxicat* sensitive approbat* relativ* letter* quit".split(),
    "ru": "magazine* actual* decade* sympath* accurate* artist* complexion* data family families lunatic* prospect* "
          "replica* record* conductor* academic* ammunition* angina applicat* aspirant* audience* balloon* "
          "baton* brilliant* cabinet* carton* chef compositor* concurrent* decorat* director* dramatic* extravagant* "
          "fabric* figure* genial* intelligent* list mark novel* paragraph* partisan* pathetic* physician* "
          "provision* realiz* realis* revision* satin* smoking stool* tablet* talon* trap troop* product* control* "
          "pretend* instance* insult* clay herb* desert* spirit* silicon* obligation* salute* patron* technique* "
          "concern* credit* occupation* mixer* resin* expertise* liquidat* mayor* pamphlet* rubric* climax* cart "
          "lyric* objective* principal* servant* solid* tribune* virtuos* cream* agitat* advocate* anecdote* "
          "camera* conserv* corpus* fantas* magnet* accent* affair* aggregat* career* costume* fiction* gymnasi* "
          "invalid personal public receipt* routine* specific* commission* mode racket* rock session* stamp* "
          "subject* tax occasion* nature original*".split(),
}
FF_SUFFIXES = ("", "s", "es", "ed", "d", "ing", "ly", "er", "ers", "ies")


def ff_hit(parent: dict) -> str | None:
    toks = re.findall(r"[a-z]+", parent["src"].lower())
    for entry in FF_STEMS[target_lang(parent["lp"])]:
        stem = entry.rstrip("*")
        for t in toks:
            if (t.startswith(stem) if entry.endswith("*") else t in {stem + x for x in FF_SUFFIXES}):
                return t
    return None


def parent_pool(lp: str, split: str, args, excluded_stories: set[str]) -> tuple[list[dict], collections.Counter]:
    pool, _ = source_pool(lp, split)
    stats: collections.Counter = collections.Counter(parents=len(pool))
    out = []
    for p in pool:
        if len(p["mt"]) > args.max_chars or len(p["src"]) > args.max_chars:
            stats["skip_long"] += 1
        elif FORBIDDEN_RE.search(p["src"]) or FORBIDDEN_RE.search(p["mt"]):
            stats["skip_integrity_word"] += 1
        elif story(p["doc"]) in excluded_stories:
            stats["skip_realdoc"] += 1
        elif len(re.findall(r"\w+", p["src"])) < args.min_src_words:
            stats["skip_short"] += 1
        else:
            out.append(p)
    stats["eligible"] = len(out)
    return out, stats


def select_jobs(pool: list[dict], synth_type: str, n: int, args) -> list[Job]:
    """First n jobs of a deterministic sequence: domains round-robin, parents in seeded order,
    at most --max-per-seg-type per source segment. A larger n only appends. For false_friend,
    parents with a listed word (FF_STEMS) come first, then the rest."""
    tiers = [pool]
    if synth_type == "false_friend":
        tiers = [[p for p in pool if ff_hit(p)], [p for p in pool if not ff_hit(p)]]
    per_seg: collections.Counter = collections.Counter()
    jobs: list[Job] = []
    for tier in tiers:
        by_dom: dict[str, list[dict]] = collections.defaultdict(list)
        for p in tier:
            by_dom[p["domain"]].append(p)
        queues = [collections.deque(sorted(ps, key=lambda p: rng_for(args.seed, synth_type, p["id"]).random()))
                  for _, ps in sorted(by_dom.items())]
        while len(jobs) < n and any(queues):
            for q in queues:
                while q and len(jobs) < n:
                    p = q.popleft()
                    slot = per_seg[segment_key(p)]
                    if slot >= args.max_per_seg_type:
                        continue
                    per_seg[segment_key(p)] += 1
                    max_tokens = min(args.max_tokens, 150 + est_tokens(p["mt"]) // 4)
                    jobs.append(Job(p, synth_type, build_messages(p, synth_type, slot), max_tokens,
                                    args.temperature))
                    break
    return jobs


def parse_plan(plan: str) -> list[tuple[str, str, str, int]]:
    rules = []
    for part in filter(None, (s.strip() for s in plan.split(","))):
        key, n = part.split("=")
        lp, split, t = key.split(":")
        rules.append((lp, split, t, int(n)))
    return rules


def plan_counts(plan: str, lps: list[str], splits: list[str]) -> dict[tuple[str, str, str], int]:
    counts = {}
    for lp, split, t in itertools.product(lps, splits, TYPES):
        n = 0
        for rlp, rsplit, rt, rn in parse_plan(plan):  # the last matching rule wins
            if fnmatch.fnmatch(lp, rlp) and fnmatch.fnmatch(split, rsplit) and fnmatch.fnmatch(t, rt):
                n = rn
        if n:
            counts[(lp, split, t)] = n
    return counts


# --------------------------------------------------------------------------- validation

def json_objects(text: str) -> tuple[list[dict], str]:
    """Every top-level JSON object in the reply, and the text after the last one."""
    dec, objs, i, tail = json.JSONDecoder(), [], 0, text
    while (j := text.find("{", i)) >= 0:
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
            tail = text[end:]
        i = end
    return objs, tail


def parse_obj(text: str) -> dict | None:
    """The first JSON object (used for the verifier's single reply)."""
    objs, _ = json_objects(text)
    return objs[0] if objs else None


def parse_edit(text: str) -> tuple[dict | None, str | None]:
    """(edit object, None) or (None, reason). Gemma sometimes corrects itself in a second JSON
    object: the last one counts, but only if no comment follows it."""
    objs, tail = json_objects(text)
    if not objs:
        return None, "no_json"
    if re.sub(r"[`\s]|json", "", tail):
        return None, "comment_after_json"
    return objs[-1], None


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def fold(s: str) -> str:
    return " ".join(strip_accents(s.lower()).replace("ё", "е").split())


TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def change_span(old: str, new: str) -> tuple[list[str], list[str]]:
    """Tokens that differ after removing the common token prefix and suffix."""
    a, b = TOKEN_RE.findall(old), TOKEN_RE.findall(new)
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return a[p:len(a) - s], b[p:len(b) - s]


TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s",
                     "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "i", "", "e", "iu", "ia"]))


def phonetic(word: str) -> str:
    """Rough cross-language spelling key: Latin letters, c/k, y/i, ph/f, z/s, doubles collapsed."""
    w = "".join(TRANSLIT.get(c, c) for c in strip_accents(word.lower()))
    for a, b in (("ph", "f"), ("th", "t"), ("ck", "k"), ("qu", "k"), ("x", "ks"), ("c", "k"), ("y", "i"),
                 ("w", "v"), ("z", "s"), ("h", "")):
        w = w.replace(a, b)
    w = re.sub(r"[^a-z]", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def looks_alike(source_word: str, text: str) -> float:
    keys = [phonetic(w) for w in source_word.split()]
    best = 0.0
    for w in re.findall(r"\w+", text):
        if len(w) < 3:
            continue
        pw = phonetic(w)
        for k in keys:
            n = min(len(k), len(pw), 6)  # compare stems: "actually" ~ "actualmente"
            best = max(best, SequenceMatcher(None, k, pw).ratio(), SequenceMatcher(None, k[:n], pw[:n]).ratio())
    return best


def in_source(source_word: str, src: str) -> bool:
    sw, s = fold(source_word), fold(src)
    if not sw:
        return False
    if sw in s:
        return True
    toks = set(re.findall(r"\w+", s))
    for w in re.findall(r"\w+", sw):  # inflection: "charges" for "charged"
        stem = w[:max(3, len(w) - 3)]
        if not any(t.startswith(stem) for t in toks):
            return False
    return True


def cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("\u0400" <= c <= "\u04ff" for c in letters) / max(1, len(letters))


def locate(mt: str, old: str) -> tuple[int | None, str | None]:
    """Start of the single word-bounded occurrence of `old` in mt, or (None, reason)."""
    if not old:
        return None, "missing_field"
    pat = (r"(?<!\w)" if old[0].isalnum() else "") + re.escape(old) + (r"(?!\w)" if old[-1].isalnum() else "")
    hits = [m.start() for m in re.finditer(pat, mt)]
    if not hits:
        return None, ("old_not_word" if old in mt else "old_not_found")
    if len(hits) > 1:
        return None, "old_not_unique"
    return hits[0], None


def validate(parent: dict, synth_type: str, obj: dict | None, natural: dict[str, set[str]] | None = None
             ) -> tuple[str | None, dict]:
    """(None, edit) if the edit is the one we asked for, else (rejection reason, partial info)."""
    mt, src, lang = parent["mt"], parent["src"], target_lang(parent["lp"])
    if obj is None:
        return "no_json", {}
    if obj.get("skip"):
        return "skipped", {}
    fields = {k: obj.get(k) for k in ("source_word", "source_sense", "new_sense", "old", "new")}
    if not all(isinstance(fields[k], str) and fields[k].strip() for k in ("source_word", "old", "new")):
        return "missing_field", {}
    old, new = fields["old"].strip(), fields["new"].strip()
    info = {"old": old, "new": new, "source_word": fields["source_word"].strip(),
            "source_sense": (fields["source_sense"] or "").strip() if isinstance(fields["source_sense"], str) else "",
            "new_sense": (fields["new_sense"] or "").strip() if isinstance(fields["new_sense"], str) else ""}
    start, reason = locate(mt, old)
    if reason:
        return reason, info
    new_mt = mt[:start] + new + mt[start + len(old):]
    info.update(start=start, new_mt=new_mt)
    if fold(old) == fold(new) or re.sub(r"\W", "", fold(old)) == re.sub(r"\W", "", fold(new)):
        return ("unchanged" if old == new else "trivial_change"), info
    if "\n" in new:
        return "multiline", info
    if not script_ok(mt, new_mt, lang) or cyrillic_share(new) < 0.5 <= cyrillic_share(old) \
            or (lang != "ru" and cyrillic_share(new) > 0):
        return "wrong_script", info
    if collections.Counter(PROTECTED_RE.findall(new_mt)) != collections.Counter(PROTECTED_RE.findall(mt)):
        return "placeholder_changed", info
    if digit_string(new_mt) != digit_string(mt):
        return "digits_changed", info
    if negation_count(new_mt, lang) != negation_count(mt, lang):
        return "negation_changed", info
    a, b = change_span(mt, new_mt)
    if not a or not b:
        return "not_a_replacement", info
    if len(a) > 4 or len(b) > 4:
        return "edit_not_local", info
    if not any(re.match(r"\w", t) for t in a + b):
        return "punctuation_only", info
    if not in_source(info["source_word"], src):
        return "source_word_not_in_src", info
    if FORBIDDEN_RE.search(new_mt) or FORBIDDEN_RE.search(info["source_word"]):
        return "integrity_word", info
    if natural is not None and new_mt in natural.get(src, set()):
        return "matches_natural_mt", info
    if synth_type == "false_friend":
        info["alike"] = round(looks_alike(info["source_word"], new), 3)
        if info["alike"] < 0.5:
            return "not_a_false_friend", info
    return None, info


def build_record(parent: dict, synth_type: str, info: dict, model: str) -> dict:
    start, old, new, new_mt = info["start"], info["old"], info["new"], info["new_mt"]
    end = start + len(new)
    spans = shift_spans(parent["spans"], start, start + len(old), len(new))
    if LABEL[synth_type] == "Major":
        spans.append(error_span(new_mt, start, end, synth_type))
        spans.sort(key=lambda sp: (sp["start"], sp["end"]))
    rec = make_record(parent, mt=new_mt, label=LABEL[synth_type], synth_type=synth_type, generator=GENERATOR,
                      edit={"start": start, "end": end, "text": new, "orig": old}, spans=spans)
    rec["synth_model"] = model
    rec["synth_sense"] = {k: info[k] for k in ("source_word", "source_sense", "new_sense")}
    if "alike" in info:
        rec["synth_sense"]["alike"] = info["alike"]
    return rec


def collect(jobs: list[Job], cache: Cache, natural) -> tuple[list[tuple[int, dict]], list[dict]]:
    """[(job index, record)] for accepted edits and rejection rows; jobs without a reply are skipped."""
    accepted, rejected = [], []
    seen: set[tuple] = set()  # the same edit of the same source sentence (another MT system's output)
    for k, job in enumerate(jobs):
        entry = cache.get(job.key)
        if entry is None:
            continue
        obj, reason = parse_edit(entry["reply"])
        info: dict = {}
        if entry.get("finish_reason") == "length":
            reason = "truncated"
        elif reason is None:
            reason, info = validate(job.parent, job.synth_type, obj, natural)
            if reason is None:
                dup = (segment_key(job.parent), fold(info["source_word"]), fold(info["new"]))
                reason = "duplicate_edit" if dup in seen else None
                seen.add(dup)
        if reason:
            rejected.append({"parent": job.parent["id"], "lp": job.parent["lp"], "synth_type": job.synth_type,
                             "reason": reason, "mt": job.parent["mt"],
                             **{k: v for k, v in info.items() if k in ("old", "new", "source_word", "new_mt", "alike")},
                             "reply": entry["reply"]})
        else:
            rec = build_record(job.parent, job.synth_type, info, entry.get("model", DO_MODEL))
            if job.synth_type == "false_friend":
                rec["synth_sense"]["ff_list_hit"] = ff_hit(job.parent)
            accepted.append((k, rec))
    return accepted, rejected


# --------------------------------------------------------------------------- verification

def verify_jobs(lp: str, split: str, accepted: dict[str, list[tuple[int, dict]]], args) -> list[tuple[Job, list[dict]]]:
    """Batch b holds the accepted records of jobs 2b and 2b+1 of every type (blind, shuffled): at most
    8 items, mixed types, and stable when a type's job list grows."""
    lang = target_lang(lp)
    by_k: dict[int, list[dict]] = collections.defaultdict(list)
    for t in TYPES:
        for k, rec in accepted.get(t, []):
            by_k[k // 2].append(rec)
    out = []
    for k in sorted(by_k):
        recs = sorted(by_k[k], key=lambda r: r["id"])
        rng_for(args.seed, "verify", lp, split, str(k)).shuffle(recs)
        items = "\n\n".join(VERIFY_ITEM.format(n=i + 1, src=r["src"], mt=r["mt"], old=r["synth_edit"]["orig"],
                                               new=r["synth_edit"]["text"]) for i, r in enumerate(recs))
        fmt = ", ".join(f'"{i + 1}": "SAME, CHANGED or BROKEN"' for i in range(len(recs)))
        text = VERIFY_PROMPT.format(lang=LANG_NAME[lang], items=items, fmt=fmt)
        assert_clean_prompt(text)
        job = Job({"id": f"verify|{lp}|{split}|{k}", "lp": lp, "mt": ""}, "verify",
                  [{"role": "user", "content": text}], 16 + 8 * len(recs), 0.0)
        out.append((job, recs))
    return out


def parse_verdicts(reply: str, n: int) -> list[str | None]:
    obj = parse_obj(reply) or {}
    out = []
    for i in range(n):
        v = obj.get(str(i + 1))
        v = v.strip().upper() if isinstance(v, str) else None
        out.append(v if v in ("SAME", "CHANGED", "BROKEN") else None)
    return out


# --------------------------------------------------------------------------- spend

def cache_spend(cache: Cache) -> tuple[int, int, int, float]:
    n = tin = tout = 0
    for e in cache.data.values():
        u = e.get("usage") or {}
        n += 1
        tin += u.get("prompt_tokens") or 0
        tout += u.get("completion_tokens") or 0
    return n, tin, tout, cost(tin, tout)


def call_all(todo: list[Job], cache: Cache, args, what: str, cap: float) -> None:
    """Call `todo` until the sense cache's spend reaches `cap`. The hard guard is synth_llm's Budget:
    actual cost of finished calls + worst case of calls in flight <= cap - spend so far."""
    if not todo:
        return
    _, _, _, spent = cache_spend(cache)
    room = cap - spent
    if len(todo) > args.max_requests:
        print(f"  --max-requests {args.max_requests}: calling only the first {args.max_requests}")
        todo = todo[:args.max_requests]
    chars = sum(len(j.messages[0]["content"]) for j in todo)
    typical = cost(int(chars / 4.3), sum(min(j.max_tokens, 90) for j in todo))  # 4.3 chars/token measured
    print(f"{what}: {len(todo)} calls; spent ${spent:.4f}, cap ${cap:.2f} (room ${room:.4f}); "
          f"typical cost ~${typical:.4f} (4.3 chars/token), worst case ${sum(j.worst_cost for j in todo):.4f}")
    if args.dry_run or args.cache_only or room <= 0:
        return
    args.max_cost = room
    stats = run_jobs(todo, cache, args)
    print(f"api ({what}): {stats}")


# --------------------------------------------------------------------------- self-test

def self_test() -> None:
    base = {"year": 0, "system": "test", "doc": "test", "domain": "test", "rater": "test", "label": "OK"}
    es = {**base, "id": "t-es", "lp": "en-es",
          "src": "The bank on the river charges a small fee, and you can actually pay it 2 times.",
          "mt": "El banco del río cobra una pequeña tarifa, y en realidad puedes pagarla 2 veces.", "spans": []}
    ru = {**base, "id": "t-ru", "lp": "en-ru", "src": "I read the magazine before the meeting, not after it.",
          "mt": "Я прочитал журнал до встречи, а не после неё.", "spans": []}

    def o(sw, old, new):
        return {"source_word": sw, "source_sense": "s", "new_sense": "n", "old": old, "new": new}

    cases = [
        (es, "wrong_sense", o("charges", "cobra", "carga"), None),
        (es, "wrong_sense", o("charges", "una pequeña tarifa", "una pequeña carga eléctrica muy fuerte del río"), "edit_not_local"),
        (es, "false_friend", o("actually", "en realidad", "actualmente"), None),
        (es, "false_friend", o("actually", "en realidad", "ahora mismo"), "not_a_false_friend"),
        (es, "antonym_term", o("small", "pequeña", "gran"), None),
        (es, "antonym_term", o("2", "2 veces", "3 veces"), "digits_changed"),
        (es, "synonym_ok", o("fee", "tarifa", "cuota"), None),
        (es, "synonym_ok", o("fee", "tarifa", "Tarifa"), "trivial_change"),
        (es, "synonym_ok", o("fee", "tarifa", "tarifa"), "unchanged"),
        (es, "synonym_ok", o("fee", "tarif", "cuot"), "old_not_word"),
        (es, "synonym_ok", o("fee", "comisión", "cuota"), "old_not_found"),
        (es, "synonym_ok", o("wallet", "tarifa", "cuota"), "source_word_not_in_src"),
        (es, "antonym_term", o("can", "puedes", "no puedes"), "negation_changed"),
        (es, "wrong_sense", o("river", "del río", "del río grande"), "not_a_replacement"),
        (es, "synonym_ok", {"skip": True}, "skipped"),
        (es, "synonym_ok", None, "no_json"),
        (es, "synonym_ok", o("fee", "tarifa", "pareja"), "integrity_word"),
        (ru, "false_friend", o("magazine", "журнал", "магазин"), None),
        (ru, "antonym_term", o("before", "до", "после"), None),
        (ru, "antonym_term", o("after", "не после", "после"), "negation_changed"),
        (ru, "synonym_ok", o("read", "прочитал", "read"), "wrong_script"),
    ]
    assert locate(ru["mt"], "до") == (ru["mt"].index("до "), None)
    for parent, t, obj, want in cases:
        got, info = validate(parent, t, obj)
        assert got == want, (t, obj, got, want)
        if got is None:
            rec = build_record(parent, t, info, "test")
            for sp in rec["spans"]:
                assert rec["mt"][sp["start"]:sp["end"]] == sp["text"], sp
            assert rec["mt"][rec["synth_edit"]["start"]:rec["synth_edit"]["end"]] == rec["synth_edit"]["text"]
            assert (rec["label"] == "OK") == (t == "synonym_ok")
    for sw, ff in [("eventually", "eventualmente"), ("sensible", "sensible"), ("embarrassed", "embarazada"),
                   ("library", "librería"), ("decade", "декада"), ("sympathetic", "симпатичный"),
                   ("accurate", "аккуратный"), ("artist", "артист"), ("realize", "realizar"), ("carpet", "carpeta"),
                   ("fabric", "fábrica"), ("family", "фамилия"), ("lecture", "lectura"), ("record", "рекорд")]:
        assert looks_alike(sw, ff) >= 0.5, (sw, ff, looks_alike(sw, ff))
    for sw, w in [("actually", "realmente"), ("magazine", "журнал"), ("library", "biblioteca"), ("bill", "factura")]:
        assert looks_alike(sw, w) < 0.5, (sw, w, looks_alike(sw, w))
    for t in TYPES:  # every prompt template is free of integrity-list words
        for lp in ("en-es", "en-ru"):
            build_messages({"lp": lp, "src": "A short clean sentence here.", "mt": "x"}, t, 1)
    assert FORBIDDEN_RE.search("They file a tax Return late") and FORBIDDEN_RE.search("la devolución")
    assert FORBIDDEN_RE.search("lavar los parches") and not FORBIDDEN_RE.search("la lavadora nueva")
    assert parse_verdicts('```json\n{"1": "same", "2": "CHANGED", "3": "maybe", "4": "broken"}\n```', 5) == \
        ["SAME", "CHANGED", None, "BROKEN", None]
    two = '```json\n{"old": "a", "new": "a"}\n```\n*Wait, same word.*\n```json\n{"old": "a", "new": "b"}\n```'
    assert parse_edit(two) == ({"old": "a", "new": "b"}, None)
    assert parse_edit(two + "\n*Hmm, maybe not.*")[1] == "comment_after_json"
    assert parse_edit('{"skip": true}') == ({"skip": True}, None) and parse_edit("none")[1] == "no_json"
    assert change_span("a b c d", "a x y d") == (["b", "c"], ["x", "y"])
    assert plan_counts("en-es:train:*=7,*:test:*=3,en-es:train:false_friend=9", ["en-es", "en-ru"], ["train", "test"]) \
        == {**{("en-es", "train", t): 7 for t in TYPES}, ("en-es", "train", "false_friend"): 9,
            **{(lp, "test", t): 3 for lp in ("en-es", "en-ru") for t in TYPES}}
    print(f"self-test OK ({len(cases)} validation cases)")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", default=DEFAULT_PLAN, help="comma-separated LP:SPLIT:TYPE=N rules (globs; last match wins)")
    ap.add_argument("--lps", nargs="+", default=["en-es", "en-ru"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    ap.add_argument("--phase", default="all", choices=["all", "gen", "verify"])
    ap.add_argument("--no-verify", action="store_true", help="write code-validated rows without the Gemma check")
    ap.add_argument("--max-per-seg-type", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--min-src-words", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--total-cap", type=float, default=0.90, help="USD cap on everything in the sense cache")
    ap.add_argument("--gen-cap", type=float, default=0.70,
                    help="generation stops once the sense cache holds this much spend (room for verification)")
    ap.add_argument("--max-requests", type=int, default=20000)
    ap.add_argument("--base-url", default=DO_BASE_URL)
    ap.add_argument("--cache", type=Path, default=CACHE_PATH)
    ap.add_argument("--out-dir", type=Path, default=SYNTH_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", type=int, default=1)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    cache = Cache(args.cache)
    counts = plan_counts(args.plan, args.lps, args.splits)
    stories = {story(d) for d in realdoc_docs()}
    pools, natural, jobs = {}, {}, {}
    for lp, split in sorted({(lp, s) for lp, s, _ in counts}):
        pools[(lp, split)], st = parent_pool(lp, split, args, stories)
        natural[(lp, split)] = natural_mts(lp, split)
        print(f"pool {lp} {split}: {dict(st)}")
    for (lp, split, t), n in counts.items():
        jobs[(lp, split, t)] = select_jobs(pools[(lp, split)], t, n, args)
        if len(jobs[(lp, split, t)]) < n:
            print(f"  {lp} {split} {t}: only {len(jobs[(lp, split, t)])} of {n} jobs available")

    if args.phase in ("all", "gen"):
        groups = [[j for j in js if cache.get(j.key) is None] for js in jobs.values()]
        todo = [j for batch in itertools.zip_longest(*groups) for j in batch if j is not None]
        n_all = sum(len(js) for js in jobs.values())
        print(f"generation: {n_all} jobs, {n_all - len(todo)} cached, {len(todo)} need a call")
        if args.dry_run:
            for j in (todo or [j for js in jobs.values() for j in js])[:args.show]:
                print(f"\n--- {j.parent['id']} [{j.synth_type}] max_completion_tokens={j.max_tokens}\n"
                      f"{j.messages[0]['content']}\n")
        call_all(todo, cache, args, "generation", min(args.gen_cap, args.total_cap))

    results: dict[tuple[str, str], dict[str, list[tuple[int, dict]]]] = collections.defaultdict(dict)
    rejects: dict[str, list[dict]] = collections.defaultdict(list)
    for (lp, split, t), js in jobs.items():
        acc, rej = collect(js, cache, natural[(lp, split)])
        results[(lp, split)][t] = acc
        rejects[split] += rej

    verify = not args.no_verify and args.phase != "gen"
    vjobs = {key: verify_jobs(*key, acc, args) for key, acc in results.items()} if verify else {}
    if verify:
        todo = [j for vs in vjobs.values() for j, _ in vs if cache.get(j.key) is None]
        print(f"verification: {sum(len(v) for v in vjobs.values())} batches, {len(todo)} need a call")
        if args.dry_run and todo:
            print(f"\n--- {todo[0].parent['id']}\n{todo[0].messages[0]['content']}\n")
        call_all(todo, cache, args, "verification", args.total_cap)

    report: dict = {}
    for split in args.splits:
        rows: list[dict] = []
        for lp in args.lps:
            if (lp, split) not in results:
                continue
            verdict: dict[str, str | None] = {}
            unverified: set[str] = set()
            for job, recs in vjobs.get((lp, split), []):
                entry = cache.get(job.key)
                if entry is None:
                    unverified.update(r["id"] for r in recs)
                    continue
                for r, v in zip(recs, parse_verdicts(entry["reply"], len(recs))):
                    verdict[r["id"]] = v
            for t in TYPES:
                if t not in results[(lp, split)]:
                    continue
                stat = collections.Counter(jobs=len(jobs[(lp, split, t)]), replied=0)
                stat["replied"] = sum(cache.get(j.key) is not None for j in jobs[(lp, split, t)])
                stat["code_ok"] = len(results[(lp, split)][t])
                for _, rec in results[(lp, split)][t]:
                    if not verify:
                        rows.append(rec)
                        continue
                    if rec["id"] in unverified:
                        stat["unverified"] += 1
                        continue
                    v = verdict.get(rec["id"])
                    agree = v == EXPECT[rec["label"]]
                    rec["synth_verify"] = {"model": DO_MODEL, "verdict": v, "expected": EXPECT[rec["label"]],
                                           "agree": agree}
                    if agree:
                        rows.append(rec)
                        stat["kept"] += 1
                    else:
                        why = {None: "verify_missing", "BROKEN": "verify_broken"}.get(v, "verify_disagree")
                        stat[why] += 1
                        rejects[split].append({"parent": rec["synth_parent"], "lp": lp, "synth_type": t,
                                               "reason": why, "mt": None, "record": rec})
                if not verify:
                    stat["kept"] = stat["code_ok"]
                rej = collections.Counter(r["reason"] for r in rejects[split] if r["lp"] == lp and r["synth_type"] == t)
                report[f"{lp} {split} {t}"] = {**stat, "rejects": dict(rej.most_common())}
                print(f"== {lp} {split} {t}: {dict(stat)} rejects {dict(rej.most_common())}")
        if not rows and not rejects[split]:
            continue
        for lp in args.lps:
            assert_hygiene(rows, lp, split)
        for r in rows:
            assert not FORBIDDEN_RE.search(r["src"]) and not FORBIDDEN_RE.search(r["mt"]), r["id"]
            assert story(r["doc"]) not in stories, r["id"]
        rows.sort(key=lambda r: r["id"])
        write_jsonl(args.out_dir / f"sense.{split}.jsonl", rows)
        write_jsonl(args.out_dir / f"sense.{split}.rejects.jsonl", rejects[split])
        print(f"wrote {args.out_dir / f'sense.{split}.jsonl'} ({len(rows)}) and rejects ({len(rejects[split])})")
    n, tin, tout, spent = cache_spend(cache)
    print(f"sense cache: {n} replies, {tin} input + {tout} output tokens, ${spent:.4f} at list price "
          f"(${PRICE_IN}/M in, ${PRICE_OUT}/M out)")
    (args.out_dir / "sense.report.json").write_text(json.dumps(
        {"plan": args.plan, "verify": verify, "spend": {"calls": n, "tokens_in": tin, "tokens_out": tout,
                                                        "usd": round(spent, 4)}, "types": report},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
