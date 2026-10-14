"""Synthetic meaning errors from Gemma 4 on DigitalOcean serverless inference: ONE controlled
edit of a clean MT segment, validated by code before it is kept.

usage: uv run python scripts/synth_llm.py [--lps en-ru en-es] [--splits train test]
           [--types neg_remove neg_insert omission addition ok_paraphrase] [--per-type 20]
           [--max-requests 100] [--max-cost 0.50] [--max-tokens 400] [--workers 4]
           [--dry-run] [--cache-only] [--self-test]

Edit types (label):
  neg_remove     drop/flip one negation the source has          (Major, span category mistranslation)
  neg_insert     negate one affirmative statement               (Major, mistranslation)
  omission       delete one clause/phrase of >= 3 words         (Major, omission; zero-width span)
  addition       insert 3-8 words of unsupported content        (Major, addition)
  ok_paraphrase  reword with the same meaning, digits, negation (OK)

Every reply is checked by code (see `validate`): the negation-marker count must go down/up
for neg_remove/neg_insert and stay equal otherwise; an omission must remove >= 3 words and
>= 10% of the characters without adding any; an addition must add words without removing
any; digits must be unchanged except that an omission may drop some; a paraphrase must
keep digits, negations, names and placeholders. Rejected replies go to
data/synth/llm.<split>.rejects.jsonl with the reason.

Split hygiene: parents come from secondread.synth.source_pool(lp, split), i.e. clean units
of the same split (train docs -> train synth, test docs -> held-out synth test).

Cost and safety: replies are cached in data/synth/cache/llm_cache.jsonl (keyed by model,
messages and sampling params), so re-runs never pay twice. Before any call the script prints
a worst-case estimate (input tokens estimated as chars/2.5, output as max_completion_tokens)
at $0.18/M input and $0.50/M output, trims the job list to --max-requests and --max-cost, and
stops submitting once actual spend plus in-flight worst case reaches --max-cost.
--dry-run prints the plan and prompts and needs no key.

Only public WMT text is sent to the API (never user XLIFF): DO has no EU data residency
(private prep notes, 03-verification-reports.md, DigitalOcean section 3).

Terms (checked 2026-10-14, docs/LICENSES-NOTES.md): Gemma 4 is Apache-2.0 and is not covered
by the Gemma Terms of Use, and DO's Service Terms, TOS and AUP do not restrict training on
outputs. So llm.train.jsonl may be used for training. Keep the synthetic/synth_* provenance
fields. Re-check before switching to a different generator (Gemma <= 3 would make the LoRA a
"Model Derivative" with pass-through obligations).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import difflib
import hashlib
import itertools
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secondread.synth import (SYNTH_DIR, assert_hygiene, digit_string, error_span, make_record,  # noqa: E402
                              negation_count, read_jsonl, rng_for, segment_key, shift_spans, source_pool,
                              target_lang, write_jsonl)

# Endpoint and model id: private prep notes, 03-verification-reports.md, DigitalOcean sections 1 and 3 (VERIFIED).
DO_BASE_URL = "https://inference.do-ai.run/v1"
DO_MODEL = "gemma-4-31B-it"
PRICE_IN, PRICE_OUT = 0.18, 0.50  # USD per 1M tokens (DO pricing page; cached input is cheaper)
GENERATOR = "llm-gemma4-v1"
CACHE_PATH = SYNTH_DIR / "cache" / "llm_cache.jsonl"

TYPES = ("neg_remove", "neg_insert", "omission", "addition", "ok_paraphrase")
LABEL = {t: ("OK" if t.startswith("ok_") else "Major") for t in TYPES}
LANG_NAME = {"ru": "Russian", "es": "Latin American Spanish"}
NEG_WORDS = {"ru": "«не», «нет», «никогда», «ни», «ничего»", "es": "«no», «nunca», «nadie», «nada», «ni»"}
NEG_FLIP = {"ru": "«никогда» -> «всегда»", "es": "«nunca» -> «siempre»"}
NEG_ADD = {"ru": "«не» before a verb", "es": "«no» before a verb"}

INSTRUCTIONS = {
    "neg_remove": ("Remove ONE negation so that this part of the translation now states the opposite of "
                   "the source: delete a negation word ({neg}) or replace it with its positive counterpart "
                   "(e.g. {flip}). Change only the words that grammar then requires."),
    "neg_insert": ("Negate ONE affirmative statement so that this part of the translation contradicts the "
                   "source, e.g. add {add}. Change only the words that grammar then requires."),
    "omission": ("Delete ONE clause or phrase of at least three words that carries meaningful content of the "
                 "source, so that this information is missing from the translation. Do not add, reorder or "
                 "rephrase anything; adjust punctuation at the cut only if needed."),
    "addition": ("Insert ONE short phrase of 3 to 8 words that adds a plausible detail NOT present in the "
                 "source (e.g. a time, place, reason, qualifier or extra claim). The phrase must not contain "
                 "digits or a negation. Do not remove or rephrase anything."),
    "ok_paraphrase": ("Rewrite the translation with different wording or word order while keeping EXACTLY the "
                      "meaning of the source: same facts, same numbers written the same way, same names, same "
                      "negations, same tense. Change at least two words. The result must be a correct, natural "
                      "translation of the source."),
}
PROMPT = """You are creating test data for a checker that detects meaning errors in machine translation.

Source (English): {src}
Translation ({lang}): {mt}

Task: {instruction}

Rules:
- Make exactly this one edit and keep everything else identical (same punctuation, quotes, spacing).
- Keep every number, name, URL, tag and placeholder (such as PRS_ORG or #NAME#) exactly as written{keep_note}.
- The result must stay fluent, grammatical {lang}, so the edit cannot be spotted from grammar alone.
- Output the complete edited translation, not only the changed part.

Answer with JSON only: {{"edited": "<full edited translation>"}}"""


# --------------------------------------------------------------------------- jobs

@dataclass
class Job:
    parent: dict
    synth_type: str
    messages: list[dict]
    max_tokens: int
    temperature: float

    @property
    def key(self) -> str:
        payload = {"model": DO_MODEL, "messages": self.messages, "temperature": self.temperature,
                   "max_completion_tokens": self.max_tokens}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    @property
    def est_in(self) -> int:
        return est_tokens(self.messages[0]["content"]) + 8  # + chat template overhead

    @property
    def worst_cost(self) -> float:
        return cost(self.est_in, self.max_tokens)


def est_tokens(text: str) -> int:
    """Conservative token estimate. No Gemma tokenizer offline; Cyrillic runs ~2.5-3.5 chars/token."""
    return math.ceil(len(text) / 2.5)


def cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6


def build_messages(parent: dict, synth_type: str) -> list[dict]:
    lang = target_lang(parent["lp"])
    instruction = INSTRUCTIONS[synth_type].format(neg=NEG_WORDS[lang], flip=NEG_FLIP[lang], add=NEG_ADD[lang])
    keep_note = " (unless they are inside the part you delete)" if synth_type == "omission" else ""
    text = PROMPT.format(src=parent["src"], mt=parent["mt"], lang=LANG_NAME[lang], instruction=instruction,
                         keep_note=keep_note)
    # A single user turn: no system role, which not every chat template (Gemma's included) has.
    return [{"role": "user", "content": text}]


CLAUSE_RE = re.compile(r"[,;:—–]|\s(?:и|а|но|что|который|которая|когда|y|pero|que|cuando|porque)\s", re.IGNORECASE)


def eligible(parent: dict, synth_type: str) -> bool:
    lang, mt, src = target_lang(parent["lp"]), parent["mt"], parent["src"]
    words = len(re.findall(r"\w+", mt))
    neg_mt, neg_src = negation_count(mt, lang), negation_count(src, "en")
    if synth_type == "neg_remove":
        return neg_mt >= 1 and neg_src >= 1           # the source is negated, so removing it is an error
    if synth_type == "neg_insert":
        return neg_mt == 0 and neg_src == 0 and words >= 4
    if synth_type == "omission":
        return words >= 10 and bool(CLAUSE_RE.search(mt))
    return words >= 5                                   # addition, ok_paraphrase


def select_jobs(lp: str, split: str, args) -> list[Job]:
    """Deterministic, split-local selection: one edit per parent, at most --max-per-seg edits per
    source segment, types filled round-robin (scarcest first) so none is starved of segments."""
    pool, _ = source_pool(lp, split)
    pool = [p for p in pool if len(p["mt"]) <= args.max_chars and len(p["src"]) <= args.max_chars]
    queues = {t: sorted((p for p in pool if eligible(p, t)), key=lambda p: rng_for(args.seed, t, p["id"]).random())
              for t in args.types}
    queues = {t: collections.deque(q) for t, q in sorted(queues.items(), key=lambda kv: len(kv[1]))}
    per_seg: collections.Counter = collections.Counter()
    taken: collections.Counter = collections.Counter()
    used: set[str] = set()
    jobs = []
    while any(queues.values()):
        for t, queue in queues.items():
            while queue and taken[t] < args.per_type:
                p = queue.popleft()
                if p["id"] in used or per_seg[segment_key(p)] >= args.max_per_seg:
                    continue
                used.add(p["id"])
                per_seg[segment_key(p)] += 1
                taken[t] += 1
                max_tokens = min(args.max_tokens, max(64, int(est_tokens(p["mt"]) * 1.5) + 40))
                jobs.append(Job(p, t, build_messages(p, t), max_tokens, args.temperature))
                break
            if taken[t] >= args.per_type:
                queue.clear()
    return sorted(jobs, key=lambda j: (j.synth_type, j.parent["id"]))


# --------------------------------------------------------------------------- cache

class Cache:
    """Append-only JSONL cache of API replies, safe to share between worker threads."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {e["key"]: e for e in read_jsonl(path)} if path.exists() else {}

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, entry: dict) -> None:
        with self.lock:
            self.data[entry["key"]] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- API

class Budget:
    """Spend guard: actual cost of finished calls + worst case of calls in flight <= limit."""

    def __init__(self, limit: float):
        self.limit, self.spent, self.reserved = limit, 0.0, 0.0
        self.lock = threading.Lock()

    def reserve(self, amount: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + amount > self.limit:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self.lock:
            self.reserved -= reserved
            self.spent += actual


def make_client(base_url: str):
    from openai import OpenAI  # imported lazily: --dry-run and --self-test need no SDK or key

    key = os.environ.get("DO_MODEL_KEY")
    if not key:
        raise SystemExit("DO_MODEL_KEY is not set (create a model access key limited to gemma-4-31B-it at "
                         "https://cloud.digitalocean.com/model-studio/manage-keys). Use --dry-run to preview.")
    # The SDK retries 429/5xx with backoff and honours Retry-After.
    return OpenAI(base_url=base_url, api_key=key, max_retries=4, timeout=120)


_headers_logged = threading.Event()


def call(client, job: Job) -> dict:
    raw = client.chat.completions.with_raw_response.create(
        model=DO_MODEL, messages=job.messages, temperature=job.temperature,
        max_completion_tokens=job.max_tokens)
    resp = raw.parse()
    if not _headers_logged.is_set():  # tokens-per-day limit for this account (02-corrections.md #26)
        _headers_logged.set()
        limits = {k: v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
        print(f"  rate-limit headers: {limits or 'none'}")
    choice = resp.choices[0]
    extra = choice.message.model_extra or {}
    usage = resp.usage
    return {
        "key": job.key, "model": resp.model or DO_MODEL, "synth_type": job.synth_type, "parent": job.parent["id"],
        "messages": job.messages, "params": {"temperature": job.temperature, "max_completion_tokens": job.max_tokens},
        "reply": choice.message.content or "", "finish_reason": choice.finish_reason,
        # 02-corrections.md #28: check whether Gemma 4 on DO emits a separate reasoning field
        "has_reasoning": bool(extra.get("reasoning_content") or extra.get("reasoning")),
        "usage": {"prompt_tokens": usage.prompt_tokens if usage else None,
                  "completion_tokens": usage.completion_tokens if usage else None},
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_jobs(jobs: list[Job], cache: Cache, args) -> dict:
    import openai

    client = make_client(args.base_url)
    budget = Budget(args.max_cost)
    stats = collections.Counter()
    todo = iter(jobs)
    in_flight: dict[cf.Future, tuple[Job, float]] = {}
    consecutive_errors, fatal = 0, None

    def finish(fut: cf.Future) -> str | None:
        """Cache a finished call and book its cost; returns a fatal error message, if any."""
        nonlocal consecutive_errors
        job, reserved = in_flight.pop(fut)
        try:
            entry = fut.result()
        except openai.AuthenticationError as e:
            budget.settle(reserved, 0.0)
            return f"authentication failed ({e.status_code}): check DO_MODEL_KEY"
        except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
            budget.settle(reserved, 0.0)  # failed calls are not billed for output; input cost unknown
            stats["api_error"] += 1
            consecutive_errors += 1
            print(f"  API error on {job.parent['id']} ({job.synth_type}): {type(e).__name__}: {e}")
            return "5 API errors in a row, aborting (see messages above)" if consecutive_errors >= 5 else None
        consecutive_errors = 0
        u = entry["usage"]
        budget.settle(reserved, cost(u["prompt_tokens"] or job.est_in, u["completion_tokens"] or job.max_tokens))
        cache.put(entry)
        stats["called"] += 1
        stats["tokens_in"] += u["prompt_tokens"] or 0
        stats["tokens_out"] += u["completion_tokens"] or 0
        stats["with_reasoning"] += entry["has_reasoning"]
        if stats["called"] % 25 == 0:
            print(f"  {stats['called']} calls, ${budget.spent:.4f}")
        return None

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while fatal is None:
            while len(in_flight) < args.workers:
                job = next(todo, None)
                if job is None:
                    break
                if not budget.reserve(job.worst_cost):
                    stats["skipped_budget"] += 1 + sum(1 for _ in todo)
                    print(f"  budget guard: stopping at ${budget.spent:.4f} spent (limit ${args.max_cost})")
                    break
                in_flight[pool.submit(call, client, job)] = (job, job.worst_cost)
            if not in_flight:
                break
            done, _ = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                fatal = finish(fut) or fatal
        # on a fatal error, still cache the replies of calls already in flight (they are paid for)
        for fut in cf.as_completed(list(in_flight)):
            finish(fut)
    stats["spent_usd"] = round(budget.spent, 6)
    if fatal:
        print(f"api: {dict(stats)}")
        raise SystemExit(fatal)
    return dict(stats)


# --------------------------------------------------------------------------- validation

PROTECTED_RE = re.compile(r"\b[A-Z]{2,}_[A-Z]{2,}\b|#\w+#|<[^<>\s]+>|https?://\S+|@\w+")


def parse_reply(text: str) -> str | None:
    """The "edited" field of the first JSON object in the reply (tolerates code fences and preambles)."""
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip())
    i = t.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[i:])
    except json.JSONDecodeError:
        return None
    value = obj.get("edited") if isinstance(obj, dict) else None
    return value.strip() if isinstance(value, str) else None


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def word_diff(old: str, new: str) -> tuple[int, int]:
    """(words deleted, words inserted) between two texts."""
    sm = difflib.SequenceMatcher(None, words(old), words(new), autojunk=False)
    deleted = inserted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            deleted += i2 - i1
            inserted += j2 - j1
    return deleted, inserted


def script_ok(old: str, new: str, lang: str) -> bool:
    letters = lambda s: [c for c in s if c.isalpha()]  # noqa: E731
    cyr = lambda s: sum("\u0400" <= c <= "\u04ff" for c in letters(s)) / max(1, len(letters(s)))  # noqa: E731
    return cyr(new) >= cyr(old) - 0.15 if lang == "ru" else cyr(new) <= cyr(old) + 0.01


def validate(parent: dict, synth_type: str, new: str | None) -> str | None:
    """None if the edit is the one we asked for, else the rejection reason."""
    old, lang = parent["mt"], target_lang(parent["lp"])
    if not new:
        return "no_json"
    if " ".join(new.split()) == " ".join(old.split()):
        return "unchanged"
    if "\n" in new and "\n" not in old:
        return "multiline"
    if not 0.3 <= len(new) / len(old) <= 2.0:
        return "length_ratio"
    if not script_ok(old, new, lang):
        return "wrong_script"
    kept = collections.Counter(PROTECTED_RE.findall(new))
    need = collections.Counter(PROTECTED_RE.findall(old))
    if (kept - need) if synth_type == "omission" else (kept != need):  # an omission may drop some
        return "placeholder_changed"
    neg_old, neg_new = negation_count(old, lang), negation_count(new, lang)
    d_old, d_new = digit_string(old), digit_string(new)
    deleted, inserted = word_diff(old, new)
    if synth_type == "neg_remove":
        if neg_new >= neg_old:
            return "negation_not_removed"
        if deleted + inserted > 6:
            return "edit_not_local"
    elif synth_type == "neg_insert":
        if neg_new <= neg_old:
            return "negation_not_added"
        if deleted + inserted > 6:
            return "edit_not_local"
    elif synth_type == "omission":
        if deleted < 3 or len(new) > 0.9 * len(old):
            return "omission_too_small"
        if inserted > 1:
            return "omission_added_words"
        if collections.Counter(d_new) - collections.Counter(d_old):
            return "digits_added"
        if neg_new > neg_old:
            return "negation_added"
        return None  # an omission may drop digits together with the clause
    elif synth_type == "addition":
        if inserted < 2 or len(new) < len(old) + 8:
            return "addition_too_small"
        if deleted > 1:
            return "addition_removed_words"
        if neg_new != neg_old:
            return "negation_changed"
    elif synth_type == "ok_paraphrase":
        if neg_new != neg_old:
            return "negation_changed"
        if deleted + inserted < 2:
            return "paraphrase_trivial"
        if not 0.7 <= len(new) / len(old) <= 1.4:
            return "length_ratio"
    if d_new != d_old:
        return "digits_changed"
    return None


def changed_region(old: str, new: str) -> tuple[int, int, int]:
    """(start, old_end, new_end): the minimal differing region (common prefix/suffix removed),
    widened so that it never cuts a word of `new` in half."""
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    s = 0
    while s < min(len(old), len(new)) - p and old[-1 - s] == new[-1 - s]:
        s += 1
    old_end, new_end = len(old) - s, len(new) - s
    while 0 < p < len(new) and new[p - 1].isalnum() and new[p].isalnum():
        p -= 1
    while 0 < new_end < len(new) and new[new_end - 1].isalnum() and new[new_end].isalnum():
        new_end += 1
        old_end += 1
    return p, old_end, new_end


def error_bounds(new: str, start: int, end: int, synth_type: str) -> tuple[int, int]:
    """Where to put the Major span: the inserted text without edge spaces; for a pure deletion,
    the next word (the statement that lost its negation), or a zero-width point for an omission,
    as WMT/Unbabel annotators mark omissions."""
    while start < end and new[start].isspace():
        start += 1
    while end > start and new[end - 1].isspace():
        end -= 1
    if (start == end and synth_type != "omission") or synth_type == "neg_insert":
        m = re.compile(r"\w+").search(new, end)  # "не" + "объявила": mark the negated word too
        if m and (start == end or m.start() - end <= 1):
            start, end = (m.start() if start == end else start), m.end()
    return start, end


def build_record(parent: dict, synth_type: str, new: str, model: str) -> dict:
    start, old_end, new_end = changed_region(parent["mt"], new)
    spans = shift_spans(parent["spans"], start, old_end, new_end - start)
    if LABEL[synth_type] == "Major":
        spans.append(error_span(new, *error_bounds(new, start, new_end, synth_type), synth_type))
        spans.sort(key=lambda sp: (sp["start"], sp["end"]))
    rec = make_record(parent, mt=new, label=LABEL[synth_type], synth_type=synth_type, generator=GENERATOR,
                      edit={"start": start, "end": new_end, "text": new[start:new_end],
                            "orig": parent["mt"][start:old_end]}, spans=spans)
    rec["synth_model"] = model
    return rec


def collect(jobs: list[Job], cache: Cache) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for job in jobs:
        entry = cache.get(job.key)
        if entry is None:
            continue
        new = parse_reply(entry["reply"])
        reason = "truncated" if entry.get("finish_reason") == "length" else validate(job.parent, job.synth_type, new)
        if reason:
            rejected.append({"parent": job.parent["id"], "lp": job.parent["lp"], "synth_type": job.synth_type,
                             "reason": reason, "mt": job.parent["mt"], "edited": new, "reply": entry["reply"]})
        else:
            accepted.append(build_record(job.parent, job.synth_type, new, entry.get("model", DO_MODEL)))
    return accepted, rejected


# --------------------------------------------------------------------------- self-test

def self_test() -> None:
    """Offline checks of the validators on hand-made replies (no API, no data files)."""
    base = {"year": 0, "system": "test", "doc": "test", "domain": "test", "rater": "test", "label": "OK"}
    ru = {**base, "id": "t-ru", "lp": "en-ru", "src": "I did not see the 2 cars, but my friend saw them yesterday.",
          "mt": "Я не видел 2 машины, но мой друг видел их вчера.", "spans": []}
    es = {**base, "id": "t-es", "lp": "en-es", "src": "The museum will open its new hall in the city center.",
          "mt": "El museo abrirá su nueva sala en el centro de la ciudad.", "spans": []}
    cases = [
        (ru, "neg_remove", "Я видел 2 машины, но мой друг видел их вчера.", None),
        (ru, "neg_remove", "Я не видел 3 машины, но мой друг видел их вчера.", "negation_not_removed"),
        (ru, "neg_insert", "Я не видел 2 машины, но мой друг не видел их вчера.", None),
        (ru, "omission", "Я не видел 2 машины.", None),
        (ru, "omission", "Я не видел 2 машины, но мой друг их видел.", "omission_too_small"),
        (ru, "addition", "Я не видел 2 машины, но мой друг видел их вчера возле старого вокзала.", None),
        (ru, "addition", "Я не видел 5 машин, но мой друг видел их вчера возле вокзала.", "addition_removed_words"),
        (ru, "ok_paraphrase", "Я не заметил 2 машины, а мой друг вчера их видел.", None),
        (ru, "ok_paraphrase", "Я заметил 2 машины, а мой друг вчера их видел.", "negation_changed"),
        (ru, "ok_paraphrase", "I did not see the 2 cars.", "wrong_script"),
        (es, "neg_insert", "El museo no abrirá su nueva sala en el centro de la ciudad.", None),
        (es, "addition", "El museo abrirá su nueva sala de arte moderno en el centro de la ciudad.", None),
        (es, "ok_paraphrase", "El museo abrirá su nueva sala en el centro de la ciudad!", "paraphrase_trivial"),
        (es, "ok_paraphrase", "El museo abrirá su nueva sala en el centro de la ciudad, sin falta.",
         "negation_changed"),
        (es, "ok_paraphrase", "El museo inaugurará su sala nueva en pleno centro de la ciudad.", None),
    ]
    for parent, t, new, want in cases:
        got = validate(parent, t, new)
        assert got == want, (t, new, got, want)
        if got is None:
            rec = build_record(parent, t, new, "test")
            for sp in rec["spans"]:
                assert rec["mt"][sp["start"]:sp["end"]] == sp["text"], sp
    assert parse_reply('```json\n{"edited": "Да."}\n```') == "Да."
    assert parse_reply('Sure! {"edited": "Sí."} Hope it helps') == "Sí."
    assert parse_reply("no json here") is None
    rec = build_record(ru, "omission", "Я не видел 2 машины.", "test")
    assert rec["synth_edit"]["orig"] == ", но мой друг видел их вчера" and rec["spans"][0]["category"] == "omission"
    assert rec["spans"][0]["start"] == rec["spans"][0]["end"]  # zero-width, at the cut
    rec = build_record(ru, "neg_remove", "Я видел 2 машины, но мой друг видел их вчера.", "test")
    assert rec["synth_edit"]["orig"] == "не " and rec["spans"][0]["text"] == "видел", rec
    rec = build_record(ru, "neg_insert", "Я не видел 2 машины, но мой друг не видел их вчера.", "test")
    assert rec["spans"][0]["text"] == "не видел", rec["spans"]
    rec = build_record(es, "addition", "El museo abrirá su nueva sala de arte moderno en el centro de la ciudad.",
                       "test")
    assert rec["spans"][0]["text"] == "de arte moderno", rec["spans"]
    print(f"self-test OK ({len(cases)} validation cases)")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lps", nargs="+", default=["en-ru", "en-es"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--types", nargs="+", default=list(TYPES), choices=TYPES)
    ap.add_argument("--per-type", type=int, default=20, help="jobs per (lp, split, type)")
    ap.add_argument("--max-per-seg", type=int, default=4, help="edits per source segment, all types together")
    ap.add_argument("--max-chars", type=int, default=600, help="skip longer segments (cost)")
    ap.add_argument("--max-requests", type=int, default=100, help="hard cap on API calls in this run")
    ap.add_argument("--max-cost", type=float, default=0.50, help="USD cap on this run (worst case)")
    ap.add_argument("--max-tokens", type=int, default=400, help="max_completion_tokens cap per call")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--base-url", default=os.environ.get("DO_BASE_URL", DO_BASE_URL))
    ap.add_argument("--cache", type=Path, default=CACHE_PATH)
    ap.add_argument("--out-dir", type=Path, default=SYNTH_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print plan and prompts, call nothing")
    ap.add_argument("--show", type=int, default=2, help="prompts to print with --dry-run")
    ap.add_argument("--cache-only", action="store_true", help="use cached replies only, call nothing")
    ap.add_argument("--self-test", action="store_true", help="run offline validator tests and exit")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    cache = Cache(args.cache)
    jobs = {(lp, split): select_jobs(lp, split, args) for split in args.splits for lp in args.lps}
    all_jobs = [j for js in jobs.values() for j in js]
    # pending calls, interleaved across (lp, split, type) so --max-requests/--max-cost trimming stays balanced
    groups: dict[tuple, list[Job]] = collections.defaultdict(list)
    for (lp, split), js in jobs.items():
        for job in js:
            if cache.get(job.key) is None:
                groups[(lp, split, job.synth_type)].append(job)
    todo = [job for batch in itertools.zip_longest(*groups.values()) for job in batch if job is not None]

    print(f"jobs: {len(all_jobs)} ({len(all_jobs) - len(todo)} cached, {len(todo)} need a call)")
    for (lp, split), js in jobs.items():
        print(f"  {lp} {split}: " + ", ".join(f"{t}={n}" for t, n in sorted(collections.Counter(
            j.synth_type for j in js).items())))
    if len(todo) > args.max_requests:
        print(f"  --max-requests {args.max_requests}: calling only the first {args.max_requests}")
        todo = todo[:args.max_requests]
    worst = sum(j.worst_cost for j in todo)
    if worst > args.max_cost:
        kept, acc = [], 0.0
        for j in todo:
            if acc + j.worst_cost > args.max_cost:
                break
            kept.append(j)
            acc += j.worst_cost
        print(f"  --max-cost ${args.max_cost}: worst case ${worst:.4f} too high, calling only {len(kept)}")
        todo, worst = kept, acc
    tin, tout = sum(j.est_in for j in todo), sum(j.max_tokens for j in todo)
    print(f"plan: {len(todo)} calls to {DO_MODEL} at {args.base_url}, ~{tin} input tokens (chars/2.5), "
          f"<= {tout} output tokens, worst case ${worst:.4f} "
          f"(${PRICE_IN}/M in, ${PRICE_OUT}/M out; typical cost is far lower)")

    if args.dry_run:
        for j in todo[:args.show] or all_jobs[:args.show]:
            print(f"\n--- {j.parent['id']} [{j.synth_type}] max_completion_tokens={j.max_tokens}\n"
                  f"{j.messages[0]['content']}")
        return
    if todo and not args.cache_only:
        stats = run_jobs(todo, cache, args)
        print(f"api: {stats}")

    for split in args.splits:
        accepted, rejected = [], []
        for lp in args.lps:
            acc, rej = collect(jobs[(lp, split)], cache)
            assert_hygiene(acc, lp, split)
            accepted += acc
            rejected += rej
            by = collections.Counter((r["synth_type"], "ok") for r in acc)
            by.update((r["synth_type"], r["reason"]) for r in rej)
            print(f"== {lp} {split}: accepted {len(acc)}, rejected {len(rej)} | " +
                  ", ".join(f"{t}:{k}={v}" for (t, k), v in sorted(by.items())))
        accepted.sort(key=lambda r: r["id"])
        write_jsonl(args.out_dir / f"llm.{split}.jsonl", accepted)
        write_jsonl(args.out_dir / f"llm.{split}.rejects.jsonl", rejected)
        print(f"wrote {args.out_dir / f'llm.{split}.jsonl'} ({len(accepted)}) and rejects ({len(rejected)})")


if __name__ == "__main__":
    main()
