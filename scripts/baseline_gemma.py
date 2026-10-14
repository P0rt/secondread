"""Gemma 4 31B baselines on DigitalOcean serverless inference (test G7).

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/baseline_gemma.py \
           --method direct|gemba SPLIT.jsonl --out results/gemma_<method>_<name>.jsonl \
           [--ids-from SCORED.jsonl | --sample N [--seed 0]] [--limit N] [--out-all FILE] \
           [--cap 1.20] [--concurrency 8] [--rpm 110] [--top-logprobs 10] [--dry-run] [--cache-only]

Both methods write scored jsonl in the format of scripts/probe_local.py, so scripts/report.py
reads them directly: one row per segment with id, label (gold), risk in [0, 1], doc and lp
(language pair; bootstrap grouping), plus method-specific fields. No source or MT text, and
no text quoted by the model, goes into the scored file: results/ is committed, WMT text is
not. Raw model replies stay in the cache (data/cache/, git-ignored).

Methods
-------
direct  The SecondRead classifier prompt, secondread.prompt.user_text() (one user turn, no
        system turn), temperature 0, max_completion_tokens 3, logprobs with --top-logprobs
        alternatives. A token is a label variant if its text, stripped of blanks and markup
        (*, _, `, quotes, trailing punctuation) and lower-cased, is "ok"/"okay" (OK) or starts
        with "major" or is a >= 3-char prefix of it (Major): "Major", " Major", "MAJOR" and
        "Major." all count, and the probabilities of the variants are summed.
        Gemma often answers "**Major**", so the first token is either a label or "**". The
        label mass is therefore marginalised over the first two positions:
            p(L) = p0(L) + p0(m) * p1(L | m)
        where p0 is the first position's distribution, m the markup token actually sampled
        there (only that path's continuation is observed) and p1 the second position's.
        When the first sampled token is itself a label, p(L) = p0(L) (the markup paths are
        assumed to split like the direct ones). risk = p(Major) / (p(Major) + p(OK)).
          status=both        both labels in the top-k lists (the normal case)
          status=major_only  OK is in no list: its logprob is bounded by the smallest listed
          status=ok_only     logprob at that position, and the bound is used (same for Major)
          status=text        no label in any list: risk 1/0 from the reply text
          status=unparsed    the reply starts with neither label: risk 0.5
gemba   GEMBA-MQM (Kocmi & Federmann 2023), few-shot prompt in prompts/gemba_mqm.txt
        (CC BY-SA 4.0, see that file's header), temperature 0, max_completion_tokens 256.
        The system turn is sent as a system message (--system native) or prepended to the
        first user turn (--system merge). The reply is parsed line by line into
        (severity, category) errors; the category is normalised to the GEMBA class names
        (accuracy/mistranslation, fluency/grammar, ...), so no quoted text is kept.
        Scoring follows GEMBA-MQM: critical 25, major 5, minor 1, "non-translation" always
        counts as critical, only the first 5 errors (critical first, then major, minor) count,
        and the penalty is capped at 25.
          risk      = GEMBA penalty over meaning-family errors only, / 25  (0 .. 1)
          risk_all  = GEMBA penalty over all errors, / 25 (= -GEMBA score / 25; --out-all
                      writes a scored file with risk = risk_all for comparison)
          pred      = "Major" iff there is at least one critical or major error in the
                      meaning family. Meaning family = the label rule of scripts/convert.py
                      for the Google MQM typology: accuracy/*, terminology/* and
                      non-translation. Fluency, style, locale convention and other are not.
          status    = ok | no_error_only (the whole reply is "no-error": no errors) |
                      no_levels (no "Critical:/Major:/Minor:" header and not a bare
                      "no-error": a parse failure, scored as no errors). Truncated replies
                      (finish_reason=length) are parsed as far as they go and counted.
                      See parse_gemba() for where the parser departs from GEMBA's.

Cost and safety
---------------
Every successful reply is cached in data/cache/gemma_baseline.jsonl, keyed by (method, id)
with a hash of the request; reruns with the same prompt are free. --cap is the budget for the
WHOLE track: spend already recorded in the cache counts against it. Before any call the script
projects the worst case (input tokens estimated as chars / 2.5 + overhead, output as
max_completion_tokens) at $0.18/M input and $0.50/M output and refuses to start if cache spend
plus that projection exceeds --cap (--trim keeps the prefix that fits). While running, each
call reserves its worst case and nothing is submitted once spend + reservations reach --cap.
Calls are async, at most --concurrency in flight, started at most --rpm per minute; 429 and
5xx are retried with backoff (Retry-After honoured, a 429 pauses all workers).
Only public WMT text is sent. The key is read from DO_MODEL_KEY and never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from secondread import metrics as M  # noqa: E402
from secondread.prompt import user_text  # noqa: E402

DO_BASE_URL = "https://inference.do-ai.run/v1"
DO_MODEL = "gemma-4-31B-it"
PRICE_IN, PRICE_OUT, PRICE_CACHED = 0.18, 0.50, 0.036  # USD per 1M tokens (DO pricing page)
CACHE_PATH = ROOT / "data" / "cache" / "gemma_baseline.jsonl"
GEMBA_PROMPT = ROOT / "prompts" / "gemba_mqm.txt"
LANG_NAME = {"en": "English", "ru": "Russian", "es": "Spanish", "de": "German"}
MAX_ATTEMPTS = 8


# --------------------------------------------------------------------------- prompts

def load_turns(path: Path) -> list[dict]:
    """Chat turns from a '@@@ role' file; '#' header lines before the first turn are skipped."""
    turns, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        m = re.fullmatch(r"@@@ (system|user|assistant)\n?", line)
        if m:
            cur = {"role": m.group(1), "content": ""}
            turns.append(cur)
        elif cur is not None:
            cur["content"] += line
        elif not line.startswith("#") and line.strip():
            raise SystemExit(f"{path}: text before the first '@@@' line that is not a '#' header: {line!r}")
    for t in turns:
        t["content"] = t["content"].rstrip("\n")
    if not turns or turns[-1]["role"] != "user":
        raise SystemExit(f"{path}: the last turn must be the user template")
    return turns


def gemba_messages(turns: list[dict], src: str, mt: str, lp: str, system: str) -> list[dict]:
    s, t = lp.split("-")
    fill = {"source_lang": LANG_NAME[s], "source_seg": src, "target_lang": LANG_NAME[t], "target_seg": mt}
    msgs = [dict(x) for x in turns]
    msgs[-1]["content"] = re.sub(r"\{(source_lang|source_seg|target_lang|target_seg)\}",
                                 lambda m: fill[m.group(1)], msgs[-1]["content"])  # one pass
    if system == "merge" and msgs[0]["role"] == "system":
        sys_text = msgs.pop(0)["content"]
        msgs[0] = {"role": "user", "content": sys_text + "\n\n" + msgs[0]["content"]}
    return msgs


def build_request(method: str, row: dict, args, turns) -> dict:
    if method == "direct":
        return {"model": DO_MODEL, "temperature": 0, "max_completion_tokens": args.direct_max_tokens,
                "logprobs": True, "top_logprobs": args.top_logprobs,
                "messages": [{"role": "user", "content": user_text(row["src"], row["mt"], row["lp"])}]}
    return {"model": DO_MODEL, "temperature": 0, "max_completion_tokens": args.gemba_max_tokens,
            "messages": gemba_messages(turns, row["src"], row["mt"], row["lp"], args.system)}


def request_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def est_tokens(text: str) -> int:
    """Conservative: no Gemma tokenizer offline; Cyrillic runs ~2.5-3.5 chars per token."""
    return math.ceil(len(text) / 2.5)


def worst_cost(body: dict) -> float:
    tin = sum(est_tokens(m["content"]) + 6 for m in body["messages"]) + 8
    return (tin * PRICE_IN + body["max_completion_tokens"] * PRICE_OUT) / 1e6


def usage_cost(u: dict) -> tuple[float, float]:
    """(cost at the list price, cost if cached prompt tokens get the cached rate)."""
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return ((pt * PRICE_IN + ct * PRICE_OUT) / 1e6,
            ((pt - cached) * PRICE_IN + cached * PRICE_CACHED + ct * PRICE_OUT) / 1e6)


# --------------------------------------------------------------------------- direct: logprobs

FILLER = " \t\r\n*_`\"'“”«»"


def label_of(tok: str) -> str | None:
    t = tok.strip(FILLER + ".,:;!?").lower()
    if not t:
        return None
    if t.startswith("major") or (len(t) >= 3 and "major".startswith(t)):
        return "Major"
    if t in ("ok", "okay"):
        return "OK"
    return None


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def label_mass(pos: dict) -> tuple[dict[str, float | None], float]:
    """Per-label probability at one position (None if no variant is listed) and the bound
    (smallest listed probability) that any unlisted token's probability is below."""
    cands = dict(pos["top"])  # token -> logprob; the sampled token is usually listed too
    cands.setdefault(pos["token"], pos["logprob"])
    by_label: dict[str, list[float]] = collections.defaultdict(list)
    for t, lp in cands.items():
        lab = label_of(t)
        if lab:
            by_label[lab].append(lp)
    mass = {lab: (math.exp(logsumexp(by_label[lab])) if by_label[lab] else None) for lab in ("Major", "OK")}
    return mass, math.exp(min(cands.values()))


def score_direct(entry: dict) -> dict:
    positions = entry.get("logprobs") or []
    text = entry.get("content") or ""
    out = {"top": "".join(p["token"] for p in positions[:2]) if positions else None}
    if positions:
        m0, floor0 = label_mass(positions[0])
        paths = [(1.0, m0, floor0)]
        if not positions[0]["token"].strip(FILLER) and len(positions) > 1:  # sampled markup, e.g. "**"
            m1, floor1 = label_mass(positions[1])
            paths.append((math.exp(positions[0]["logprob"]), m1, floor1))
        found = {lab: any(m[lab] is not None for _, m, _ in paths) for lab in ("Major", "OK")}
        if any(found.values()):
            p = {lab: sum(w * (m[lab] if m[lab] is not None else (0.0 if found[lab] else fl))
                          for w, m, fl in paths) for lab in ("Major", "OK")}
            status = "both" if all(found.values()) else ("major_only" if found["Major"] else "ok_only")
            return out | {"risk": p["Major"] / (p["Major"] + p["OK"]), "p_major": p["Major"], "p_ok": p["OK"],
                          "status": status}
    m = re.match(r"[\W_]*(major|ok)\b", text, re.IGNORECASE)
    if m:
        r = 1.0 if m.group(1).lower() == "major" else 0.0
        return out | {"risk": r, "p_major": None, "p_ok": None, "status": "text"}
    return out | {"risk": 0.5, "p_major": None, "p_ok": None, "status": "unparsed"}


# --------------------------------------------------------------------------- gemba: parse + score

HEADER_RE = re.compile(r"^(critical|major|minor)(?:\s+errors?)?\s*:\s*(.*)$")
WEIGHT = {"critical": 25, "major": 5, "minor": 1}
SUBCLASSES = {
    "accuracy": ("addition", "mistranslation", "omission", "untranslated text"),
    "fluency": ("character encoding", "grammar", "inconsistency", "punctuation", "register", "spelling"),
    "locale convention": ("currency", "date", "name", "telephone", "time"),
    "style": ("awkward",),
    "terminology": ("inappropriate for context", "inconsistent use"),
}
BARE_SUB = {"addition": "accuracy", "mistranslation": "accuracy", "omission": "accuracy",
            "untranslated": "accuracy", "grammar": "fluency", "spelling": "fluency",
            "punctuation": "fluency", "register": "fluency", "awkward": "style"}


def canonical(line: str) -> str:
    """GEMBA class name from the category part of an error line ('accuracy/omission - "x"')."""
    head = re.split(r'\s+-\s+|\s*[-–—]?\s*["“«]', line, maxsplit=1)[0].strip()
    if "non-translation" in head or "non translation" in head:
        return "non-translation"
    for fam, subs in SUBCLASSES.items():
        if fam in head:
            sub = next((s for s in subs if s in head or s.split()[0] in head), None)
            return f"{fam}/{sub}" if sub else fam
    if "other" in head:
        return "other"
    for word, fam in BARE_SUB.items():
        if word in head:
            sub = next((s for s in SUBCLASSES[fam] if s.startswith(word)), word)
            return f"{fam}/{sub}"
    return "unknown"


def is_meaning(cat: str) -> bool:
    return cat.startswith(("accuracy", "terminology")) or cat == "non-translation"


NO_ERROR_REPLIES = ("no-error", "no error", "no errors", "none")


def parse_gemba(text: str) -> dict:
    """(severity, category) errors from a GEMBA-MQM reply. Differences from GEMBA's own parser,
    all for replies outside the few-shot format: severity headers may carry markup or an
    inline error ("Major: accuracy/... - ..."); a reply that is only "no-error" means no
    errors (status no_error_only); after the first severity header, any other line ending in
    ":" ("Explanation:", "Final assessment:") ends the error list, so commentary is not
    counted; and lines whose category cannot be recognised are skipped (counted as ignored)
    instead of being scored as errors."""
    errors: list[tuple[str, str]] = []
    level, seen, orphans, ignored = None, False, 0, 0
    for raw in text.splitlines():
        line = raw.replace("**", "").strip().lower()
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line).strip().strip("*_#` ").strip()
        if not line:
            continue
        m = HEADER_RE.match(line)
        if m:
            level, seen, line = m.group(1), True, m.group(2).strip()
            if not line:
                continue
        elif seen and line.endswith(":"):
            break
        if "no-error" in line or "no error" in line or line in ("none", "n/a", "-"):
            continue
        if level is None:
            orphans += 1
            continue
        cat = canonical(line)
        if cat == "unknown":
            ignored += 1
            continue
        errors.append(("critical" if cat == "non-translation" else level, cat))
    bare = text.replace("*", "").strip().strip(".").lower()
    status = "ok" if seen else ("no_error_only" if bare in NO_ERROR_REPLIES else "no_levels")
    return {"errors": errors, "status": status, "orphans": orphans, "ignored": ignored}


def gemba_penalty(errors: list[tuple[str, str]]) -> int:
    ordered = [e for lev in ("critical", "major", "minor") for e in errors if e[0] == lev]
    return min(25, sum(WEIGHT[lev] for lev, _ in ordered[:5]))


def score_gemba(entry: dict) -> dict:
    p = parse_gemba(entry.get("content") or "")
    errs = p["errors"]
    meaning = [e for e in errs if is_meaning(e[1])]
    sev = collections.Counter(lev for lev, _ in errs)
    return {"risk": gemba_penalty(meaning) / 25, "risk_all": gemba_penalty(errs) / 25,
            "mqm": -gemba_penalty(errs),
            "pred": "Major" if any(lev in ("critical", "major") for lev, _ in meaning) else "OK",
            "n_critical": sev["critical"], "n_major": sev["major"], "n_minor": sev["minor"],
            "meaning_errors": [[lev, cat] for lev, cat in meaning],
            "ignored_lines": p["ignored"],
            "status": p["status"], "orphan_lines": p["orphans"],
            "truncated": entry.get("finish_reason") == "length"}


# --------------------------------------------------------------------------- cache / budget

class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.total_cost = 0.0
        self.total_cost_cached = 0.0
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if line.strip():
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        if i == len(lines) - 1:  # another process is appending right now
                            continue
                        raise
                    self.data[e["key"]] = e  # last write wins
                    self.total_cost += e["cost"]  # every line was a paid call, superseded or not
                    self.total_cost_cached += e.get("cost_cached", e["cost"])
        self.fh = None

    def get(self, key: str, req_hash: str) -> dict | None:
        e = self.data.get(key)
        return e if e and e["req_hash"] == req_hash else None

    def put(self, entry: dict) -> None:
        if self.fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.path, "a", encoding="utf-8")
        self.fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.data[entry["key"]] = entry


class Budget:
    def __init__(self, limit: float):
        self.limit, self.spent, self.reserved = limit, 0.0, 0.0

    def reserve(self, amount: float) -> bool:
        if self.spent + self.reserved + amount > self.limit:
            return False
        self.reserved += amount
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual


class Pacer:
    """Start at most `rpm` requests per minute across all workers; a 429 pauses everyone."""

    def __init__(self, rpm: float):
        self.interval, self.next = 60.0 / rpm, 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        if start > now:
            await asyncio.sleep(start - now)

    def pause(self, seconds: float) -> None:
        self.next = max(self.next, time.monotonic() + seconds)


class Fatal(Exception):
    pass


class RequestFailed(Exception):
    pass


# --------------------------------------------------------------------------- API

async def post(client: httpx.AsyncClient, body: dict, pacer: Pacer, stats: collections.Counter):
    last = ""
    for attempt in range(MAX_ATTEMPTS):
        await pacer.wait()
        t0 = time.perf_counter()
        delay = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
        try:
            r = await client.post("/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = f"{type(e).__name__}"
            stats["retry_transport"] += 1
        else:
            dt = time.perf_counter() - t0
            if r.status_code == 200:
                return r.json(), dt, attempt + 1, r.headers
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (401, 403):
                raise Fatal(f"authentication failed ({r.status_code}): check DO_MODEL_KEY")
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("retry-after")
                try:
                    delay = max(delay, float(ra)) if ra else delay
                except ValueError:
                    pass
                if r.status_code == 429:
                    stats["retry_429"] += 1
                    pacer.pause(delay)
                else:
                    stats["retry_5xx"] += 1
            else:
                raise RequestFailed(last)
        await asyncio.sleep(delay)
    raise RequestFailed(f"{MAX_ATTEMPTS} attempts failed, last: {last}")


def make_entry(method: str, row: dict, rh: str, res: dict, dt: float, attempts: int) -> dict:
    ch = res["choices"][0]
    msg = ch.get("message") or {}
    lps = None
    if (ch.get("logprobs") or {}).get("content"):
        lps = [{"token": p["token"], "logprob": p["logprob"],
                "top": [[t["token"], t["logprob"]] for t in (p.get("top_logprobs") or [])]}
               for p in ch["logprobs"]["content"][:4]]
    u = res.get("usage") or {}
    usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
             "prompt_tokens_details": u.get("prompt_tokens_details"),
             "completion_tokens_details": u.get("completion_tokens_details")}
    cost, cost_cached = usage_cost(usage)
    return {"key": f"{method}|{row['id']}", "method": method, "id": row["id"], "req_hash": rh,
            "model": res.get("model"), "content": msg.get("content"), "finish_reason": ch.get("finish_reason"),
            "has_reasoning": bool(msg.get("reasoning_content") or msg.get("reasoning")),
            "logprobs": lps, "usage": usage, "cost": cost, "cost_cached": cost_cached,
            "latency_s": round(dt, 4), "attempts": attempts, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


async def run_calls(jobs: list[tuple[dict, dict, str]], method: str, cache: Cache, args) -> collections.Counter:
    key = os.environ.get("DO_MODEL_KEY")
    if not key:
        raise SystemExit("DO_MODEL_KEY is not set (source .env first). Use --dry-run to preview.")
    stats: collections.Counter = collections.Counter()
    budget = Budget(args.cap - cache.total_cost)
    pacer = Pacer(args.rpm)
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stop: dict[str, str | None] = {"reason": None}
    consecutive = {"n": 0}
    headers_logged = asyncio.Event()
    t_start = time.perf_counter()
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=httpx.Timeout(90.0, connect=15.0), limits=limits,
                                 headers={"Authorization": f"Bearer {key}"}) as client:

        async def worker():
            while stop["reason"] is None:
                try:
                    row, body, rh = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                wc = worst_cost(body)
                if not budget.reserve(wc):
                    stats["skipped_budget"] += 1 + queue.qsize()
                    stop["reason"] = (f"budget guard: ${cache.total_cost + budget.spent:.4f} spent in this track, "
                                      f"next call could exceed the cap ${args.cap:.2f}")
                    return
                try:
                    res, dt, attempts, hdrs = await post(client, body, pacer, stats)
                except Fatal as e:
                    budget.settle(wc, 0.0)
                    stop["reason"] = str(e)
                    return
                except RequestFailed as e:
                    budget.settle(wc, 0.0)
                    stats["failed"] += 1
                    consecutive["n"] += 1
                    print(f"  request failed for {row['id']}: {e}", flush=True)
                    if consecutive["n"] >= 10:
                        stop["reason"] = "10 failed requests in a row"
                    continue
                consecutive["n"] = 0
                if not headers_logged.is_set():
                    headers_logged.set()
                    rl = {k: v for k, v in hdrs.items() if k.lower().startswith("x-ratelimit")}
                    print(f"  rate-limit headers: {rl or 'none'}", flush=True)
                entry = make_entry(method, row, rh, res, dt, attempts)
                budget.settle(wc, entry["cost"])
                cache.put(entry)
                stats["called"] += 1
                stats["tokens_in"] += entry["usage"]["prompt_tokens"] or 0
                stats["tokens_out"] += entry["usage"]["completion_tokens"] or 0
                stats["tokens_cached"] += (entry["usage"]["prompt_tokens_details"] or {}).get("cached_tokens") or 0
                stats["cost_micro"] += round(entry["cost"] * 1e6)
                if stats["called"] % args.progress == 0:
                    el = time.perf_counter() - t_start
                    print(f"  {stats['called']}/{len(jobs)} calls, ${budget.spent:.4f} this run, "
                          f"{stats['called'] / el * 60:.0f} req/min", flush=True)

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    stats["wall_s"] = round(time.perf_counter() - t_start)
    if stop["reason"]:
        print(f"STOPPED: {stop['reason']}", flush=True)
    return stats


# --------------------------------------------------------------------------- main

def select_rows(args) -> list[dict]:
    rows = M.load_scored(args.split)
    if args.ids_from:
        by_id = {r["id"]: r for r in rows}
        ids = [r["id"] for r in M.load_scored(args.ids_from)]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} ids of {args.ids_from} are not in {args.split}")
        rows = [by_id[i] for i in ids]
    elif args.sample:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    if args.limit:
        rows = rows[: args.limit]
    return rows


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("split", help="split jsonl (data/splits/*.jsonl)")
    ap.add_argument("--method", choices=("direct", "gemba"), required=True)
    ap.add_argument("--out", help="scored jsonl to write")
    ap.add_argument("--out-all", help="gemba only: also write a scored file with risk = risk_all (all errors)")
    ap.add_argument("--ids-from", help="score exactly the ids of this scored/split jsonl, in its order")
    ap.add_argument("--sample", type=int, help="score a seeded random sample of N rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="only the first N selected rows (pilots)")
    ap.add_argument("--cap", type=float, default=1.20, help="USD cap for the whole track, cache spend included")
    ap.add_argument("--trim", action="store_true", help="if the projection exceeds --cap, run the prefix that fits")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=110, help="max request starts per minute (account limit 120)")
    ap.add_argument("--top-logprobs", type=int, default=10)
    ap.add_argument("--direct-max-tokens", type=int, default=3)
    ap.add_argument("--gemba-max-tokens", type=int, default=256)
    ap.add_argument("--system", choices=("native", "merge"), default="native",
                    help="gemba: send the GEMBA system text as a system turn, or prepend it to the first user turn")
    ap.add_argument("--base-url", default=os.environ.get("DO_BASE_URL", DO_BASE_URL))
    ap.add_argument("--progress", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and one prompt; no calls, no key")
    ap.add_argument("--cache-only", action="store_true", help="no calls: score what the cache has")
    args = ap.parse_args()
    if args.concurrency > 8:
        raise SystemExit("--concurrency is limited to 8")

    rows = select_rows(args)
    turns = load_turns(GEMBA_PROMPT) if args.method == "gemba" else None
    cache = Cache(CACHE_PATH)
    jobs, hits = [], 0
    for r in rows:
        body = build_request(args.method, r, args, turns)
        rh = request_hash(body)
        if cache.get(f"{args.method}|{r['id']}", rh):
            hits += 1
        else:
            jobs.append((r, body, rh))
    worst = sum(worst_cost(b) for _, b, _ in jobs)
    print(f"{args.method} on {args.split}: {len(rows)} rows, {hits} cached, {len(jobs)} to call; "
          f"worst case ${worst:.4f}. Track spend so far ${cache.total_cost:.4f} "
          f"(${cache.total_cost_cached:.4f} with the cached-input rate), cap ${args.cap:.2f}", flush=True)
    if cache.total_cost + worst > args.cap and jobs and not (args.dry_run or args.cache_only):
        if not args.trim:
            raise SystemExit(f"projected ${cache.total_cost + worst:.4f} > cap ${args.cap:.2f}: refusing to start "
                             "(lower the row count, raise --cap, or pass --trim)")
        room, kept = args.cap - cache.total_cost, []
        for j in jobs:
            room -= worst_cost(j[1])
            if room < 0:
                break
            kept.append(j)
        print(f"  --trim: keeping {len(kept)} of {len(jobs)} calls", flush=True)
        jobs = kept
    if args.dry_run:
        if jobs:
            print(json.dumps(jobs[0][1], ensure_ascii=False, indent=1))
        return

    stats = collections.Counter()
    if jobs and not args.cache_only:
        stats = asyncio.run(run_calls(jobs, args.method, cache, args))
        print(f"this run: {stats['called']} calls in {stats['wall_s']} s, tokens in {stats['tokens_in']:,} "
              f"(cached {stats['tokens_cached']:,}) out {stats['tokens_out']:,}, ${stats['cost_micro'] / 1e6:.4f}; "
              f"retries 429={stats['retry_429']} 5xx={stats['retry_5xx']} transport={stats['retry_transport']}; "
              f"failed {stats['failed']}, skipped by budget {stats['skipped_budget']}", flush=True)
    print(f"track spend now ${cache.total_cost + stats['cost_micro'] / 1e6:.4f} at list price", flush=True)

    scored, entries = [], []
    for r in rows:
        body = build_request(args.method, r, args, turns)
        e = cache.get(f"{args.method}|{r['id']}", request_hash(body))
        if e is None:
            continue
        entries.append(e)
        sc = score_direct(e) if args.method == "direct" else score_gemba(e)
        scored.append({"id": r["id"], "label": r["label"], "risk": sc.pop("risk"), "doc": r.get("doc"),
                       "lp": r.get("lp"), **sc})
    if not scored:
        print("nothing scored")
        return
    n = len(scored)
    if n < len(rows):
        print(f"WARNING: {len(rows) - n} of {len(rows)} rows have no reply and are left out")
    lat = [e["latency_s"] for e in entries]
    tin = sum(e["usage"]["prompt_tokens"] or 0 for e in entries)
    tout = sum(e["usage"]["completion_tokens"] or 0 for e in entries)
    cost = sum(e["cost"] for e in entries)
    print(f"scored {n}: tokens in {tin:,} out {tout:,} (mean {tin / n:.0f} / {tout / n:.1f} per segment), "
          f"${cost:.4f} = ${cost / n * 1000:.3f} per 1000 segments; latency p50 {statistics.median(lat):.2f} s "
          f"p95 {pct(lat, 0.95):.2f} s; reasoning field in {sum(e['has_reasoning'] for e in entries)}; "
          f"finish_reason {dict(collections.Counter(e['finish_reason'] for e in entries))}")
    st = collections.Counter(s["status"] for s in scored)
    print(f"status: {dict(st)}")
    if args.method == "direct":
        fails = st["text"] + st["unparsed"]
        print(f"labels not in top-{args.top_logprobs} (text fallback): {fails} ({fails / n:.1%}); "
              f"unparsed {st['unparsed']}; one label only {st['major_only'] + st['ok_only']}; "
              f"first tokens {dict(collections.Counter(s['top'] for s in scored).most_common(6))}")
    else:
        print(f"parse failures (no severity headers): {st['no_levels']} ({st['no_levels'] / n:.1%}); "
              f"truncated {sum(s['truncated'] for s in scored)}; lines skipped (unrecognised category) "
              f"{sum(s['ignored_lines'] for s in scored)}; orphan lines {sum(s['orphan_lines'] for s in scored)}; "
              f"pred Major {sum(s['pred'] == 'Major' for s in scored)}")
    y, s = M.to_arrays(scored)
    print(f"Major rate {y.mean():.3f}; AUROC {M.auroc(y, s):.3f} PR-AUC {M.pr_auc(y, s):.3f} "
          f"recall@10% {M.recall_at(y, s):.3f}")
    if args.method == "gemba":
        pred = np.asarray([x["pred"] == "Major" for x in scored])
        tp = int((pred & (y == 1)).sum())
        print(f"pred=Major rule: precision {tp / max(1, pred.sum()):.3f} recall {tp / max(1, y.sum()):.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for x in scored:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")
    if args.out_all and args.method == "gemba":
        with open(args.out_all, "w", encoding="utf-8") as f:
            for x in scored:
                f.write(json.dumps({**x, "risk": x["risk_all"], "risk_meaning": x["risk"]}, ensure_ascii=False) + "\n")
        print(f"wrote {args.out_all}")


if __name__ == "__main__":
    main()
