"""SecondRead review agent: a fixed tool pipeline over MT segments, traced as a Sentry AI agent.

    parse_xliff -> number_check -> negation_check -> lora_score -> aggregate

Trace shape (private prep notes, 02-corrections.md items 22-24):

    gen_ai.invoke_agent   "invoke_agent SecondRead"        one per review
      gen_ai.execute_tool "execute_tool parse_xliff"       one per tool run, over all segments
      gen_ai.execute_tool "execute_tool number_check"
      gen_ai.execute_tool "execute_tool negation_check"
      gen_ai.execute_tool "execute_tool lora_score"
        gen_ai.chat       "chat <model>"                   one per llama-server call
      secondread.aggregate

Span data holds fixed names, counts, scores, token counts and issue codes only. Segment
text, segment ids, reason strings and file names never go into spans or log messages;
secondread.telemetry scrubs whatever integrations add on their own.

Usage:
    agent = ReviewAgent()
    result = await agent.review([{"src": ..., "mt": ..., "lp": "en-ru"}])
    result = await agent.review(xliff_bytes, filename="job.sdlxliff")
    review_sync(...)  # the same, for scripts
"""
from __future__ import annotations

import os

# The product is offline: never let the tokenizer reach the Hugging Face Hub.
# Must be set before huggingface_hub is imported (the renderer loads it lazily).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import logging
import math
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import PurePath
from typing import Any

import httpx
import sentry_sdk

from secondread import checks
from secondread.prompt import LABEL_IDS, LANG, generation_prompt_ids
from secondread.telemetry import SafeError

log = logging.getLogger("secondread.agent")

AGENT_NAME = "SecondRead"
DEFAULT_LLAMA_URL = os.getenv("SECONDREAD_LLAMA_URL", "http://127.0.0.1:9931")
DEFAULT_THRESHOLD = float(os.getenv("SECONDREAD_THRESHOLD", "0.5"))
N_PROBS = 200          # the base model can rank a label below the top 20 (02-corrections item 15)
CONCURRENCY = 4        # matches llama-server -np 4 (parallel slots)
LOCAL_COST_USD = 0.0   # local inference is free; set explicitly or Sentry prices by model name
MAX_SEGMENTS = 5000


# ------------------------------------------------------------------ errors ----

class ReviewError(SafeError):
    """Bad input. ``code`` and the message are text-free; ``status`` is the HTTP status."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.status = code, message, status


class LlamaServerError(SafeError):
    """llama-server failed; ``code`` is e.g. "llama_unreachable" or "llama_http_500"."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ------------------------------------------------------------------- types ----

@dataclass
class Segment:
    index: int                # 0-based position in the request
    id: str                   # caller's id, or the XLIFF segment key (for write-back)
    src: str
    mt: str
    lp: str
    meta: dict[str, Any] = field(default_factory=dict)  # XLIFF origin/state/match, if any


@dataclass
class ModelScore:
    risk: float | None = None       # p(Major) / (p(Major) + p(OK))
    p_major: float | None = None
    p_ok: float | None = None
    labels_in_top: bool = False     # both labels were in the top-N; else a bound was used
    prompt_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class SegmentResult:
    """One reviewed segment; the wire format the UI reads (ui/API.md)."""
    index: int
    id: str
    lp: str
    src: str
    mt: str
    risk: float
    flagged: bool
    # [{code, issue, message, confidence, spans: [{side, start, end}]}], strongest first.
    # Messages quote the segment: for the translator only, never logged or traced.
    reasons: list[dict[str, Any]]
    issues: list[str]               # machine codes, e.g. ["number_changed", "model_high"]
    # {"number_check": {...}, "negation_check": {...}, "lora_score": float | None}
    signals: dict[str, Any]
    model: dict[str, Any] | None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewResult:
    segments: list[SegmentResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        lps = {s.lp for s in self.segments}
        return {"lp": lps.pop() if len(lps) == 1 else None, **asdict(self),
                "timing_ms": self.summary["timing_ms"]}


# --------------------------------------------------------------- llama.cpp ----

class LlamaClient:
    """Minimal async client for llama-server's /completion with token-id prompts."""

    def __init__(self, base_url: str = DEFAULT_LLAMA_URL, *, n_probs: int = N_PROBS, timeout: float = 120.0):
        self.base_url = base_url
        self.n_probs = n_probs
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._info: dict[str, Any] | None = None

    async def aclose(self) -> None:
        await self._http.aclose()

    async def info(self, refresh: bool = False) -> dict[str, Any]:
        """Server status, model name, slot context size and loaded LoRA adapters (cached)."""
        if self._info is not None and not refresh:
            return self._info
        try:
            props = (await self._http.get("/props", timeout=5)).json()
            adapters = (await self._http.get("/lora-adapters", timeout=5)).json()
        except (httpx.HTTPError, ValueError):
            self._info = None
            return {"ok": False, "url": self.base_url}
        base = PurePath(props.get("model_path") or "model").stem
        loras = [{"id": a.get("id", i), "name": PurePath(a.get("path", "")).stem, "scale": a.get("scale")}
                 for i, a in enumerate(adapters if isinstance(adapters, list) else [])]
        self._info = {
            "ok": True, "url": self.base_url, "base_model": base,
            "model": "+".join([base, *(a["name"] for a in loras)]),
            "n_ctx_slot": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "slots": props.get("total_slots"), "lora_adapters": loras,
        }
        return self._info

    async def score(self, prompt_ids: list[int], lora_scale: float | None, lora_ids: list[int]) -> ModelScore:
        body: dict[str, Any] = {"prompt": prompt_ids, "n_predict": 1, "n_probs": self.n_probs,
                                "temperature": 0, "cache_prompt": False}
        if lora_scale is not None and lora_ids:
            body["lora"] = [{"id": i, "scale": lora_scale} for i in lora_ids]
        t0 = time.perf_counter()
        try:
            r = await self._http.post("/completion", json=body)
        except httpx.TimeoutException as e:
            raise LlamaServerError("llama_timeout") from e
        except httpx.HTTPError as e:
            raise LlamaServerError("llama_unreachable") from e
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise LlamaServerError(f"llama_http_{r.status_code}")
        # The response echoes the prompt text and the generated token: read numbers only.
        res = r.json()
        score = ModelScore(prompt_tokens=int(res.get("tokens_evaluated") or len(prompt_ids)), latency_ms=latency_ms)
        probs = _label_probs(res)
        if probs is None:
            score.error = "no_label_probs"
        else:
            score.p_major, score.p_ok, score.labels_in_top = probs
            score.risk = score.p_major / (score.p_major + score.p_ok)
        return score


def _label_probs(res: Mapping[str, Any]) -> tuple[float, float, bool] | None:
    """p(Major), p(OK) for the first generated token, from the top-N list.

    A label missing from the top-N has at most the smallest listed logprob, so
    that bound stands in for it (the probe used -30, which overstates certainty).
    """
    try:
        first = res["completion_probabilities"][0]
    except (KeyError, IndexError, TypeError):
        return None
    cand = first.get("top_logprobs") or first.get("probs") or []
    logp = {c.get("id"): c["logprob"] if "logprob" in c else math.log(max(c.get("prob", 0.0), 1e-30))
            for c in cand}
    if not logp:
        return None
    found = {name: logp[tid] for name, tid in LABEL_IDS.items() if tid in logp}
    if not found:
        return None
    floor = min(logp.values())
    return math.exp(found.get("Major", floor)), math.exp(found.get("OK", floor)), len(found) == 2


# ------------------------------------------------------------ input parsing ----

def _norm_lp(lp: str | None) -> str | None:
    """"en-RU" / "en_ru" -> "en-ru"; anything else is returned lower-cased for _check_lp to reject."""
    if not lp:
        return None
    parts = lp.strip().replace("_", "-").lower().split("-")
    return "-".join(parts) if len(parts) != 2 else f"{parts[0]}-{parts[1]}"


def _check_lp(lp: str | None) -> str:
    if lp is None:
        raise ReviewError("lp_required", "language pair is required (e.g. 'en-ru')")
    s, _, t = lp.partition("-")
    if s not in LANG or t not in LANG or s == t:
        raise ReviewError("lp_unsupported", f"unsupported language pair; use two of {sorted(LANG)}")
    return lp


def parse_segments(items: Iterable[Mapping[str, Any]], default_lp: str | None) -> list[Segment]:
    """Raw segments: dicts with "src", "mt", optional "id" and "lp"."""
    out = []
    for i, it in enumerate(items):
        src, mt, lp = it.get("src"), it.get("mt"), it.get("lp")
        if not isinstance(src, str) or not isinstance(mt, str) or not isinstance(lp, (str, type(None))):
            raise ReviewError("bad_segment", f"segment {i}: 'src', 'mt' (and 'lp') must be strings")
        lp = _check_lp(_norm_lp(lp) or default_lp)
        out.append(Segment(i, str(it.get("id") if it.get("id") is not None else i), src, mt, lp))
    return out


def parse_xliff_bytes(data: bytes, default_lp: str | None, filename: str | None = None,
                      include_locked: bool = False) -> tuple[list[Segment], dict[str, Any]]:
    """XLIFF 1.2 / 2.x / SDLXLIFF via secondread.xliff. Untranslated segments, and locked
    ones unless ``include_locked``, are skipped (they are counted in the returned info)."""
    try:
        from secondread import xliff
    except ModuleNotFoundError as e:
        if e.name != "secondread.xliff":
            raise
        raise ReviewError("xliff_unavailable", "XLIFF parsing is not installed; send JSON segments", 501) from e
    try:
        doc = xliff.load(data, filename)
    except xliff.XliffError as e:
        raise ReviewError("xliff_invalid", "not a readable XLIFF 1.2 / 2.x / SDLXLIFF file") from e
    lp = _check_lp(default_lp or _norm_lp(doc.lp))
    segs, skipped = [], Counter()
    for s in doc.segments:
        if not s.tgt_plain:
            skipped["no_target"] += 1
            continue
        if s.locked and not include_locked:
            skipped["locked"] += 1
            continue
        meta = {"key": s.key, "origin": s.origin, "state": s.state, "match_percent": s.match_percent}
        segs.append(Segment(len(segs), s.key, s.src_plain, s.tgt_plain, lp, meta))
    return segs, {"format": doc.format, "skipped": dict(skipped), "total": len(doc.segments)}


# -------------------------------------------------------------- aggregation ----

# Reason families the UI knows (ui/API.md); the specific issue code rides along as "issue".
_FAMILY = {"number_check": "number_mismatch", "negation_check": "negation_mismatch"}


def _reason(code: str, issue: str, message: str, confidence: float | None,
            spans: Iterable[tuple[str, int, int]] = ()) -> dict[str, Any]:
    return {"code": code, "issue": issue, "message": message, "confidence": confidence,
            "spans": [{"side": side, "start": a, "end": b} for side, a, b in spans]}


def aggregate(model: ModelScore | None, results: list[checks.CheckResult], threshold: float,
              model_label: str) -> tuple[float, bool, list[dict[str, Any]], list[str]]:
    """Merge model and rule signals into (risk, flagged, reasons, issue codes).

    Noisy-OR: each signal is an independent chance that the segment has a major
    error, risk = 1 - prod(1 - p). One strong signal flags on its own; two
    medium ones together can. Each rule contributes only its strongest finding,
    so three weak "extra number" hits do not add up to a flag.
    """
    probs = [c.risk for c in results if c.issues]
    reasons = [_reason(_FAMILY.get(c.name, c.name), i.code, i.reason, i.confidence, i.spans)
               for c in results for i in c.issues]
    codes = [code for c in results for code in c.codes]
    if model is not None and model.risk is not None:
        probs.append(model.risk)
        if model.risk >= threshold:
            codes.append("model_high")
        reasons.append(_reason("model_risk", "model_high" if model.risk >= threshold else "model_score",
                               f"Model ({model_label}): P(Major) {model.risk:.0%}", model.risk))
    elif model is not None:
        codes.append("model_unavailable")
        reasons.append(_reason("model_unavailable", model.error or "error",
                               f"Model unavailable ({model.error}): rule checks only", None))
    reasons.sort(key=lambda r: -(r["confidence"] or 0.0))
    risk = 1.0 - math.prod(1.0 - p for p in probs)
    return risk, risk >= threshold, reasons, codes


def _signals(results: list[checks.CheckResult], model: ModelScore | None) -> dict[str, Any]:
    out: dict[str, Any] = {r.name: {"flagged": bool(r.issues), "risk": r.risk, "issues": r.codes,
                                    "skipped": r.skipped, **r.stats} for r in results}
    out["lora_score"] = model.risk if model is not None else None
    return out


# -------------------------------------------------------------------- agent ----

def _set(span: Any, data: Mapping[str, Any]) -> None:
    # span.set_data, not start_span(attributes=...): the latter raises TypeError in 2.72.0.
    for k, v in data.items():
        span.set_data(k, v)


@contextmanager
def _tool(name: str, n_segments: int):
    with sentry_sdk.start_span(op="gen_ai.execute_tool", name=f"execute_tool {name}") as span:
        _set(span, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name,
                    "gen_ai.tool.type": "function", "gen_ai.agent.name": AGENT_NAME,
                    "secondread.segments": n_segments})
        yield span


@contextmanager
def _root_transaction():
    """FastAPI opens a transaction per request; scripts get their own."""
    if sentry_sdk.get_current_span() is not None:
        yield
    else:
        with sentry_sdk.start_transaction(op="secondread.review", name="secondread review"):
            yield


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


class ReviewAgent:
    def __init__(self, llama_url: str = DEFAULT_LLAMA_URL, *, n_probs: int = N_PROBS,
                 concurrency: int = CONCURRENCY, timeout: float = 120.0):
        self.llama = LlamaClient(llama_url, n_probs=n_probs, timeout=timeout)
        self._slots = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self.llama.aclose()

    @staticmethod
    def warm_up() -> None:
        """Load the tokenizer/renderer (several seconds the first time)."""
        generation_prompt_ids("warm up", "warm up", "en-ru")

    async def health(self) -> dict[str, Any]:
        return await self.llama.info(refresh=True)

    async def review(self, payload: bytes | Iterable[Mapping[str, Any]], *, lp: str | None = None,
                     lora_scale: float | None = None, threshold: float | None = None,
                     filename: str | None = None, include_locked: bool = False) -> ReviewResult:
        """Review raw segments or an XLIFF file (bytes). Raises ReviewError on bad input."""
        threshold = DEFAULT_THRESHOLD if threshold is None else threshold
        if not 0.0 <= threshold <= 1.0:
            raise ReviewError("bad_threshold", "threshold must be in [0, 1]")
        default_lp = _norm_lp(lp)
        t_start = time.perf_counter()
        with _root_transaction(), sentry_sdk.start_span(op="gen_ai.invoke_agent",
                                                        name=f"invoke_agent {AGENT_NAME}") as agent_span:
            _set(agent_span, {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": AGENT_NAME,
                              "gen_ai.provider.name": "llama.cpp", "secondread.threshold": threshold})

            # 1. parse_xliff: XLIFF bytes or raw segments -> Segment list
            with _tool("parse_xliff", 0) as span:
                kind = "xliff" if isinstance(payload, (bytes, bytearray)) else "segments"
                span.set_data("secondread.input_kind", kind)
                if kind == "xliff":
                    segs, xinfo = parse_xliff_bytes(bytes(payload), default_lp, filename, include_locked)
                    span.set_data("secondread.xliff_format", xinfo["format"])
                    for reason, k in xinfo["skipped"].items():
                        span.set_data(f"secondread.skipped.{reason}", k)
                else:
                    segs, xinfo = parse_segments(payload, default_lp), None
                if len(segs) > MAX_SEGMENTS:
                    raise ReviewError("too_many_segments", f"at most {MAX_SEGMENTS} segments per review", 413)
                span.set_data("secondread.segments", len(segs))
            agent_span.set_data("secondread.segments", len(segs))
            t_parsed = time.perf_counter()

            # 2-3. rule checks (pure regex, microseconds per segment)
            rule_results: list[list[checks.CheckResult]] = [[] for _ in segs]
            for name, fn in (("number_check", checks.number_check), ("negation_check", checks.negation_check)):
                with _tool(name, len(segs)) as span:
                    counts: Counter[str] = Counter()
                    for seg, bucket in zip(segs, rule_results, strict=True):
                        res = fn(seg.src, seg.mt, seg.lp)
                        bucket.append(res)
                        counts.update(res.codes)
                        if res.skipped:
                            counts[f"skipped_{res.skipped}"] += 1
                    span.set_data("secondread.segments_with_issues",
                                  sum(1 for b in rule_results if b[-1].issues))
                    for code, k in counts.items():
                        span.set_data(f"secondread.issue.{code}", k)
            t_checked = time.perf_counter()

            # 4. lora_score: one llama-server call per segment, up to CONCURRENCY at a time
            info = await self.llama.info(refresh=True)
            model_name = info.get("model", "unknown") if info["ok"] else "unavailable"
            lora_ids = [a["id"] for a in info.get("lora_adapters", [])] if info["ok"] else []
            lora_active = bool(lora_ids) and lora_scale != 0
            model_label = "LoRA" if lora_active else "base model, no LoRA"
            agent_span.set_data("gen_ai.request.model", model_name)
            with _tool("lora_score", len(segs)) as tool_span:
                _set(tool_span, {"gen_ai.request.model": model_name, "secondread.lora_loaded": bool(lora_ids),
                                 "secondread.lora_scale": lora_scale})
                if info["ok"]:
                    n_ctx = info.get("n_ctx_slot") or 4096
                    scores = await asyncio.gather(*(self._score(s, tool_span, model_name, n_ctx, lora_scale, lora_ids)
                                                    for s in segs))
                else:
                    scores = [ModelScore(error="llama_unreachable") for _ in segs]
                errors = Counter(s.error for s in scores if s.error)
                for code, k in errors.items():
                    tool_span.set_data(f"secondread.error.{code}", k)
                if errors:
                    tool_span.set_status("internal_error")
                    # One error event per review, not per segment; its message is a code.
                    sentry_sdk.capture_exception(LlamaServerError(errors.most_common(1)[0][0]))
                    log.warning("lora_score: %d of %d model calls failed (%s)", sum(errors.values()),
                                len(segs), ",".join(sorted(errors)))
            t_scored = time.perf_counter()

            # 5. aggregate
            with sentry_sdk.start_span(op="secondread.aggregate", name="aggregate signals") as span:
                out = []
                for seg, rules, score in zip(segs, rule_results, scores, strict=True):
                    risk, flagged, reasons, codes = aggregate(score, rules, threshold, model_label)
                    out.append(SegmentResult(
                        index=seg.index, id=seg.id, lp=seg.lp, src=seg.src, mt=seg.mt,
                        risk=round(risk, 4), flagged=flagged, reasons=reasons, issues=codes,
                        signals=_signals(rules, score), model=asdict(score), meta=seg.meta))
                n_flagged = sum(r.flagged for r in out)
                span.set_data("secondread.flagged", n_flagged)
            t_end = time.perf_counter()

            ok_scores = [s for s in scores if s.error is None]
            lat = [s.latency_ms for s in ok_scores]
            in_tokens = sum(s.prompt_tokens for s in ok_scores)
            total_ms = (t_end - t_start) * 1000
            summary = {
                "segments": len(out), "flagged": n_flagged, "threshold": threshold, "input": kind,
                "xliff": xinfo, "model": model_name, "model_available": bool(ok_scores),
                "lora_loaded": bool(lora_ids), "lora_scale": lora_scale, "lora_active": lora_active,
                "model_errors": dict(errors),
                "tokens": {"input": in_tokens, "output": len(ok_scores)}, "cost_usd": LOCAL_COST_USD,
                "timing_ms": {
                    "total": round(total_ms, 1),
                    "per_segment": round(total_ms / len(out), 1) if out else None,
                    "parse": round((t_parsed - t_start) * 1000, 1),
                    "checks": round((t_checked - t_parsed) * 1000, 1),
                    "model": round((t_scored - t_checked) * 1000, 1),
                    "model_call_p50": round(statistics.median(lat), 1) if lat else None,
                    "model_call_p95": round(_pct(lat, 0.95), 1) if lat else None,
                },
            }
            _set(agent_span, {
                "gen_ai.usage.input_tokens": in_tokens, "gen_ai.usage.output_tokens": len(ok_scores),
                "gen_ai.usage.total_tokens": in_tokens + len(ok_scores),
                "gen_ai.cost.total_tokens": LOCAL_COST_USD,
                "secondread.flagged": n_flagged, "secondread.input_kind": kind,
                "secondread.lora_active": lora_active, "secondread.model_errors": sum(errors.values()),
                "secondread.ms_per_segment": summary["timing_ms"]["per_segment"],
            })
            log.info("review: %d segments, %d flagged, %.0f ms", len(out), n_flagged, total_ms)
            return ReviewResult(out, summary)

    async def _score(self, seg: Segment, parent: Any, model_name: str, n_ctx: int,
                     lora_scale: float | None, lora_ids: list[int]) -> ModelScore:
        ids = generation_prompt_ids(seg.src, seg.mt, seg.lp)  # fast after warm_up; not thread-safe
        if len(ids) >= n_ctx:
            return ModelScore(prompt_tokens=len(ids), error="too_long")
        async with self._slots:
            # parent.start_child, not start_span(): concurrent tasks share one Sentry scope,
            # and entering spans as context managers there would mis-parent siblings.
            span = parent.start_child(op="gen_ai.chat", name=f"chat {model_name}")
            _set(span, {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "llama.cpp",
                        "gen_ai.system": "llama.cpp", "gen_ai.agent.name": AGENT_NAME,
                        "gen_ai.request.model": model_name, "gen_ai.response.model": model_name,
                        "gen_ai.request.max_tokens": 1, "gen_ai.request.temperature": 0.0,
                        "secondread.segment_index": seg.index, "secondread.lora_scale": lora_scale})
            try:
                score = await self.llama.score(ids, lora_scale, lora_ids)
            except LlamaServerError as e:
                span.set_status("internal_error")
                span.set_data("secondread.error", e.code)
                score = ModelScore(prompt_tokens=len(ids), error=e.code)
            else:
                _set(span, {"gen_ai.usage.input_tokens": score.prompt_tokens, "gen_ai.usage.output_tokens": 1,
                            "gen_ai.usage.total_tokens": score.prompt_tokens + 1,
                            "gen_ai.cost.input_tokens": LOCAL_COST_USD,
                            "gen_ai.cost.output_tokens": LOCAL_COST_USD,
                            "gen_ai.cost.total_tokens": LOCAL_COST_USD,
                            "secondread.p_major": score.p_major, "secondread.risk": score.risk,
                            "secondread.labels_in_top": score.labels_in_top,
                            "secondread.latency_ms": round(score.latency_ms, 1)})
                if score.error:
                    span.set_data("secondread.error", score.error)
            finally:
                span.finish()
            return score


def review_sync(payload: bytes | Iterable[Mapping[str, Any]], **kwargs: Any) -> ReviewResult:
    """Blocking helper for scripts: one agent, one review."""
    async def run() -> ReviewResult:
        agent = ReviewAgent()
        try:
            return await agent.review(payload, **kwargs)
        finally:
            await agent.aclose()
    return asyncio.run(run())
