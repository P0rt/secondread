"""Sentry setup for SecondRead: opt-in, traces only, and no segment text ever leaves.

Sentry is off unless ``SENTRY_DSN`` is set; with no DSN the SDK has no transport
and nothing is sent. When it is on, three layers keep translation text out:

1. We never put text into telemetry. Spans carry fixed names, numbers and
   machine codes only (see ``secondread.agent``). Segment ids are not sent
   either: CAT tools sometimes use the source string as the id.
2. SDK options: ``send_default_pii=False``, ``include_local_variables=False``
   (no frame locals in stack traces), ``max_request_body_size="never"``
   (no request bodies), and no OpenAI/httpx auto-instrumentation, so model
   calls are traced only by our manual ``gen_ai.chat`` spans (no duplicates,
   private prep notes, 02-corrections.md item 22).
3. ``before_send`` / ``before_send_transaction`` / ``before_breadcrumb``
   scrub what integrations add on their own: exception messages, log
   messages and params, query strings, request headers and unknown span data
   (item 23). ``scripts/canary_check.py`` proves it end to end.
"""
from __future__ import annotations

import os
import re
from typing import Any

import sentry_sdk

__all__ = ["SafeError", "init_sentry", "sentry_enabled", "scrub_event", "scrub_breadcrumb"]

REDACTED = "[redacted]"

_initialized = False
_enabled = False


class SafeError(Exception):
    """An exception whose message is text-free by construction (codes, numbers,
    local URLs). Only these keep their message in Sentry; every other exception
    is reported with its type and stack trace but a redacted message."""


def init_sentry(dsn: str | None = None, *, transport: Any = None, force: bool = False, **overrides: Any) -> bool:
    """Initialise Sentry once per process. Returns True when events can be sent.

    ``dsn`` defaults to ``$SENTRY_DSN``. ``transport`` lets tests capture
    envelopes in memory instead of sending them. Repeated calls are no-ops
    unless ``force=True``, so a test can initialise first and the server
    module's own call will not replace that setup.
    """
    global _initialized, _enabled
    if _initialized and not force:
        return _enabled
    from sentry_sdk.integrations.httpx import HttpxIntegration
    from sentry_sdk.integrations.openai import OpenAIIntegration

    dsn = dsn if dsn is not None else (os.getenv("SENTRY_DSN") or None)
    options: dict[str, Any] = dict(
        dsn=dsn,
        traces_sample_rate=1.0,
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        # Manual gen_ai spans only: the OpenAI integration would add its own
        # gen_ai.chat spans; httpx spans would duplicate every model call.
        disabled_integrations=[OpenAIIntegration(), HttpxIntegration()],
        before_send=scrub_event,
        before_send_transaction=scrub_event,
        before_breadcrumb=scrub_breadcrumb,
        # The default is the hostname, which often contains a person's name.
        server_name="secondread-local",
        environment=os.getenv("SENTRY_ENVIRONMENT", "local"),
        release=os.getenv("SENTRY_RELEASE", "secondread@0.1.0"),
    )
    if transport is not None:
        options["transport"] = transport
    options.update(overrides)
    sentry_sdk.init(**options)
    _initialized, _enabled = True, bool(dsn)
    return _enabled


def sentry_enabled() -> bool:
    return _enabled


# ------------------------------------------------------------- scrubbing ----

# Span/trace data keys we keep. Everything else an integration might attach is dropped.
_SAFE_DATA_PREFIXES = ("gen_ai.", "secondread.", "http.", "thread.", "sentry.", "server.", "network.",
                       "url", "code.")
# gen_ai.* keys that carry prompts, outputs or tool payloads (never set by us,
# blocked in case an integration does). Usage and cost keys stay.
_GEN_AI_CONTENT = re.compile(r"messages|prompt|completion|instructions|arguments|result|text|input|output"
                             r"|tool\.call|tool_calls|available_tools")
_GEN_AI_NUMERIC_OK = ("gen_ai.usage.", "gen_ai.cost.")
_HTTP_UNSAFE = {"http.query", "http.fragment", "http.request.body", "http.response.body"}
# Kept values must look like codes, names or numbers, never prose.
_CODE_LIKE = re.compile(r"[A-Za-z0-9_.:+\-/]{0,80}")
_SAFE_HEADERS = {"content-type", "content-length", "accept", "user-agent"}
_QUERY = re.compile(r"\?[^\s\"']*")


def _safe_value(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, float)):
        return True
    if isinstance(v, str):
        return _CODE_LIKE.fullmatch(v) is not None
    if isinstance(v, (list, tuple)):   # short lists of codes; long ones could be token ids
        return len(v) <= 16 and all(_safe_value(x) for x in v)
    return False


def _scrub_data(data: dict | None) -> dict:
    out: dict = {}
    for k, v in (data or {}).items():
        if not isinstance(k, str) or not k.startswith(_SAFE_DATA_PREFIXES) or k in _HTTP_UNSAFE:
            continue
        if k.startswith("gen_ai.") and not k.startswith(_GEN_AI_NUMERIC_OK) and _GEN_AI_CONTENT.search(k):
            continue
        if k.startswith(("url", "http.url")) and isinstance(v, str):
            v = _QUERY.sub("", v)
        if _safe_value(v):   # numbers, booleans and code-like strings only
            out[k] = v
    return out


def _scrub_exception_values(event: dict) -> None:
    safe_types = {c.__name__ for c in _all_subclasses(SafeError)}
    for exc in (event.get("exception") or {}).get("values") or []:
        if not (str(exc.get("module", "")).startswith("secondread") and exc.get("type") in safe_types):
            exc["value"] = REDACTED
        for frame in (exc.get("stacktrace") or {}).get("frames") or []:
            frame.pop("vars", None)
    for thread in (event.get("threads") or {}).get("values") or []:
        for frame in (thread.get("stacktrace") or {}).get("frames") or []:
            frame.pop("vars", None)


def _all_subclasses(cls: type) -> set[type]:
    out = {cls}
    for sub in cls.__subclasses__():
        out |= _all_subclasses(sub)
    return out


def _scrub_request(event: dict) -> None:
    req = event.get("request")
    if not isinstance(req, dict):
        return
    for key in ("data", "query_string", "cookies", "env"):
        req.pop(key, None)
    if isinstance(req.get("url"), str):
        req["url"] = _QUERY.sub("", req["url"])
    if isinstance(req.get("headers"), dict):
        req["headers"] = {k: v for k, v in req["headers"].items() if k.lower() in _SAFE_HEADERS}


def scrub_event(event: dict, hint: dict | None = None) -> dict:
    """``before_send`` and ``before_send_transaction``: keep structure, drop content."""
    _scrub_exception_values(event)
    _scrub_request(event)
    for key in ("extra", "user"):
        event.pop(key, None)
    if "logentry" in event:   # error events created from log records
        event["logentry"] = {"message": REDACTED, "params": []}
    if isinstance(event.get("message"), str):
        event["message"] = REDACTED
    tags = event.get("tags")
    if isinstance(tags, dict):
        event["tags"] = {k: v for k, v in tags.items() if _safe_value(v)}

    # A raw URL as the transaction name (unmatched route) could carry anything.
    if (event.get("transaction_info") or {}).get("source") == "url":
        event["transaction"] = "<unmatched route>"
        if isinstance(event.get("request"), dict):
            event["request"]["url"] = "<unmatched route>"

    trace = (event.get("contexts") or {}).get("trace")
    if isinstance(trace, dict) and "data" in trace:
        trace["data"] = _scrub_data(trace["data"])
    for span in event.get("spans") or []:
        span["data"] = _scrub_data(span.get("data"))
        if isinstance(span.get("tags"), dict):
            span["tags"] = {k: v for k, v in span["tags"].items() if _safe_value(v)}
        if isinstance(span.get("description"), str):
            span["description"] = _QUERY.sub("", span["description"])[:200]

    crumbs = event.get("breadcrumbs")
    values = crumbs.get("values") if isinstance(crumbs, dict) else crumbs
    if isinstance(values, list):
        kept = [c for c in (scrub_breadcrumb(dict(c), {}) for c in values) if c is not None]
        if isinstance(crumbs, dict):
            crumbs["values"] = kept
        else:
            event["breadcrumbs"] = kept
    return event


def scrub_breadcrumb(crumb: dict, hint: dict | None = None) -> dict | None:
    """``before_breadcrumb``: keep when/where/what-kind, drop what was said.

    Log breadcrumbs keep logger name and level but lose the message (log lines
    may interpolate segment text). HTTP breadcrumbs keep method, status and the
    URL without query string. Everything else keeps only its category.
    """
    out = {k: crumb[k] for k in ("type", "category", "level", "timestamp") if k in crumb}
    data = crumb.get("data") or {}
    if crumb.get("type") == "http" or crumb.get("category") in ("httplib", "http"):
        out["data"] = {k: data[k] for k in ("method", "status_code", "reason") if k in data}
        if isinstance(data.get("url"), str):
            out["data"]["url"] = _QUERY.sub("", data["url"])
    elif crumb.get("message"):
        out["message"] = REDACTED
    return out
