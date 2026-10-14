"""G8: prove that no segment text reaches Sentry. Offline: envelopes are captured in memory.

Initialises Sentry with a fake DSN and an in-memory transport, runs real reviews through the
FastAPI app in-process (llama-server must be up on 127.0.0.1:9931), then searches every
captured envelope for the canary "SR-CANARY-<uuid>" and for the segment texts.

Leak paths exercised (docs/prep/02-corrections.md item 23):
  json      review of JSON segments: request transaction, invoke_agent/execute_tool/chat spans
  xliff     review of an XLIFF body; canary also in unit ids, file name and query string
  invalid   a 422 request whose body contains the canary
  crash     an unhandled exception whose message quotes the segment (-> 500 + error event)
  logging   logger.warning (breadcrumb) and logger.error with an f-string (error event),
            capture_message, capture_exception, add_breadcrumb with data
  offline   llama-server unreachable: the agent reports an error event, script-side transaction

It also checks the capture is not vacuous (those spans and events really were captured), that
no socket was opened to anything but loopback, and, as a control, that the same search does
find the canary once the scrubbers are switched off.

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/canary_check.py
Exit code 0 = pass.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import sys
import uuid
from collections import Counter
from unittest import mock

# ---- 1. no network: only loopback sockets may be opened ------------------------------------
BLOCKED: list[str] = []
_real_connect, _real_getaddrinfo = socket.socket.connect, socket.getaddrinfo


def _is_loopback(host: object) -> bool:
    if host in ("localhost", "testserver"):
        return True
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return False


def _guarded_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):
        BLOCKED.append(f"connect {address[0]}")
        raise OSError("canary_check: outbound network blocked")
    return _real_connect(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if host is not None and not _is_loopback(host if isinstance(host, str) else host.decode()):
        BLOCKED.append(f"dns {host}")
        raise socket.gaierror("canary_check: DNS blocked")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.getaddrinfo = _guarded_getaddrinfo

# ---- 2. Sentry with an in-memory transport, before the app is imported ---------------------
import sentry_sdk  # noqa: E402
from sentry_sdk.transport import Transport  # noqa: E402

from secondread.telemetry import init_sentry  # noqa: E402

FAKE_DSN = "https://0123456789abcdef0123456789abcdef@o0.ingest.invalid/0"


class CaptureTransport(Transport):
    """Keeps envelopes in memory; never touches the network."""

    def __init__(self, options=None):
        super().__init__(options)
        self.envelopes = []

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)

    def flush(self, timeout, callback=None):
        return None

    def kill(self):
        return None


capture = CaptureTransport()
init_sentry(dsn=FAKE_DSN, transport=capture)

from fastapi.testclient import TestClient  # noqa: E402

from secondread import checks  # noqa: E402
from secondread.agent import ReviewAgent  # noqa: E402
from secondread.server import app  # noqa: E402

CANARY = f"SR-CANARY-{uuid.uuid4()}"
UUID_PART = CANARY.removeprefix("SR-CANARY-")


def tagged(text: str) -> str:
    # Canary at both ends, so a truncated prefix or suffix still carries it.
    return f"{CANARY} {text} {CANARY}"


SEGMENTS = [
    {"id": CANARY, "src": tagged("Do not open the valve before 3 p.m."), "mt": tagged("Откройте клапан до 15:00.")},
    {"id": f"{CANARY}-2", "src": tagged("Wait 15 minutes for the pump to cool down."),
     "mt": tagged("Подождите 50 минут, пока насос остынет.")},
    {"id": f"{CANARY}-3", "src": tagged("Tap the Settings icon at the bottom of the screen."),
     "mt": tagged("Нажмите значок «Настройки» внизу экрана.")},
]
XLIFF = f"""<?xml version="1.0" encoding="UTF-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
  <file original="{CANARY}.docx" source-language="en-US" target-language="ru-RU" datatype="plaintext">
    <body>
      <trans-unit id="{CANARY}-u1"><source>{tagged("Never mix the two cleaning agents.")}</source>
        <target>{tagged("Смешайте два чистящих средства.")}</target></trans-unit>
      <trans-unit id="{CANARY}-u2"><source>{tagged("The tank holds 120 litres.")}</source>
        <target>{tagged("Бак вмещает 210 литров.")}</target></trans-unit>
    </body>
  </file>
</xliff>""".encode()

# Needles must exist only at runtime: Sentry attaches source-code context lines to stack
# frames, so any string literal in this file can legitimately show up in an error event.
# Real dev-split segments (never in any source file) cover leaks of untagged text.
REAL = []
try:
    with open("data/splits/en-ru.dev.jsonl", encoding="utf-8") as f:
        REAL = [{"src": r["src"], "mt": r["mt"]} for r in map(json.loads, f) if len(r["src"]) > 40][:2]
except FileNotFoundError:
    print("note: data/splits/en-ru.dev.jsonl not found; checking tagged segments only")
NEEDLES = [CANARY, UUID_PART] + [r[k] for r in REAL for k in ("src", "mt")]


def boom(src: str, mt: str, lp: str):
    raise ValueError(f"cannot check segment {src!r} -> {mt!r}")  # text in an exception message


def run_probes() -> dict:
    out = {}
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/api/review", json={"segments": SEGMENTS + REAL, "lp": "en-ru"})
        assert r.status_code == 200, r.text
        out["json"] = r.json()
        r = client.post(f"/api/review?filename={CANARY}.xlf&threshold=0.5", content=XLIFF,
                        headers={"content-type": "application/xml"})
        assert r.status_code == 200, r.text
        out["xliff"] = r.json()
        r = client.post("/api/review", json={"segments": [{"src": CANARY}], "note": CANARY})
        assert r.status_code == 422, r.status_code
        with mock.patch.object(checks, "negation_check", boom):
            r = client.post("/api/review", json={"segments": SEGMENTS[:1], "lp": "en-ru"})
        assert r.status_code == 500, r.status_code

    log = logging.getLogger("secondread.canary")
    log.warning("reviewing segment %s", CANARY)                    # breadcrumb with params
    log.error(f"segment failed: {SEGMENTS[0]['src']}")              # error event, f-string
    if REAL:
        log.error("real segment failed: %s", REAL[0]["mt"])
    sentry_sdk.add_breadcrumb(category="secondread", message=CANARY, data={"src": CANARY})
    sentry_sdk.capture_message(f"probe {CANARY}")
    try:
        raise RuntimeError(f"probe {SEGMENTS[1]['mt']}")
    except RuntimeError:
        sentry_sdk.capture_exception()

    async def offline() -> dict:
        agent = ReviewAgent("http://127.0.0.1:9")  # nothing listens on port 9
        try:
            return (await agent.review(SEGMENTS, lp="en-ru")).to_dict()
        finally:
            await agent.aclose()
    out["offline"] = asyncio.run(offline())
    sentry_sdk.flush()
    return out


# ---- 3. envelope inspection ------------------------------------------------------------------

def items(envelopes):
    for env in envelopes:
        for item in env.items:
            raw = item.payload.get_bytes()
            try:
                doc = json.loads(raw)
            except ValueError:
                doc = None
            yield item.headers.get("type"), raw, doc


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def spans(typ, doc):
    """(op, data) for transaction spans (v1) and streamed gen_ai span items (v2)."""
    if typ == "transaction":
        trace = (doc.get("contexts") or {}).get("trace") or {}
        yield trace.get("op"), trace.get("data") or {}
        for s in doc.get("spans") or []:
            yield s.get("op"), s.get("data") or {}
    elif typ == "span":
        for s in doc.get("items") or []:
            attrs = {k: v.get("value") if isinstance(v, dict) else v for k, v in (s.get("attributes") or {}).items()}
            yield attrs.get("sentry.op"), attrs


def leaks(envelopes) -> Counter:
    hits = Counter()
    for env in envelopes:
        blob = env.serialize()
        for n in NEEDLES:
            hits[n] += blob.count(n.encode())
    for _, _, doc in items(envelopes):
        for s in strings(doc):
            for n in NEEDLES:
                hits[n] += s.count(n)
    return hits


def main() -> int:
    results = run_probes()
    envs = list(capture.envelopes)
    types = Counter(t for t, _, _ in items(envs))
    ops, tools, chat_tokens, chat_example = Counter(), set(), [], None
    error_values, transactions = [], Counter()
    for typ, _, doc in items(envs):
        if not isinstance(doc, dict):
            continue
        if typ == "transaction":
            transactions[doc.get("transaction")] += 1
        if typ == "event":
            for e in (doc.get("exception") or {}).get("values") or []:
                error_values.append((e.get("type"), e.get("value")))
            if doc.get("logentry"):
                error_values.append(("logentry", doc["logentry"].get("message")))
        for op, data in spans(typ, doc):
            ops[op] += 1
            if op == "gen_ai.execute_tool":
                tools.add(data.get("gen_ai.tool.name"))
            if op == "gen_ai.chat" and data.get("gen_ai.usage.input_tokens"):
                chat_tokens.append(data["gen_ai.usage.input_tokens"])
                chat_example = chat_example or {k: v for k, v in sorted(data.items()) if k.startswith(("gen_ai", "secondread"))}

    hits = leaks(envs)
    s = results["json"]["summary"]
    print(f"canary: {CANARY}")
    print(f"envelopes: {len(envs)}  items: {dict(types)}")
    print(f"transactions: {dict(transactions)}")
    print(f"span ops: {dict(ops)}")
    print(f"tools traced: {sorted(t for t in tools if t)}")
    print(f"error events (type, value): {error_values}")
    print(f"example gen_ai.chat span data: {json.dumps(chat_example, ensure_ascii=False)}")
    print(f"json review: {s['segments']} segments, {s['flagged']} flagged, model {s['model']}, "
          f"{s['timing_ms']['per_segment']} ms/segment, model call p50 {s['timing_ms']['model_call_p50']} ms")
    for seg in results["json"]["segments"] + results["xliff"]["segments"]:
        print(f"   risk {seg['risk']:.2f} flagged={seg['flagged']} issues={seg['issues']}")
    print(f"blocked outbound attempts: {BLOCKED}")
    print(f"needle hits: {sum(hits.values())} " + json.dumps({k[:40]: v for k, v in hits.items() if v}, ensure_ascii=False))

    failures = []
    if sum(hits.values()):
        failures.append(f"segment text found in envelopes: {dict(+hits)}")
    if BLOCKED:
        failures.append(f"outbound network attempted: {BLOCKED}")
    expected_tools = {"parse_xliff", "number_check", "negation_check", "lora_score"}
    if not expected_tools <= tools:
        failures.append(f"missing tool spans: {expected_tools - tools}")
    if ops["gen_ai.invoke_agent"] < 3 or len(chat_tokens) < 5:
        failures.append(f"capture looks vacuous: {dict(ops)}, chat spans with tokens: {len(chat_tokens)}")
    if not any(t == "ValueError" for t, _ in error_values):
        failures.append("the unhandled exception was not captured, so its scrubbing is untested")
    if ("LlamaServerError", "llama_unreachable") not in error_values:
        failures.append("SafeError message was not kept (or the offline probe did not report)")
    if results["xliff"]["summary"]["segments"] != 2:
        failures.append(f"xliff probe parsed {results['xliff']['summary']['segments']} segments, expected 2")

    # ---- 4. control: with scrubbing off the same search must find the canary ----------------
    control = CaptureTransport()
    init_sentry(dsn=FAKE_DSN, transport=control, force=True, before_send=None,
                before_send_transaction=None, before_breadcrumb=None)
    sentry_sdk.capture_exception(RuntimeError(CANARY))
    sentry_sdk.flush()
    control_hits = sum(leaks(control.envelopes).values())
    print(f"control (scrubbers off): {control_hits} needle hits (must be > 0)")
    if not control_hits:
        failures.append("control found nothing: the detector itself is broken")

    print("PASS: canary appears in 0 envelopes" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
