"""Local SecondRead API: FastAPI on 127.0.0.1:8787 (the Vite UI proxies /api here, so no CORS).

    GET  /api/health   llama-server reachability, model, LoRA adapters, Sentry on/off
    POST /api/review   one of
        application/json     {"segments": [{"src": "...", "mt": "...", "lp"?: "en-ru", "id"?: "..."}],
                              "lp"?: "en-ru", "lora_scale"?: 1.0, "threshold"?: 0.5}
        multipart/form-data  field "xliff" = the file; optional fields "lp", "lora_scale", "threshold"
                             (what the UI sends; parsed with the stdlib, python-multipart is not needed)
        raw XLIFF body       Content-Type application/xml, text/xml, application/x-xliff+xml or
                             application/octet-stream
        Query parameters work for every form: ?lp=en-ru&lora_scale=1&threshold=0.5
        &filename=job.sdlxliff&include_locked=false (body fields win over query parameters).
    Response and error shapes: ui/API.md. Errors are {"detail": "...", "code": "..."}.

Run:  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python -m secondread.server
Sentry is off unless SENTRY_DSN is set (see secondread.telemetry).
"""
from __future__ import annotations

import os

from secondread.telemetry import init_sentry, sentry_enabled

init_sentry()  # before FastAPI is imported and the app is built, so the integration hooks in

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import HTTP
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from secondread.agent import MAX_SEGMENTS, ReviewAgent, ReviewError

log = logging.getLogger("secondread.server")

HOST = os.getenv("SECONDREAD_HOST", "127.0.0.1")
PORT = int(os.getenv("SECONDREAD_PORT", "8787"))
MAX_BODY_BYTES = 50 * 1024 * 1024
XML_TYPES = {"application/xml", "text/xml", "application/x-xliff+xml", "application/xliff+xml",
             "application/octet-stream"}


class SegmentIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    src: str
    mt: str
    lp: str | None = None
    id: str | int | None = None


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segments: list[SegmentIn] = Field(max_length=MAX_SEGMENTS)
    lp: str | None = None
    lora_scale: float | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = ReviewAgent()
    await asyncio.to_thread(agent.warm_up)  # tokenizer load, ~6 s the first time
    info = await agent.health()
    log.info("llama-server %s, model %s", "up" if info["ok"] else "DOWN", info.get("model"))
    app.state.agent = agent
    yield
    await agent.aclose()


app = FastAPI(title="SecondRead", version="0.1.0", lifespan=lifespan, docs_url="/api/docs",
              openapi_url="/api/openapi.json")


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"detail": message, "code": code, **extra}, status_code=status)


def _multipart(content_type: str, body: bytes) -> tuple[bytes, str | None, dict[str, str]]:
    """multipart/form-data -> (the "xliff" file, its file name, the other fields as text).

    Parsed with the stdlib MIME parser, so python-multipart is not needed.
    """
    head = f"Content-Type: {content_type}\r\n\r\n".encode("latin-1")
    msg = BytesParser(policy=HTTP).parsebytes(head + body)
    if not msg.is_multipart():
        raise ReviewError("bad_request", "malformed multipart body")
    data, fname, fields = None, None, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if name == "xliff":
            data, fname = payload, part.get_filename()
        elif name:
            fields[str(name)] = payload.decode("utf-8", "replace").strip()
    if data is None:
        raise ReviewError("bad_request", "multipart field 'xliff' (the file) is required")
    return data, fname, fields


def _opt_float(v: str | None, name: str) -> float | None:
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        raise ReviewError("bad_request", f"'{name}' must be a number") from None


def _first(*values: Any) -> Any:
    """Body fields win over query parameters."""
    return next((v for v in values if v is not None), None)


@app.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    llama = await request.app.state.agent.health()
    adapters = llama.get("lora_adapters") or []
    return {"status": "ok" if llama["ok"] else "degraded", "offline": True,
            "model": llama.get("base_model"), "adapter": adapters[0]["name"] if adapters else None,
            "llama": llama, "sentry": {"enabled": sentry_enabled()}, "xliff": _xliff_available()}


def _xliff_available() -> bool:
    try:
        import secondread.xliff  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


@app.post("/api/review", openapi_extra={"requestBody": {"required": True, "content": {
    "application/json": {"schema": ReviewIn.model_json_schema()},
    "multipart/form-data": {"schema": {"type": "object", "required": ["xliff"], "properties": {
        "xliff": {"type": "string", "format": "binary"}, "lp": {"type": "string"},
        "lora_scale": {"type": "number"}, "threshold": {"type": "number"}}}},
    "application/xml": {"schema": {"type": "string", "format": "binary"}}}}})
async def review(request: Request,
                 lp: str | None = Query(None, description="language pair, e.g. en-ru (XLIFF: overrides the file)"),
                 lora_scale: float | None = Query(None, description="per-request LoRA scale (0 = base model)"),
                 threshold: float | None = Query(None, ge=0.0, le=1.0),
                 filename: str | None = Query(None, description="original file name; helps detect .sdlxliff"),
                 include_locked: bool = Query(False, description="also score locked segments")):
    agent: ReviewAgent = request.app.state.agent
    raw_ctype = request.headers.get("content-type") or ""
    ctype = raw_ctype.split(";")[0].strip().lower()
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return _error(413, "too_large", f"request body over {MAX_BODY_BYTES // 2**20} MB")
    try:
        if ctype == "application/json":
            try:
                req = ReviewIn.model_validate_json(body)
            except ValidationError as e:
                return _error(422, "bad_request", "invalid JSON body",
                              errors=json.loads(e.json(include_input=False, include_url=False)))
            result = await agent.review([s.model_dump() for s in req.segments], lp=_first(req.lp, lp),
                                        lora_scale=_first(req.lora_scale, lora_scale),
                                        threshold=_first(req.threshold, threshold))
        elif ctype in XML_TYPES or ctype.endswith("+xml"):
            result = await agent.review(body, lp=lp, lora_scale=lora_scale, threshold=threshold,
                                        filename=filename, include_locked=include_locked)
        elif ctype == "multipart/form-data":
            data, fname, form = _multipart(raw_ctype, body)
            result = await agent.review(data, lp=_first(form.get("lp") or None, lp),
                                        lora_scale=_first(_opt_float(form.get("lora_scale"), "lora_scale"), lora_scale),
                                        threshold=_first(_opt_float(form.get("threshold"), "threshold"), threshold),
                                        filename=fname or filename, include_locked=include_locked)
        else:
            return _error(415, "unsupported_media_type", "use application/json or an XML content type")
    except ReviewError as e:
        return _error(e.status, e.code, e.message)
    return result.to_dict()


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
