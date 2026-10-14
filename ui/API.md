# SecondRead UI ↔ backend contract

The UI was written against the shapes below. `secondread/server.py` implements them (checked
against it on 2026-10-14: JSON pairs, multipart XLIFF 1.2 / 2.0 / SDLXLIFF, health, errors).
The UI is lenient (see "Tolerated variations"). If the backend changes, `ui/src/api.ts` is the
only file that parses responses.

The UI talks to the backend through Vite's `server.proxy` (`/api` → `http://127.0.0.1:8787`),
so requests are same-origin: no CORS and no preflight for the Sentry `sentry-trace` / `baggage`
headers. In production the backend can serve `ui/dist` itself under the same origin.

## `GET /api/health`

Called on page load, every 8 s while the backend is down and every 30 s while it is up.

```json
{ "status": "ok", "offline": true, "model": "Qwen3.5-4B-Q4_K_M", "adapter": "secondread-lora v1",
  "adapter_version": "v1", "scoring": "LoRA v1", "operating_point": "review",
  "operating_point_source": "file", "thresholds": { "en-es": 0.4984, "en-ru": 0.1857 },
  "threshold_source": "dev_review",
  "operating_points": {
    "review": { "thresholds": { "en-es": 0.4984, "en-ru": 0.1857 }, "source": "dev_review", "target_recall": 0.6 },
    "f1":     { "thresholds": { "en-es": 0.7019, "en-ru": 0.1609 }, "source": "dev_f1" } },
  "llama": { "ok": true, "lora_adapters": [ { "id": 0, "name": "secondread-lora", "scale": 1.0,
             "sha256": "5337b2b7…", "version": "v1" } ] }, "sentry": { "enabled": false } }
```

| field            | type                   | notes                                                           |
|------------------|------------------------|-----------------------------------------------------------------|
| `status`         | `"ok"` \| `"degraded"` | `degraded` = model server down: header says "rule checks only"  |
| `offline`        | boolean                | optional, default `true`; `false` drops the "runs on this Mac" claim |
| `model`          | string                 | optional, shown as secondary detail in the header               |
| `adapter`        | string \| null         | optional; `null` means the base model runs without the LoRA. The backend sends the file name plus its version, e.g. `secondread-lora v1` |
| `adapter_version`| string \| null         | not read by the UI. Found by the adapter file's sha256 in `secondread/thresholds.json`, so a renamed file keeps its version; `unknown` for a file not listed there |
| `operating_point`| `"review"` \| `"f1"`  | not read by the UI. The flag line a request without `threshold` or `operating_point` gets. `review` (default): the highest threshold with dev recall of Major ≥ 0.60; `f1`: the F1-optimal one. Both are picked on the fused dev risk (`scripts/pick_thresholds.py`) |
| `operating_point_source` | string         | not read by the UI. `env` (`SECONDREAD_OPERATING_POINT`), `file` (`default_operating_point` in `secondread/thresholds.json`), `default` (built-in `review`) or `argument` |
| `thresholds`     | object                 | not read by the UI. `{lp: threshold}` at the active operating point, used when a request sends none; `threshold_source` says where they come from (`dev_review`, `dev_f1`, `env`, `fallback`, or `dev_<point>+fallback` when some pairs have no calibrated value) |
| `operating_points` | object               | not read by the UI. `{point: {thresholds, source, target_recall?}}`: what each operating point would apply to the loaded adapter |
| `rule_calibration` | object               | not read by the UI. `{method, fitted_as, generated, pairs}` of `secondread/rule_calibration.json`, the train-measured weights of the rule findings in `risk` |
| `llama`          | object                 | optional; `base_model` / `lora_adapters[].name` are used if `model` / `adapter` are missing |
| `sentry.enabled` | boolean                | optional; `true` adds "Sentry on, no text sent" to the header   |

Any HTTP 200 with JSON counts as reachable. A network error, non-200 or non-JSON reply shows
"Checker not running".

## `POST /api/review`

### Request, form 1: JSON (pasted pairs)

```http
Content-Type: application/json

{ "lp": "en-ru", "segments": [ { "id": "1", "src": "…", "mt": "…" } ] }
```

- `lp`: `"en-ru"` or `"en-es"` (the language pairs the model was trained on).
- `operating_point` (optional): `"review"` or `"f1"`, the dev-calibrated flag line to use; any
  other value is a 400 `bad_operating_point`. Default: the server's (`GET /api/health`). A
  `threshold` in the request overrides both. The multipart form and the query string
  (`?operating_point=f1`) take the same field; the UI does not send it yet.
- `id`: opaque string, not necessarily unique. Pasted pairs are numbered `"1"`, `"2"`, … in paste
  order, unless the paste has three columns whose first looks like an id (`A7`, `12`, `seg-3`), in
  which case that id is sent. The UI keys segments by position, never by id.
- The UI shows segments in the order the response returns them.

### Request, form 2: multipart (XLIFF file)

```http
Content-Type: multipart/form-data; boundary=…

xliff = <file>   (XLIFF 1.2 / 2.0 / SDLXLIFF; field name is exactly "xliff")
lp    = "en-ru"  (the UI pre-fills it from the file's source/target language when it can)
```

The backend parses the file (`parse_xliff` tool span) and uses the `<trans-unit>` / `<unit>` /
`mrk mid` ids as segment ids. The UI shows an id next to `#N` when it is 24 characters or
shorter (SDLXLIFF GUIDs are hidden). The file name is only sent inside the multipart body,
never in the URL, so it stays out of access logs and trace URLs.

### Response (both forms)

```json
{
  "lp": "en-ru",
  "segments": [
    {
      "id": "10",
      "src": "… (didn't make me close until months after I started …)",
      "mt": "… (я сблизился с ним только через несколько месяцев …)",
      "risk": 0.833,
      "signals": {
        "number_check":   { "flagged": false, "src_only": [], "mt_only": [] },
        "negation_check": { "flagged": true, "src_count": 1, "mt_count": 0 },
        "lora_score": 0.583
      },
      "reasons": [
        { "code": "negation_mismatch",
          "message": "1 negation(s) in source, 0 in MT",
          "spans": [ { "side": "src", "start": 87, "end": 93 } ] },
        { "code": "model_risk", "message": "Model P(Major) 58%", "spans": [] }
      ]
    }
  ],
  "timing_ms": 1840
}
```

| field                       | type                  | notes |
|-----------------------------|-----------------------|-------|
| `segments`                  | array                 | **in document order**; the UI numbers them #1…#N in this order |
| `segments[].risk`           | number in [0, 1]      | the final fused score; the UI sorts by it and never recomputes it. logit(risk) = logit(`lora_score`) + the train-calibrated `weight` of the strongest finding of each rule check (`secondread/rulecal.py`); without findings it equals `lora_score` |
| `signals.lora_score`        | number \| null        | `p(Major) / (p(Major) + p(OK))` from the first generated token; `null` if not run |
| `signals.number_check`      | object \| bool \| null | `flagged` drives nothing by itself; reasons do. `src_only` / `mt_only` are numbers as strings |
| `signals.negation_check`    | object \| bool \| null | `src_count` / `mt_count` = negation markers found on each side |
| `reasons[].code`            | string                | known: `number_mismatch`, `negation_mismatch`, `model_risk`; others shown as-is |
| `reasons[].message`         | string                | optional, short, shown under the segment |
| `reasons[].spans`           | array                 | optional; `side` is `"src"` or `"mt"` |
| `reasons[].confidence`      | number \| null        | not read by the UI. Rule findings: the train P(Major \| this finding) for the pair (`secondread/rule_calibration.json`); `model_risk`: the model risk |
| `reasons[].weight`          | number                | rule findings only, not read by the UI: the log-odds the finding adds to the model risk (≥ 0; often small). The finding is listed and highlighted whatever its weight |
| `spans[].start` / `.end`    | int                   | **Unicode code-point offsets** (Python `str` indices), end exclusive |
| `timing_ms`                 | number \| object      | total wall time; an object like `{ "total": 1840, … }` is also accepted |
| `segments[].flagged`        | boolean               | optional; the backend's verdict (`risk ≥ threshold`) |
| `summary`                   | object                | optional; the UI reads `flagged`, `threshold`, `model_available`, `lora_active`, `xliff.skipped`, `timing_ms.total` |
| `summary.threshold`         | number \| null        | the flag threshold. Without a `threshold` in the request it is the dev-calibrated one for the active adapter, operating point and language pair (`secondread/thresholds.json`); `null` when a JSON request mixes pairs with different thresholds |
| `summary.operating_point`   | `"review"` \| `"f1"`  | the operating point of this review (the request's, else the server's). With `threshold_source` `request` or `env`, the threshold did not come from it |
| `summary.thresholds`        | object                | `{lp: threshold}` actually applied, one entry per language pair in the request |
| `summary.threshold_source`  | string                | `request`, `env` (`SECONDREAD_THRESHOLD`), `dev_review` or `dev_f1` (the calibration file at that operating point), `fallback` (0.5: unknown adapter, a LoRA scale other than 0 or 1, no model, or no value for that point) or `dev_<point>+fallback` |
| `summary.adapter_version`   | string \| null        | e.g. `v1` when the LoRA is active, else `null` |

`reasons` is what gets highlighted. A segment with no reasons can still rank high on `risk`, and
every rule finding is listed, even when its calibrated weight barely moves `risk`.
A `model_risk` reason only gets a chip and a note when `lora_score ≥ summary.threshold`
(0.5 when the summary has none), because the risk bar already shows lower scores. Extra fields (`issue`,
`confidence`, `weight`, `model`, `meta`, `index`) are ignored.

### Errors

Non-2xx with `{ "detail": "…", "code": "…" }` (`{ "error": { "message" } }` is accepted too).
The UI shows the status and the detail locally. It never forwards the detail to Sentry, because
it may quote segment text. A 502/503/504 that is not JSON (the Vite proxy when the backend is
down) is shown as "Can't reach the local checker".

## Demo file: `ui/public/demo-results.json`

Same shape as the `/api/review` response, plus two demo-only fields:

- `meta`: `{ placeholder, title, note, attribution, lp, doc, system, generated }`. When
  `placeholder` is true the UI shows a "Placeholder scores" banner with `note`.
- `segments[].gold`: `{ label: "Major" | "OK", source, spans: [{ side, text, category, severity }] }`,
  the human MQM annotation, shown under each segment for comparison.

The current file is real `secondread/server.py` output (`scripts/make_demo_results.py --api …`,
2026-10-14 evening) for 10 consecutive segments of one WMT22 en-ru test document, scored with
LoRA v1 active and the en-ru dev threshold (0.1609), so `placeholder` is false: 7 of 10 flagged,
3 of the 4 gold Major segments among them. Without a backend, the script's default mode writes a
stand-in (regex checks + llama-server score) in the same format.

## Tolerated variations

So the UI keeps working while the backend is in flux, `ui/src/api.ts` also accepts:

- `reasons` as plain strings: codes (`"number_mismatch"`) become chips, sentences become notes;
- codes are grouped by prefix: `number_*` / `unit_*` → number, `negation_*` → negation,
  `model_*` / `lora` → model risk (except `model_unavailable`); a separate `issues` code list works too;
- `signals.*` or `checks.*` with `{ risk, issues: [codes], skipped }` instead of `flagged`;
- `number_check` / `negation_check` as plain booleans;
- `risk` missing, in which case it falls back to `lora_score`, then to 0;
- spans outside the text, which are clamped or dropped.

If a finding has no spans, the UI highlights what it names instead: the numbers in
`number_check.src_only` / `mt_only` or in the number message, and the quoted words in a negation
message (`The source has a negation ('not') …`).

## Privacy

- The browser never calls a model. All scoring runs in the local backend.
- Optional Sentry (`VITE_SENTRY_DSN`) gets errors and timings only. The Console integration is
  not installed and `beforeBreadcrumb` drops `console` anyway; breadcrumbs are allow-listed
  (`fetch`, `xhr`, `navigation`, `ui.click`); HTTP bodies, cookies, query strings and stack
  locals are not collected. Every event and span is scrubbed: on-screen segment texts are
  replaced, and error messages also lose quoted fragments and any word of 5+ characters that
  occurs in a segment. No segment text, `detail` or XLIFF content is sent.
- Browser canary test (2026-10-14, fake ingest on localhost): `SR-CANARY-<uuid>` in a pasted pair,
  then `console.log/error` of it, an uncaught `Error` and an unhandled rejection quoting it, a
  `JSON.parse` error on it, a click on the segment, and a 500 whose `detail` quotes it. Result:
  4 error events and the pageload/fetch spans arrived, the canary appeared in none of them, and
  `sentry-trace` / `baggage` reached the backend through the proxy.
