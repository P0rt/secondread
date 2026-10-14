// Backend calls and response normalisation. This is the only module that knows the wire format
// (see ../API.md); everything else works with the normalised types below.
//
// Two response shapes are accepted: the one secondread/server.py returns (reasons as sentences,
// machine codes in `issues`, model score in `model`, timings in `summary`), and the shape the UI
// was first written against (structured reasons with spans, `signals`, top-level `timing_ms`),
// which the demo file uses.

export type Side = 'src' | 'mt';
export type ReasonKind = 'number' | 'negation' | 'model' | 'other';

/** Highlight range in code-point offsets (Python str indices), end exclusive. */
export interface Span {
  side: Side;
  start: number;
  end: number;
}

/** One finding, shown as a chip. */
export interface Reason {
  kind: ReasonKind;
  code: string;
  spans: Span[];
}

/** A human-readable line under the segment. It may quote the text, so it never leaves the page. */
export interface Note {
  kind: ReasonKind;
  text: string;
  /** A short message from a structured reason, shown after its kind ("Negation: …"). */
  labelled: boolean;
}

/** A string named by a finding (a number, a negation word) to highlight when no spans are given. */
export interface Hint {
  kind: ReasonKind;
  token: string;
  numeric: boolean;
}

export interface NumberCheck {
  flagged: boolean;
  srcOnly: string[];
  mtOnly: string[];
}

export interface NegationCheck {
  flagged: boolean;
}

export interface GoldLabel {
  label: string;
  source?: string;
  spans: { side?: string; text: string; category?: string; severity?: string }[];
}

export interface Segment {
  /** 1-based position in the document; stable across sort orders. */
  index: number;
  id: string;
  src: string;
  mt: string;
  risk: number;
  /** Backend verdict (risk ≥ its threshold), when it sends one. */
  flagged?: boolean;
  numberCheck: NumberCheck | null;
  negationCheck: NegationCheck | null;
  loraScore: number | null;
  reasons: Reason[];
  notes: Note[];
  hints: Hint[];
  gold?: GoldLabel;
}

export interface DemoMeta {
  placeholder: boolean;
  title?: string;
  note?: string;
  attribution?: string;
}

export interface ReviewResult {
  origin: 'api' | 'demo';
  lp?: string;
  segments: Segment[];
  timingMs?: number;
  flaggedCount?: number;
  threshold?: number;
  /** e.g. "base model, no LoRA" when the fine-tuned adapter was not used. */
  modelNote?: string;
  /** e.g. "3 skipped (untranslated or locked)". */
  skippedNote?: string;
  meta?: DemoMeta;
}

export interface Health {
  /** The backend answered and says it can score. */
  ok: boolean;
  /** The backend answered but its model server is down (rule checks only). */
  degraded: boolean;
  offline: boolean;
  model?: string;
  /** Adapter name(s); null means none loaded; undefined means unknown. */
  adapter?: string | null;
  telemetry?: boolean;
}

export interface PairInput {
  id: string;
  src: string;
  mt: string;
}

/**
 * Error with a message that is safe to show and to report: it never contains segment text.
 * `detail` holds the backend's own message, which may quote text, and stays in the page.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail?: string;
  constructor(message: string, status = 0, detail?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

const REVIEW_URL = '/api/review';
const HEALTH_URL = '/api/health';

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const res = await fetch(HEALTH_URL, { headers: { accept: 'application/json' }, signal });
  if (!res.ok) throw new ApiError(`health check failed (${res.status})`, res.status);
  const o = asRecord(await readJson(res));
  const llama = asRecord(o.llama);
  const adapters = Array.isArray(llama.lora_adapters)
    ? llama.lora_adapters.map((a) => asRecord(a).name).filter((n): n is string => typeof n === 'string')
    : undefined;
  const ok = o.status === undefined ? o.ok !== false : o.status === 'ok';
  return {
    ok,
    degraded: !ok && (o.status === 'degraded' || llama.ok === false),
    offline: o.offline !== false,
    model: str(o.model) ?? str(llama.base_model) ?? str(llama.model),
    adapter: typeof o.adapter === 'string' || o.adapter === null ? o.adapter : adapters && (adapters.join(', ') || null),
    telemetry: asRecord(o.sentry).enabled === true ? true : undefined,
  };
}

export function reviewPairs(lp: string, pairs: PairInput[], signal?: AbortSignal): Promise<ReviewResult> {
  return postReview({
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'application/json' },
    body: JSON.stringify({ lp, segments: pairs }),
    signal,
  });
}

/**
 * Sends the file as multipart/form-data: the file in the `xliff` field, the language pair in
 * `lp`. The file name travels inside the multipart body, never in the URL, so it stays out of
 * access logs and trace URLs.
 */
export function reviewXliff(lp: string, file: File, signal?: AbortSignal): Promise<ReviewResult> {
  const form = new FormData();
  form.append('xliff', file, file.name);
  form.append('lp', lp);
  // No content-type header: the browser sets the multipart boundary itself.
  return postReview({ method: 'POST', headers: { accept: 'application/json' }, body: form, signal });
}

/** Precomputed results shipped with the static build (public/demo-results.json). */
export async function loadDemo(): Promise<ReviewResult> {
  const res = await fetch(`${import.meta.env.BASE_URL}demo-results.json`, { cache: 'no-cache' });
  if (!res.ok) throw new ApiError(`demo file missing (${res.status})`, res.status);
  return normalizeReview(await readJson(res), 'demo');
}

async function postReview(init: RequestInit): Promise<ReviewResult> {
  let res: Response;
  try {
    res = await fetch(REVIEW_URL, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError('could not reach the local checker');
  }
  if (!res.ok) {
    // A proxy (Vite's, or a reverse proxy) answers 502/503/504 in plain text when the backend
    // is down. That is "not running", not a backend error.
    const isJson = (res.headers.get('content-type') ?? '').includes('json');
    if (!isJson && [502, 503, 504].includes(res.status)) throw new ApiError('could not reach the local checker');
    let detail: string | undefined;
    try {
      detail = errorDetail(await res.json());
    } catch {
      /* body was not JSON; the status is enough */
    }
    throw new ApiError(`review request failed (${res.status})`, res.status, detail);
  }
  return normalizeReview(await readJson(res), 'api');
}

/** `{"detail": "…", "code": "…"}` (FastAPI style), or `{"error": {"message"}}`. */
function errorDetail(body: unknown): string | undefined {
  const o = asRecord(body);
  const msg = asRecord(o.error).message;
  if (typeof msg === 'string') return msg;
  if (typeof o.detail === 'string') return o.detail;
  return o.detail === undefined ? undefined : JSON.stringify(o.detail);
}

/** Response.json() can throw a SyntaxError that quotes the body; replace it with a clean error. */
async function readJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    throw new ApiError(`response was not JSON (${res.status})`, res.status);
  }
}

// ---------------------------------------------------------------------------------------------
// Normalisation

type Rec = Record<string, unknown>;
const asRecord = (v: unknown): Rec => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Rec) : {});
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | undefined => (typeof v === 'string' && v ? v : undefined);
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const strings = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);

export function normalizeReview(raw: unknown, origin: ReviewResult['origin']): ReviewResult {
  const o = asRecord(raw);
  if (!Array.isArray(o.segments)) throw new ApiError('unexpected response: no segments array');
  const segments = o.segments.map((s, i) => normalizeSegment(asRecord(s), i));
  const sum = asRecord(o.summary);
  const m = asRecord(o.meta);

  const skipped = Object.values(asRecord(asRecord(sum.xliff).skipped)).reduce<number>((a, v) => a + (num(v) ?? 0), 0);
  let modelNote: string | undefined;
  if (sum.model_available === false) modelNote = 'model unavailable, rule checks only';
  else if (sum.lora_active === false) modelNote = 'base model, no LoRA';

  const flaggedSegs = segments.filter((s) => s.flagged !== undefined);
  return {
    origin,
    lp: str(o.lp) ?? str(asRecord(o.segments[0]).lp),
    segments,
    timingMs: num(o.timing_ms) ?? num(asRecord(o.timing_ms).total) ?? num(asRecord(sum.timing_ms).total) ?? undefined,
    flaggedCount: num(sum.flagged) ?? (flaggedSegs.length ? flaggedSegs.filter((s) => s.flagged).length : undefined),
    threshold: num(sum.threshold) ?? undefined,
    modelNote,
    skippedNote: skipped > 0 ? `${skipped} skipped (untranslated or locked)` : undefined,
    meta:
      o.meta === undefined
        ? undefined
        : { placeholder: m.placeholder === true, title: str(m.title), note: str(m.note), attribution: str(m.attribution) },
  };
}

/** Machine codes look like `number_changed`; anything else in `reasons` is a sentence. */
const CODE = /^[a-z][a-z0-9_]*$/;

function kindOfCode(code: string): ReasonKind {
  const c = code.toLowerCase();
  if (c === 'model_unavailable') return 'other';
  if (c.startsWith('number') || c.startsWith('unit')) return 'number';
  if (c.startsWith('negation')) return 'negation';
  if (c.startsWith('model') || c === 'lora') return 'model';
  return 'other';
}

function kindOfText(text: string): ReasonKind {
  if (/^model\b/i.test(text)) return 'model';
  if (/negation/i.test(text)) return 'negation';
  if (/\b(number|unit)\b/i.test(text)) return 'number';
  return 'other';
}

function normalizeSegment(s: Rec, i: number): Segment {
  const src = typeof s.src === 'string' ? s.src : '';
  const mt = typeof s.mt === 'string' ? s.mt : '';
  const sig = asRecord(s.signals);
  const checks = asRecord(s.checks);
  const lora = num(sig.lora_score) ?? num(asRecord(s.model).risk);
  const lens = { src: [...src].length, mt: [...mt].length };

  const reasons: Reason[] = [];
  const notes: Note[] = [];
  for (const r of Array.isArray(s.reasons) ? s.reasons : []) {
    if (typeof r === 'string') {
      if (CODE.test(r)) reasons.push({ kind: kindOfCode(r), code: r, spans: [] });
      else notes.push({ kind: kindOfText(r), text: r, labelled: false });
      continue;
    }
    const o = asRecord(r);
    const code = str(o.code) ?? 'other';
    const kind = kindOfCode(code);
    reasons.push({ kind, code, spans: normalizeSpans(o.spans, lens) });
    if (typeof o.message === 'string') notes.push({ kind, text: o.message, labelled: true });
  }
  for (const code of strings(s.issues)) {
    if (!reasons.some((r) => r.code === code)) reasons.push({ kind: kindOfCode(code), code, spans: [] });
  }

  const numberCheck = normalizeCheck(sig.number_check ?? checks.number_check);
  const negationCheck = normalizeCheck(sig.negation_check ?? checks.negation_check);
  return {
    index: i + 1,
    id: s.id === undefined || s.id === null ? String(i + 1) : String(s.id),
    src,
    mt,
    risk: clamp01(num(s.risk) ?? lora ?? 0),
    flagged: typeof s.flagged === 'boolean' ? s.flagged : undefined,
    numberCheck: numberCheck && { flagged: numberCheck.flagged, srcOnly: numberCheck.srcOnly, mtOnly: numberCheck.mtOnly },
    negationCheck: negationCheck && { flagged: negationCheck.flagged },
    loraScore: lora === null ? null : clamp01(lora),
    reasons,
    notes,
    hints: hintsFrom(notes, numberCheck),
    gold: normalizeGold(s.gold),
  };
}

function normalizeSpans(v: unknown, lens: Record<Side, number>): Span[] {
  const spans: Span[] = [];
  for (const sp of Array.isArray(v) ? v : []) {
    const x = asRecord(sp);
    const side = x.side === 'src' || x.side === 'mt' ? x.side : null;
    const start = num(x.start);
    const end = num(x.end);
    if (!side || start === null || end === null) continue;
    const a = Math.max(0, Math.min(start, lens[side]));
    const b = Math.max(0, Math.min(end, lens[side]));
    if (b > a) spans.push({ side, start: a, end: b });
  }
  return spans;
}

/**
 * Accepts `true`/`false`, `{flagged, src_only, mt_only}` (first shape) or
 * `{risk, issues: [codes], skipped}` (secondread/server.py).
 */
function normalizeCheck(v: unknown): NumberCheck | null {
  if (typeof v === 'boolean') return { flagged: v, srcOnly: [], mtOnly: [] };
  if (!v || typeof v !== 'object') return null;
  const o = v as Rec;
  const srcOnly = strings(o.src_only);
  const mtOnly = strings(o.mt_only);
  const flagged = o.flagged === true || srcOnly.length + mtOnly.length > 0 || strings(o.issues).length > 0;
  return { flagged, srcOnly, mtOnly };
}

const NUMBER_TOKEN = /\d+(?:[.,]\d+)*/g;
/** Quoted terms in a sentence like "The source has a negation ('not', 'never') but …". */
const QUOTED_TERM = /'(.+?)'(?=,\s|\)|$)/g;

function hintsFrom(notes: Note[], numberCheck: NumberCheck | null): Hint[] {
  const out = new Map<string, Hint>();
  const add = (kind: ReasonKind, token: string, numeric: boolean) => out.set(`${kind}:${token}`, { kind, token, numeric });
  for (const t of [...(numberCheck?.srcOnly ?? []), ...(numberCheck?.mtOnly ?? [])]) add('number', t, true);
  for (const n of notes) {
    if (n.kind === 'number') for (const m of n.text.matchAll(NUMBER_TOKEN)) add('number', m[0], true);
    if (n.kind === 'negation') for (const m of n.text.matchAll(QUOTED_TERM)) add('negation', m[1], false);
  }
  return [...out.values()];
}

function normalizeGold(v: unknown): GoldLabel | undefined {
  const o = asRecord(v);
  if (typeof o.label !== 'string') return undefined;
  return {
    label: o.label,
    source: str(o.source),
    spans: (Array.isArray(o.spans) ? o.spans : []).map((sp) => {
      const x = asRecord(sp);
      return { side: str(x.side), text: str(x.text) ?? '', category: str(x.category), severity: str(x.severity) };
    }),
  };
}
