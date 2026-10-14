// Optional error and performance reporting. Off unless VITE_SENTRY_DSN is set at build time, in
// which case @sentry/browser is loaded as a separate chunk. Nothing here may send segment text:
// the browser never calls a model, so Sentry only ever sees errors, timings and URLs.

import type { Breadcrumb, ErrorEvent } from '@sentry/browser';

let client: { captureMessage: typeof import('@sentry/browser').captureMessage } | null = null;

/** Texts currently on screen. Any occurrence in an outgoing event is replaced before sending. */
let sensitive: string[] = [];
/** Their longer words, so a fragment of a segment inside an error message is caught too. */
let sensitiveWords = new Set<string>();

/** Breadcrumb categories we keep. console and ui.input are the ones that can carry typed text. */
const BREADCRUMB_ALLOW = new Set(['fetch', 'xhr', 'navigation', 'ui.click']);

export async function initSentry(): Promise<void> {
  const dsn = import.meta.env.VITE_SENTRY_DSN;
  if (!dsn) return;
  const { start } = await import('./sentry-client');
  client = start(dsn, import.meta.env.MODE, {
    beforeBreadcrumb: (crumb: Breadcrumb) => {
      if (crumb.category === 'console') return null;
      return crumb.category && BREADCRUMB_ALLOW.has(crumb.category) ? crumb : null;
    },
    beforeSend: scrubEvent,
    scrub,
  });
}

/** Called whenever new segments are shown, so the scrubber knows what must never leave. */
export function setSensitiveTexts(texts: string[]): void {
  sensitive = texts.filter((t) => t.length >= 4).sort((a, b) => b.length - a.length);
  sensitiveWords = new Set(texts.flatMap((t) => t.match(WORD) ?? []).filter((w) => [...w].length >= MIN_WORD));
}

/** Report an error by a safe, text-free message. `tags` must be plain metadata. */
export function reportError(message: string, tags: Record<string, string | number> = {}): void {
  client?.captureMessage(message, { level: 'error', tags });
}

function scrubEvent(event: ErrorEvent): ErrorEvent {
  delete event.extra;
  if (event.request) {
    delete event.request.data;
    delete event.request.query_string;
    delete event.request.cookies;
  }
  return scrub(event);
}

// Quoted fragments in error messages, e.g. Chrome's `Unexpected token 'x', "Нет, вы н"... is
// not valid JSON`, can carry pieces of a response body.
const QUOTED = /"[^"\n]{2,}"|'[^'\n]{2,}'|«[^»\n]{2,}»|“[^”\n]{2,}”/g;
/** A "word" for fragment matching: a run without spaces or sentence punctuation. */
const WORD = /[^\s"'«»“”„.,;:!?()[\]{}<>]+/gu;
const MIN_WORD = 5;
const URL_QUERY = /((?:https?:\/\/[^\s?#"]*|\/api\/[^\s?#"]*))\?[^\s#"]*/g;

/**
 * Walks the event and replaces on-screen segment texts in every string. Error and log messages
 * (`value` / `message` keys) are scrubbed harder: quoted fragments and any word that occurs in a
 * segment are removed as well, since exceptions can quote partial text.
 */
function scrub<T>(event: T): T {
  const clean = (s: string, isMessage: boolean): string => {
    for (const t of sensitive) if (s.includes(t)) s = s.split(t).join('[segment]');
    // Query strings (e.g. ?filename= on the XLIFF upload) never leave the page.
    s = s.replace(URL_QUERY, '$1');
    if (!isMessage) return s;
    s = s.replace(QUOTED, '"[redacted]"');
    return sensitiveWords.size ? s.replace(WORD, (w) => (sensitiveWords.has(w) ? '[…]' : w)) : s;
  };
  const walk = (v: unknown, key: string, depth: number): unknown => {
    if (depth > 12) return v;
    if (key === 'http.query' || key === 'query_string') return undefined;
    if (typeof v === 'string') return clean(v, key === 'value' || key === 'message');
    if (Array.isArray(v)) return v.map((x) => walk(x, key, depth + 1));
    if (v && typeof v === 'object') {
      const o = v as Record<string, unknown>;
      for (const k of Object.keys(o)) o[k] = walk(o[k], k, depth + 1);
    }
    return v;
  };
  return walk(event, '', 0) as T;
}
