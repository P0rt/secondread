// Turning what the translator gives us (pasted columns, an XLIFF file) into request inputs.

import type { PairInput } from './api';

export const LANGUAGE_PAIRS = ['en-ru', 'en-es'] as const;
export type LanguagePair = (typeof LANGUAGE_PAIRS)[number];

export interface PasteParse {
  pairs: PairInput[];
  /** 1-based line numbers that had text but no MT column. */
  badLines: number[];
}

/** A short token without spaces in the first of three columns is taken as a segment id. */
const ID_COLUMN = /^[\w.:#-]{1,40}$/;

/**
 * One pair per line: `source<TAB>machine translation`, which is what copying two columns from a
 * spreadsheet produces. Three columns are read as `id, source, MT` (a common CAT-tool export)
 * when the first one looks like an id. ` ||| ` (Moses style) is accepted instead of a Tab.
 */
export function parsePastedPairs(text: string): PasteParse {
  const pairs: PairInput[] = [];
  const badLines: number[] = [];
  text.split(/\r?\n/).forEach((line, i) => {
    if (!line.trim()) return;
    let cols = line.split('\t').map((c) => c.trim());
    if (cols.length === 1) cols = line.split(' ||| ').map((c) => c.trim());
    while (cols.length > 2 && cols[cols.length - 1] === '') cols.pop();
    let id = String(pairs.length + 1);
    if (cols.length >= 3 && ID_COLUMN.test(cols[0])) id = cols.shift()!;
    const [src = '', mt = ''] = cols;
    if (!src || !mt) {
      badLines.push(i + 1);
      return;
    }
    pairs.push({ id, src, mt });
  });
  return { pairs, badLines };
}

export const XLIFF_ACCEPT = '.xlf,.xliff,.sdlxliff,.mqxliff,application/x-xliff+xml,application/xml,text/xml';

export function looksLikeXliff(file: File): boolean {
  return /\.(xlf|xliff|sdlxliff|mqxliff)$/i.test(file.name) || /xml|xliff/.test(file.type);
}

/**
 * Reads the language pair from the XLIFF header without parsing the whole file:
 * XLIFF 1.2 / SDLXLIFF use source-language/target-language, XLIFF 2.x uses srcLang/trgLang.
 * Returns e.g. "en-ru", or null when the attributes are missing.
 */
export async function sniffLanguagePair(file: File): Promise<string | null> {
  const head = await file.slice(0, 8192).text();
  const attr = (names: string) => head.match(new RegExp(`\\b(?:${names})\\s*=\\s*["']([A-Za-z]{2,3})`, 'i'))?.[1];
  const s = attr('source-language|srcLang');
  const t = attr('target-language|trgLang');
  return s && t ? `${s}-${t}`.toLowerCase() : null;
}

export function isSupportedPair(lp: string | null): lp is LanguagePair {
  return lp !== null && (LANGUAGE_PAIRS as readonly string[]).includes(lp);
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
