// DOM rendering for review results. All segment text goes in through textContent / text nodes,
// never innerHTML, because XLIFF content is untrusted.

import type { Hint, Reason, ReasonKind, ReviewResult, Segment, Side } from './api';

export type View = 'document' | 'top';

/** Visual risk bands. Only for colour; ranking always uses the raw score. */
export type RiskLevel = 'low' | 'mid' | 'high';
export const riskLevel = (r: number): RiskLevel => (r >= 0.6 ? 'high' : r >= 0.3 ? 'mid' : 'low');

/** Size of the "review first" set: the top 10% by risk, at least one segment. */
export const topCount = (n: number): number => (n === 0 ? 0 : Math.max(1, Math.ceil(n * 0.1)));

/** Highest risk first; ties keep document order so the list is stable. */
export function byRisk(segs: Segment[]): Segment[] {
  return [...segs].sort((a, b) => b.risk - a.risk || a.index - b.index);
}

const REASON_LABEL: Record<ReasonKind, string> = {
  number: 'Number mismatch',
  negation: 'Negation',
  model: 'Model risk',
  other: 'Check',
};

const LANG_NAME: Record<string, string> = { en: 'EN', ru: 'RU', es: 'ES', de: 'DE' };

type El = keyof HTMLElementTagNameMap;
export function h<K extends El>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export const pct = (x: number): string => `${Math.round(x * 100)}%`;

// --------------------------------------------------------------------------------------------
// Highlighting

interface Mark {
  start: number;
  end: number;
  kind: ReasonKind;
}

/** Earlier kinds win where highlights overlap. */
const MARK_PRIORITY: ReasonKind[] = ['number', 'negation', 'other', 'model'];

const escapeRe = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * Code-point offsets of every occurrence of a hint token. Numbers must not sit inside a longer
 * number; words must not sit inside a longer word (except contractions like "n't").
 */
function findToken(text: string, hint: Hint): Mark[] {
  const t = escapeRe(hint.token);
  const re = hint.numeric
    ? new RegExp(`(?<![\\d.,])${t}(?!\\d)`, 'g')
    : /^\p{L}/u.test(hint.token) && !/^n['’]t$/i.test(hint.token)
      ? new RegExp(`(?<![\\p{L}\\p{N}])${t}(?![\\p{L}\\p{N}])`, 'giu')
      : new RegExp(t, 'giu');
  const out: Mark[] = [];
  for (const m of text.matchAll(re)) {
    const start = [...text.slice(0, m.index)].length;
    out.push({ start, end: start + [...m[0]].length, kind: hint.kind });
  }
  return out;
}

function marksFor(seg: Segment, side: Side): Mark[] {
  const marks: Mark[] = [];
  const withSpans = new Set<ReasonKind>();
  for (const r of seg.reasons) {
    if (r.spans.length) withSpans.add(r.kind);
    for (const sp of r.spans) if (sp.side === side) marks.push({ start: sp.start, end: sp.end, kind: r.kind });
  }
  // Fallback for findings without spans: look for the numbers or words they name.
  const flaggedKinds = new Set(seg.reasons.map((r) => r.kind));
  const text = side === 'src' ? seg.src : seg.mt;
  for (const hint of seg.hints) {
    if (flaggedKinds.has(hint.kind) && !withSpans.has(hint.kind)) marks.push(...findToken(text, hint));
  }
  return marks;
}

/** Text with <mark>s, split on code points so offsets from Python line up. */
function highlighted(text: string, marks: Mark[]): DocumentFragment {
  const frag = document.createDocumentFragment();
  const cps = Array.from(text);
  if (marks.length === 0) {
    frag.append(text);
    return frag;
  }
  const kindAt: (ReasonKind | null)[] = new Array(cps.length).fill(null);
  for (const m of marks) {
    for (let i = Math.max(0, m.start); i < Math.min(cps.length, m.end); i++) {
      const cur = kindAt[i];
      if (cur === null || MARK_PRIORITY.indexOf(m.kind) < MARK_PRIORITY.indexOf(cur)) kindAt[i] = m.kind;
    }
  }
  let i = 0;
  while (i < cps.length) {
    const kind = kindAt[i];
    let j = i + 1;
    while (j < cps.length && kindAt[j] === kind) j++;
    const chunk = cps.slice(i, j).join('');
    if (kind === null) frag.append(chunk);
    else frag.append(h('mark', `hl hl-${kind}`, chunk));
    i = j;
  }
  return frag;
}

// --------------------------------------------------------------------------------------------
// Pieces

export function meter(risk: number): HTMLElement {
  const m = h('div', `meter meter-${riskLevel(risk)}`);
  m.setAttribute('role', 'meter');
  m.setAttribute('aria-valuemin', '0');
  m.setAttribute('aria-valuemax', '100');
  m.setAttribute('aria-valuenow', String(Math.round(risk * 100)));
  m.setAttribute('aria-label', 'Risk');
  const fill = h('span', 'meter-fill');
  fill.style.width = `${Math.max(2, risk * 100)}%`;
  m.append(fill);
  return m;
}

function chip(kind: ReasonKind, code: string): HTMLElement {
  const label = kind === 'other' ? code.replace(/_/g, ' ') : REASON_LABEL[kind];
  const c = h('li', `chip chip-${kind}`);
  c.append(h('span', 'chip-dot'), label);
  return c;
}

function pane(side: Side, seg: Segment, lang: string): HTMLElement {
  const p = h('div', `pane pane-${side}`);
  const name = LANG_NAME[lang] ?? lang.toUpperCase();
  p.append(h('span', 'pane-label', side === 'src' ? `Source · ${name}` : `MT · ${name}`));
  const t = h('p', 'pane-text');
  t.lang = lang;
  t.append(highlighted(side === 'src' ? seg.src : seg.mt, marksFor(seg, side)));
  p.append(t);
  return p;
}

function goldLine(seg: Segment): HTMLElement | null {
  const g = seg.gold;
  if (!g) return null;
  const p = h('p', `gold gold-${g.label === 'Major' ? 'major' : 'ok'}`);
  p.append(h('span', 'gold-src', `Human label${g.source ? ` (${g.source})` : ''}:`), ' ');
  p.append(h('strong', undefined, g.label));
  // Only a Major label lists its error spans; minor or fluency notes would just add noise.
  const majors = g.label === 'Major' ? g.spans.filter((s) => s.text) : [];
  if (majors.length) {
    p.append(' — ');
    majors.forEach((s, i) => {
      if (i) p.append('; ');
      p.append((s.category ?? 'error').replace(/_/g, ' '));
      // Omissions are often marked on a bracket or comma; quoting that would say nothing.
      if (/[\p{L}\p{N}]/u.test(s.text)) p.append(' ', h('q', undefined, s.text));
    });
  }
  return p;
}

export interface SegmentOpts {
  srcLang: string;
  mtLang: string;
  /** Model scores below this get no "Model risk" chip or note; the risk bar already shows them. */
  modelThreshold: number;
  inTop: boolean;
  view: View;
  onShowInDocument?: (index: number) => void;
}

export function segmentItem(seg: Segment, o: SegmentOpts): HTMLLIElement {
  const li = h('li', `seg seg-${riskLevel(seg.risk)}${o.inTop ? ' seg-top' : ''}`);
  li.dataset.index = String(seg.index);
  li.tabIndex = -1;

  const head = h('div', 'seg-head');
  const num = h('span', 'seg-num', `#${seg.index}`);
  // The XLIFF segment id helps find the segment in the CAT tool; skip it when it is just the index.
  // Long ids (SDLXLIFF GUIDs) are noise in this view.
  if (seg.id !== String(seg.index) && seg.id.length <= 24) num.append(h('span', 'seg-id', ` · ${seg.id}`));
  head.append(num, meter(seg.risk), h('span', 'seg-risk', pct(seg.risk)));
  // A backend may attach a model reason to every segment; it only means something when high.
  const modelHigh = seg.loraScore === null || seg.loraScore >= o.modelThreshold;
  const reasons = seg.reasons.filter((r) => r.kind !== 'model' || modelHigh);
  const notes = seg.notes.filter((n) => n.kind !== 'model' || modelHigh);
  // One chip per kind of finding (number_changed and unit_changed are both "Number mismatch").
  const chips = new Map<string, Reason>();
  for (const r of reasons) chips.set(r.kind === 'other' ? `other:${r.code}` : r.kind, r);
  if (chips.size) {
    const ul = h('ul', 'chips');
    ul.setAttribute('aria-label', 'Reasons');
    const order = (r: Reason) => MARK_PRIORITY.indexOf(r.kind);
    for (const r of [...chips.values()].sort((a, b) => order(a) - order(b))) ul.append(chip(r.kind, r.code));
    head.append(ul);
  }
  const tail = h('div', 'seg-tail');
  if (o.inTop && o.view === 'document') tail.append(h('span', 'badge', 'Top 10%'));
  if (o.view === 'top' && o.onShowInDocument) {
    const b = h('button', 'link-btn', 'Show in document');
    b.type = 'button';
    b.addEventListener('click', () => o.onShowInDocument!(seg.index));
    tail.append(b);
  }
  if (tail.childElementCount) head.append(tail);
  li.append(head);

  const body = h('div', 'seg-body');
  body.append(pane('src', seg, o.srcLang), pane('mt', seg, o.mtLang));
  li.append(body);

  if (notes.length) {
    const ul = h('ul', 'notes');
    for (const r of notes) {
      const n = h('li', `note note-${r.kind}`);
      const label = r.kind === 'other' ? '' : REASON_LABEL[r.kind];
      // Skip the label when the message already starts with it ("Number 5 …", "Model …").
      const firstWord = label.split(' ')[0].toLowerCase();
      if (r.labelled && label && !r.text.toLowerCase().startsWith(firstWord)) {
        n.append(h('span', 'note-k', `${label}:`), ' ');
      }
      n.append(r.text);
      ul.append(n);
    }
    li.append(ul);
  }
  const gold = goldLine(seg);
  if (gold) li.append(gold);
  return li;
}

/**
 * Document overview: one bar per segment in document order, height = risk. Clicking a bar jumps
 * to that segment. Drawn as a single SVG so it scales to thousands of segments.
 */
export function overview(segs: Segment[], top: Set<number>, onPick: (index: number) => void): HTMLElement {
  const wrap = h('figure', 'overview');
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  const n = segs.length;
  svg.setAttribute('viewBox', `0 0 ${n} 100`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `Risk across the document, ${n} segments`);
  const gap = n <= 60 ? 0.18 : 0; // a small gap between bars only while they are wide enough
  for (const s of segs) {
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', `ov-bar ov-${riskLevel(s.risk)}${top.has(s.index) ? ' ov-top' : ''}`);
    const hit = document.createElementNS(NS, 'rect');
    hit.setAttribute('class', 'ov-hit');
    hit.setAttribute('x', String(s.index - 1));
    hit.setAttribute('y', '0');
    hit.setAttribute('width', '1');
    hit.setAttribute('height', '100');
    const bar = document.createElementNS(NS, 'rect');
    const hgt = Math.max(3, s.risk * 100);
    bar.setAttribute('x', String(s.index - 1 + gap / 2));
    bar.setAttribute('y', String(100 - hgt));
    bar.setAttribute('width', String(1 - gap));
    bar.setAttribute('height', String(hgt));
    const title = document.createElementNS(NS, 'title');
    title.textContent = `#${s.index} · risk ${pct(s.risk)}`;
    g.append(hit, bar, title);
    g.addEventListener('click', () => onPick(s.index));
    svg.append(g);
  }
  const cap = h('figcaption', 'overview-cap');
  cap.append(h('span', undefined, 'Start'), h('span', undefined, 'Risk by position in the document'), h('span', undefined, 'End'));
  wrap.append(svg, cap);
  return wrap;
}

export function summaryText(r: ReviewResult): string {
  const n = r.segments.length;
  const parts = [`${n} segment${n === 1 ? '' : 's'}`];
  if (r.flaggedCount !== undefined) {
    const at = r.threshold !== undefined ? ` (risk ≥ ${Math.round(r.threshold * 100)}%)` : '';
    parts.push(`${r.flaggedCount} flagged${at}`);
  } else {
    const ruleHits = r.segments.filter((s) => s.reasons.some((x) => x.kind === 'number' || x.kind === 'negation')).length;
    parts.push(`${ruleHits} with a rule hit`);
  }
  if (r.timingMs !== undefined) parts.push(`checked in ${(r.timingMs / 1000).toFixed(r.timingMs < 10000 ? 1 : 0)} s`);
  if (r.modelNote) parts.push(r.modelNote);
  if (r.skippedNote) parts.push(r.skippedNote);
  return parts.join(' · ');
}
