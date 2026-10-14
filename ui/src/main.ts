import './style.css';
import { ApiError, fetchHealth, loadDemo, reviewPairs, reviewXliff, type Health, type ReviewResult } from './api';
import {
  formatBytes,
  isSupportedPair,
  looksLikeXliff,
  parsePastedPairs,
  sniffLanguagePair,
  XLIFF_ACCEPT,
} from './input';
import { byRisk, h, overview, segmentItem, summaryText, topCount, type View } from './render';
import { initSentry, reportError, setSensitiveTexts } from './sentry';

/** Static build for hosting without a backend: no /api calls, demo loads on start. */
const DEMO_ONLY = import.meta.env.VITE_DEMO_ONLY === '1';

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const ui = {
  status: $('#status'),
  statusText: $('#status .status-text'),
  statusDetail: $('#status .status-detail'),
  lp: $<HTMLSelectElement>('#lp'),
  liveInput: $('#live-input'),
  tabFile: $<HTMLButtonElement>('#tab-file'),
  tabPaste: $<HTMLButtonElement>('#tab-paste'),
  panelFile: $('#panel-file'),
  panelPaste: $('#panel-paste'),
  drop: $('#drop'),
  file: $<HTMLInputElement>('#file'),
  paste: $<HTMLTextAreaElement>('#paste'),
  pasteInfo: $('#paste-info'),
  checkPaste: $<HTMLButtonElement>('#check-paste'),
  demoBtn: $<HTMLButtonElement>('#demo-btn'),
  demoLine: $('#demo-line'),
  demoOnlyNote: $('#demo-only-note'),
  results: $('#results'),
};

interface State {
  result: ReviewResult | null;
  /** What the result came from, for the results header ("report.xlf", "12 pasted pairs"). */
  label: string;
  view: View;
  inflight: AbortController | null;
  health: 'checking' | 'ok' | 'down';
}
const state: State = { result: null, label: '', view: 'document', inflight: null, health: 'checking' };

// --------------------------------------------------------------------------------------------
// Backend status in the header

type StatusKind = 'checking' | 'ok' | 'warn' | 'down' | 'demo';

function setStatus(kind: StatusKind, text: string, detail = ''): void {
  ui.status.className = `status status-${kind}`;
  ui.statusText.textContent = text;
  ui.statusDetail.textContent = detail;
}

function describeHealth(hl: Health): void {
  const adapter = hl.adapter === null ? 'no LoRA loaded' : hl.adapter;
  const detail = [hl.model, adapter, hl.telemetry ? 'Sentry on, no text sent' : ''].filter(Boolean).join(' · ');
  if (hl.degraded) setStatus('warn', 'Model server down · rule checks only', detail);
  else if (!hl.ok) setStatus('warn', 'Checker starting…', detail);
  else if (hl.offline) setStatus('ok', 'Offline · runs on this Mac', detail);
  else setStatus('warn', 'Local checker (online mode)', detail);
}

let healthTimer: number | undefined;
async function refreshHealth(): Promise<void> {
  window.clearTimeout(healthTimer);
  try {
    const hl = await fetchHealth(AbortSignal.timeout(4000));
    // Degraded still serves rule checks, so it counts as up; only poll faster until it is ok.
    state.health = hl.ok || hl.degraded ? 'ok' : 'down';
    describeHealth(hl);
  } catch {
    state.health = 'down';
    setStatus('down', 'Checker not running', 'start the backend on 127.0.0.1:8787');
  }
  // Poll faster while the backend is down so the header flips as soon as it comes up.
  healthTimer = window.setTimeout(refreshHealth, state.health === 'ok' ? 30_000 : 8_000);
}

// --------------------------------------------------------------------------------------------
// Running a check

async function run(label: string, pending: string, job: (signal: AbortSignal) => Promise<ReviewResult>) {
  state.inflight?.abort();
  const ctrl = new AbortController();
  state.inflight = ctrl;
  showPending(pending, ctrl);
  try {
    const result = await job(ctrl.signal);
    if (ctrl.signal.aborted) return;
    state.result = result;
    state.label = result.origin === 'demo' && result.meta?.title ? `Demo: ${result.meta.title}` : label;
    state.view = 'document';
    setSensitiveTexts(result.segments.flatMap((s) => [s.src, s.mt]));
    renderResults();
    // Bring the results into view only if they start below the fold.
    if (ui.results.getBoundingClientRect().top > window.innerHeight * 0.6) {
      ui.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  } catch (err) {
    if (ctrl.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
      renderResults(); // back to whatever was shown before
      return;
    }
    showError(err);
  } finally {
    if (state.inflight === ctrl) state.inflight = null;
    ui.results.setAttribute('aria-busy', 'false');
  }
}

function showPending(text: string, ctrl: AbortController): void {
  ui.results.setAttribute('aria-busy', 'true');
  const card = h('div', 'card pending');
  const row = h('div', 'pending-row');
  row.append(h('span', 'pending-text', text));
  const cancel = h('button', 'btn btn-quiet', 'Cancel');
  cancel.type = 'button';
  cancel.addEventListener('click', () => ctrl.abort());
  row.append(cancel);
  card.append(row, h('div', 'progress'));
  ui.results.replaceChildren(card);
}

function showError(err: unknown): void {
  const card = h('div', 'card error');
  card.setAttribute('role', 'alert');
  let title = 'Something went wrong.';
  let hint = '';
  if (err instanceof ApiError) {
    if (err.status === 0 && /reach/.test(err.message)) {
      title = 'Can’t reach the local checker.';
      hint = 'Start the SecondRead backend (it listens on 127.0.0.1:8787), or open the demo.';
      void refreshHealth();
    } else if (err.status === 413) {
      title = 'That file is too large for the checker.';
    } else if (err.status === 415 || err.status === 422 || err.status === 400) {
      title = 'The checker could not read this input.';
      hint = err.detail ?? '';
    } else {
      title = `The checker returned an error (${err.status || 'no status'}).`;
      hint = err.detail ?? '';
    }
    // Only the text-free message and the status leave the page.
    reportError(err.message, { status: err.status });
  } else {
    reportError('unexpected error while checking', { kind: err instanceof Error ? err.name : typeof err });
  }
  card.append(h('p', 'error-title', title));
  if (hint) card.append(h('p', 'error-hint', hint.length > 400 ? `${hint.slice(0, 400)}…` : hint));
  ui.results.replaceChildren(card);
}

// --------------------------------------------------------------------------------------------
// Results

function renderResults(): void {
  const r = state.result;
  if (!r) {
    ui.results.replaceChildren();
    return;
  }
  const segs = r.segments;
  const lp = (r.lp ?? ui.lp.value).toLowerCase();
  const [srcLang = 'en', mtLang = 'ru'] = lp.split('-');
  const k = topCount(segs.length);
  const ranked = byRisk(segs);
  const top = new Set(ranked.slice(0, k).map((s) => s.index));

  const out: HTMLElement[] = [];

  if (r.origin === 'demo') {
    const b = h('div', `banner ${r.meta?.placeholder ? 'banner-warn' : ''}`);
    b.append(h('strong', undefined, r.meta?.placeholder ? 'Demo · placeholder scores. ' : 'Demo · precomputed results. '));
    if (r.meta?.note) b.append(r.meta.note, ' ');
    if (r.meta?.attribution) b.append(h('span', 'banner-attr', r.meta.attribution));
    out.push(b);
  }

  const head = h('div', 'results-head');
  const titles = h('div', 'results-titles');
  titles.append(h('h2', undefined, state.label || 'Results'), h('p', 'summary', summaryText(r)));
  head.append(titles, viewToggle(k, segs.length));
  out.push(head);

  if (segs.length === 0) {
    out.push(h('p', 'empty', 'No segments found. Check that the file has translated targets.'));
    ui.results.replaceChildren(...out);
    return;
  }

  if (segs.length > 1) out.push(overview(segs, top, showInDocument));
  out.push(legend());

  const list = h('ol', 'segs');
  const shown = state.view === 'top' ? ranked.slice(0, k) : segs;
  if (state.view === 'top') {
    const note = h('p', 'view-note');
    note.textContent = `The ${k} highest-risk segment${k === 1 ? '' : 's'} (top 10%), highest first. Read these before the rest.`;
    out.push(note);
  }
  const frag = document.createDocumentFragment();
  for (const s of shown) {
    frag.append(
      segmentItem(s, {
        srcLang,
        mtLang,
        modelThreshold: r.threshold ?? 0.5,
        inTop: top.has(s.index),
        view: state.view,
        onShowInDocument: showInDocument,
      }),
    );
  }
  list.append(frag);
  out.push(list);
  ui.results.replaceChildren(...out);
}

function viewToggle(k: number, n: number): HTMLElement {
  const g = h('div', 'seg-ctrl');
  g.setAttribute('role', 'group');
  g.setAttribute('aria-label', 'View');
  const make = (v: View, text: string) => {
    const b = h('button', 'seg-ctrl-btn', text);
    b.type = 'button';
    b.setAttribute('aria-pressed', String(state.view === v));
    b.addEventListener('click', () => {
      if (state.view === v) return;
      state.view = v;
      renderResults();
    });
    return b;
  };
  g.append(make('document', 'Document order'), make('top', `Review top 10% (${k} of ${n})`));
  return g;
}

function legend(): HTMLElement {
  const l = h('p', 'legend');
  const item = (cls: string, text: string) => {
    const s = h('span', 'legend-item');
    s.append(h('span', `legend-swatch ${cls}`), text);
    return s;
  };
  l.append(item('hl-number', 'Number mismatch'), item('hl-negation', 'Negation'), item('legend-model', 'Model risk = P(Major) from the model'));
  return l;
}

function showInDocument(index: number): void {
  if (state.view !== 'document') {
    state.view = 'document';
    renderResults();
  }
  const li = ui.results.querySelector<HTMLElement>(`.seg[data-index="${index}"]`);
  if (!li) return;
  li.scrollIntoView({ behavior: 'smooth', block: 'center' });
  li.focus({ preventScroll: true });
  li.classList.remove('seg-flash');
  void li.offsetWidth; // restart the animation
  li.classList.add('seg-flash');
}

// --------------------------------------------------------------------------------------------
// Input wiring

function selectTab(which: 'file' | 'paste'): void {
  const file = which === 'file';
  ui.tabFile.setAttribute('aria-selected', String(file));
  ui.tabPaste.setAttribute('aria-selected', String(!file));
  ui.tabFile.tabIndex = file ? 0 : -1;
  ui.tabPaste.tabIndex = file ? -1 : 0;
  ui.panelFile.hidden = !file;
  ui.panelPaste.hidden = file;
  if (!file) ui.paste.focus();
}

async function handleFile(file: File | undefined): Promise<void> {
  if (!file) return;
  if (!looksLikeXliff(file)) {
    showError(new ApiError('not an xliff file', 400, 'Choose an .xlf, .xliff or .sdlxliff file.'));
    return;
  }
  const sniffed = await sniffLanguagePair(file).catch(() => null);
  if (sniffed && !isSupportedPair(sniffed)) {
    const card = h('div', 'card error');
    card.setAttribute('role', 'alert');
    card.append(
      h('p', 'error-title', `This file is ${sniffed}. SecondRead checks English → Spanish and English → Russian.`),
      h('p', 'error-hint', 'Other language pairs are not supported by the model yet.'),
    );
    ui.results.replaceChildren(card);
    return;
  }
  if (isSupportedPair(sniffed)) ui.lp.value = sniffed;
  const lp = ui.lp.value;
  void run(file.name, `Checking ${file.name} (${formatBytes(file.size)})…`, (signal) => reviewXliff(lp, file, signal));
}

function updatePasteInfo(): void {
  const { pairs, badLines } = parsePastedPairs(ui.paste.value);
  ui.checkPaste.disabled = pairs.length === 0;
  ui.pasteInfo.classList.toggle('hint-warn', badLines.length > 0);
  if (!ui.paste.value.trim()) {
    ui.pasteInfo.textContent = 'One pair per line: source, Tab, machine translation.';
  } else if (badLines.length) {
    const lines = badLines.slice(0, 3).join(', ') + (badLines.length > 3 ? '…' : '');
    ui.pasteInfo.textContent = `${pairs.length} pair${pairs.length === 1 ? '' : 's'} ready · line ${lines} has no Tab between source and MT`;
  } else {
    ui.pasteInfo.textContent = `${pairs.length} pair${pairs.length === 1 ? '' : 's'} ready`;
  }
}

function submitPaste(): void {
  const { pairs } = parsePastedPairs(ui.paste.value);
  if (!pairs.length) return;
  const lp = ui.lp.value;
  const n = pairs.length;
  void run(`${n} pasted pair${n === 1 ? '' : 's'}`, `Checking ${n} segment${n === 1 ? '' : 's'}…`, (signal) =>
    reviewPairs(lp, pairs, signal),
  );
}

function openDemo(): void {
  void run('Demo', 'Loading demo results…', () => loadDemo());
}

function wireInput(): void {
  ui.file.accept = XLIFF_ACCEPT;
  ui.tabFile.addEventListener('click', () => selectTab('file'));
  ui.tabPaste.addEventListener('click', () => selectTab('paste'));
  for (const t of [ui.tabFile, ui.tabPaste]) {
    t.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      const next = t === ui.tabFile ? ui.tabPaste : ui.tabFile;
      selectTab(next === ui.tabFile ? 'file' : 'paste');
      next.focus();
    });
  }

  ui.file.addEventListener('change', () => {
    void handleFile(ui.file.files?.[0]);
    ui.file.value = ''; // allow choosing the same file again
  });
  // Drag and drop onto the drop zone; dropping anywhere else on the page must not navigate away.
  for (const ev of ['dragover', 'drop'] as const) window.addEventListener(ev, (e) => e.preventDefault());
  ui.drop.addEventListener('dragenter', () => ui.drop.classList.add('drop-over'));
  ui.drop.addEventListener('dragover', (e) => {
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
    ui.drop.classList.add('drop-over');
  });
  ui.drop.addEventListener('dragleave', (e) => {
    if (!ui.drop.contains(e.relatedTarget as Node | null)) ui.drop.classList.remove('drop-over');
  });
  ui.drop.addEventListener('drop', (e) => {
    ui.drop.classList.remove('drop-over');
    void handleFile(e.dataTransfer?.files?.[0]);
  });

  ui.paste.addEventListener('input', updatePasteInfo);
  ui.paste.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      submitPaste();
    }
  });
  ui.checkPaste.addEventListener('click', submitPaste);
  ui.demoBtn.addEventListener('click', openDemo);
}

// --------------------------------------------------------------------------------------------

function start(): void {
  void initSentry().catch(() => {
    /* reporting is optional; never break the app over it */
  });
  wireInput();

  if (DEMO_ONLY) {
    setStatus('demo', 'Static demo · no backend', 'precomputed results');
    ui.liveInput.hidden = true;
    ui.demoLine.hidden = true;
    ui.demoOnlyNote.hidden = false;
    $('#input-title').textContent = 'About this demo';
    ui.lp.closest('label')!.hidden = true;
    openDemo();
    return;
  }
  void refreshHealth();
  window.addEventListener('focus', () => {
    if (state.health !== 'ok') void refreshHealth();
  });
  if (new URLSearchParams(location.search).has('demo')) openDemo();
}

start();
