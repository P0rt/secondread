"""Ranking / detection metrics for scored segments, with document-level bootstrap CIs.

Shared by scripts/probe_local.py, scripts/eval_tinker.py and scripts/report.py.

A *scored row* is one JSON object per segment, as written by those scripts:
    {"id", "label": "Major"|"OK", "risk": float in [0, 1], "doc", "lp", ...}

Every metric takes 0/1 labels, scores and optional non-negative per-row weights.
Weights are what make the bootstrap cheap: resampling documents with replacement
is the same as weighting each row by how many times its document was drawn, so
one bootstrap replicate costs a few vectorised passes instead of a re-sort of a
materialised resample.

Ties are handled deterministically (never by input order):
  * AUROC counts a tied positive/negative pair as 1/2 (Mann-Whitney).
  * PR-AUC is average precision over distinct thresholds (sklearn's definition).
  * recall@k% takes a tie group at the cut-off fractionally, i.e. the expected
    recall under a random tie break.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

POSITIVE = "Major"


# ---------------------------------------------------------------- data helpers
def load_scored(path: str) -> list[dict]:
    """Rows of a jsonl file (scored output, or any split file)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_arrays(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(labels as 0/1 int array, risk as float array) from scored rows."""
    y = np.fromiter((r["label"] == POSITIVE for r in rows), dtype=np.int64, count=len(rows))
    s = np.fromiter((float(r["risk"]) for r in rows), dtype=np.float64, count=len(rows))
    return y, s


def group_keys(rows: Sequence[dict]) -> list[str]:
    """Bootstrap unit per row: (lp, doc). Rows without a doc fall back to their own id,
    which degrades to a plain segment-level bootstrap for those rows."""
    return [f"{r.get('lp', '')}|{r['doc']}" if r.get("doc") is not None else f"id|{r['id']}" for r in rows]


def _prep(y, s, w):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    if not (len(y) == len(s) == len(w)):
        raise ValueError("labels, scores and weights must have the same length")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return y, s, w


def _groups_desc(y, s, w):
    """Weighted positive/negative mass per distinct score, highest score first."""
    uniq, inv = np.unique(s, return_inverse=True)  # ascending
    pos = np.bincount(inv, weights=w * y, minlength=len(uniq))[::-1]
    neg = np.bincount(inv, weights=w * (1 - y), minlength=len(uniq))[::-1]
    return uniq[::-1], pos, neg


# ---------------------------------------------------------------- point metrics
def auroc(y, s, w=None) -> float:
    y, s, w = _prep(y, s, w)
    _, pos, neg = _groups_desc(y, s, w)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    neg_below = N - np.cumsum(neg)  # negatives strictly below each (descending) group
    return float((pos * (neg_below + 0.5 * neg)).sum() / (P * N))


def average_precision(y, s, w=None) -> float:
    """PR-AUC as average precision: sum_k (R_k - R_{k-1}) * P_k over distinct thresholds."""
    y, s, w = _prep(y, s, w)
    _, pos, neg = _groups_desc(y, s, w)
    P = pos.sum()
    if P == 0:
        return float("nan")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    return float((pos / P * precision).sum())


pr_auc = average_precision


def recall_at(y, s, frac: float = 0.10, w=None) -> float:
    """Share of all Major rows found when a reviewer reads the top `frac` of rows by risk.
    k = round(n * frac), at least 1 (n = total weight), matching probe_local's original rule."""
    y, s, w = _prep(y, s, w)
    _, pos, neg = _groups_desc(y, s, w)
    P, n = pos.sum(), w.sum()
    if P == 0:
        return float("nan")
    k = max(1.0, round(n * frac))
    size = pos + neg
    before = np.cumsum(size) - size  # mass ranked above each group
    take = np.clip((k - before) / np.where(size > 0, size, 1), 0.0, 1.0)  # fraction of each group read
    return float((take * pos).sum() / P)


def prf_at(y, s, threshold: float, w=None) -> dict[str, float]:
    """Precision / recall / F1 when flagging every row with risk >= threshold."""
    y, s, w = _prep(y, s, w)
    flag = s >= threshold
    tp = float((w * y * flag).sum())
    fp = float((w * (1 - y) * flag).sum())
    fn = float((w * y * ~flag).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "flag_rate": float((w * flag).sum() / w.sum())}


def best_f1_threshold(y, s, w=None) -> float:
    """Threshold (one of the observed scores) that maximises F1 for `risk >= t`.
    Pick it on dev and apply it unchanged to test. Ties go to the highest threshold."""
    y, s, w = _prep(y, s, w)
    uniq, pos, neg = _groups_desc(y, s, w)
    P = pos.sum()
    if P == 0:
        return float("inf")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    f1 = 2 * tp / (tp + fp + P)  # 2TP / (2TP + FP + FN), FN = P - TP
    return float(uniq[int(np.argmax(f1))])


# ---------------------------------------------------------------- bootstrap
def bootstrap_weights(groups: Sequence[str], n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_boot, n_rows) row weights for a cluster bootstrap: each replicate draws as many
    documents as there are, with replacement, and every row inherits its document's count.
    Reuse one matrix for several systems scored on the same rows to get *paired* CIs."""
    _, g = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    n_groups = int(g.max()) + 1 if len(g) else 0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    counts = np.stack([np.bincount(d, minlength=n_groups) for d in draws])
    return counts[:, g].astype(np.float64)


def percentile_ci(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[~np.isnan(v)]
    if not len(v):
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(metric: Callable[..., float], y, s, groups=None, *, weights: np.ndarray | None = None,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of `metric(y, s, w=...)` under a document-level bootstrap.
    Pass `groups` (one key per row, e.g. group_keys(rows)) or a precomputed `weights` matrix."""
    if weights is None:
        weights = bootstrap_weights(groups, n_boot, seed)
    return percentile_ci((metric(y, s, w=w) for w in weights), alpha)


# ---------------------------------------------------------------- one-call summary
def summarize(rows: Sequence[dict], *, threshold: float | None = None, frac: float = 0.10,
              n_boot: int = 0, seed: int = 0, weights: np.ndarray | None = None) -> dict:
    """Point metrics (and, with n_boot > 0 or a weights matrix, 95% document-bootstrap CIs).

    `threshold` should come from best_f1_threshold() on the *dev* split; F1 is then reported
    at that fixed threshold. Without it, F1 is omitted (choosing it on the same rows would
    be optimistic)."""
    y, s = to_arrays(rows)
    fns: dict[str, Callable[..., float]] = {
        "auroc": auroc,
        "pr_auc": average_precision,
        f"recall@{frac:.0%}": lambda y, s, w=None: recall_at(y, s, frac, w),
    }
    if threshold is not None:
        fns["f1"] = lambda y, s, w=None: prf_at(y, s, threshold, w)["f1"]
    out: dict = {"n": len(rows), "n_docs": len(set(group_keys(rows))), "major_rate": float(y.mean()) if len(y) else math.nan}
    out.update({k: f(y, s) for k, f in fns.items()})
    if threshold is not None:
        out["threshold"] = threshold
        out.update({k: v for k, v in prf_at(y, s, threshold).items() if k != "f1"})
    if weights is None and n_boot > 0:
        weights = bootstrap_weights(group_keys(rows), n_boot, seed)
    if weights is not None:
        out["ci"] = {k: bootstrap_ci(f, y, s, weights=weights) for k, f in fns.items()}
    return out
