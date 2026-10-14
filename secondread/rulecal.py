"""Calibrated weights of the rule checks in the app's combined risk.

The rule checks (`secondread.checks`) find cheap, explainable evidence: a number or a negation
that is in the source but not in the MT, or the other way round. Their hand-set confidences
(0.25-0.75) were combined with the model risk by noisy-OR, and that ranked worse than the model
alone, because a rule hit is a Major unit far less often than those values say (RESULTS.md,
"Rule calibration").

This module combines them with weights measured on the **train** splits only
(`scripts/calibrate_rules.py` writes `secondread/rule_calibration.json`):

    logit(risk) = logit(model risk) + sum over checks of the largest weight among its findings

* A weight is the log-odds that a finding adds to the model risk. Weights are never negative,
  so a finding can raise the risk but never lower it, and a segment without findings keeps
  exactly the model risk.
* Each check contributes only its strongest finding (as before), so three "extra number" hits
  in one segment count once.
* Without a model score (llama-server down: rule checks only) the risk is the largest
  train-measured P(Major | finding) of the segment, and 0 without findings.
* A pair with no entry of its own uses the "*" entry (fitted on all pairs pooled). A code with
  no entry, or no readable calibration file at all, gets weight 0: the finding is still shown
  as a reason, it just does not move the ranking.

Findings are always returned as reasons, whatever their weight: the explanation is part of the
product even when the evidence is weak.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from secondread import checks

log = logging.getLogger("secondread.rulecal")

RULE_CALIBRATION_PATH = Path(os.getenv("SECONDREAD_RULE_CALIBRATION",
                                       str(Path(__file__).with_name("rule_calibration.json"))))
_EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def strongest(results: Iterable[checks.CheckResult], value) -> list[float]:
    """One value per check that found something: the largest `value(issue)` among its findings."""
    return [max(value(i) for i in c.issues) for c in results if c.issues]


def combine_logodds(model_risk: float, results: Iterable[checks.CheckResult], weight) -> float:
    """logit(risk) = logit(model risk) + the strongest weight of each check; `weight(issue)` >= 0."""
    bump = sum(strongest(results, weight))
    # max(): the logit clip at 1e-6 must not pull a risk of exactly 1.0 down
    return model_risk if bump == 0 else max(model_risk, sigmoid(logit(model_risk) + bump))


def combine_noisy_or(model_risk: float, results: Iterable[checks.CheckResult], conf) -> float:
    """risk = 1 - (1 - model) * prod over checks of (1 - strongest `conf(issue)`): the form the app
    used before, kept so scripts/calibrate_rules.py can compare it on dev."""
    return 1.0 - (1.0 - model_risk) * math.prod(1.0 - p for p in strongest(results, conf))


class RuleCalibration:
    """Per language pair and issue code: weight (log-odds added to the model) and the train
    P(Major | finding). Read from secondread/rule_calibration.json."""

    def __init__(self, path: Path | str = RULE_CALIBRATION_PATH, data: Mapping[str, Any] | None = None):
        if data is None:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("no readable rule calibration file: rule findings get weight 0 (reasons only)")
                data = {}
        self.method = data.get("method_id", "none")
        self.fitted_as = data.get("fitted_as")
        self.generated = data.get("generated")
        self.pairs: dict[str, dict[str, dict[str, Any]]] = {
            lp: dict(e.get("codes") or {}) for lp, e in (data.get("pairs") or {}).items()}

    def _entry(self, lp: str, code: str) -> Mapping[str, Any] | None:
        e = self.pairs.get(lp, {}).get(code)
        return e if e is not None else self.pairs.get("*", {}).get(code)

    def weight(self, lp: str, code: str) -> float:
        e = self._entry(lp, code)
        return max(0.0, float(e.get("weight", 0.0))) if e else 0.0

    def p_major(self, lp: str, code: str) -> float | None:
        e = self._entry(lp, code)
        return float(e["p_major"]) if e and e.get("p_major") is not None else None

    def describe(self) -> dict[str, Any]:
        """Text-free summary for /api/health."""
        return {"method": self.method, "fitted_as": self.fitted_as, "generated": self.generated,
                "pairs": sorted(lp for lp in self.pairs if lp != "*")}

    def risk(self, model_risk: float | None, results: list[checks.CheckResult], lp: str) -> float:
        """The combined risk of one segment (see the module docstring)."""
        if model_risk is None:  # rule checks only; the hand-set prior if the code was never measured
            def conf(i: checks.Issue) -> float:
                p = self.p_major(lp, i.code)
                return p if p is not None else i.confidence
            return max(strongest(results, conf), default=0.0)
        return combine_logodds(model_risk, results, lambda i: self.weight(lp, i.code))
