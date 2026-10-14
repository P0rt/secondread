"""Tests for the flag operating points: metrics.max_threshold_at_recall, the thresholds file
(scripts/pick_thresholds.py) and how the backend picks a threshold (secondread.agent).

Runs under pytest, or without it (pytest is not a project dependency yet):
    HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python tests/test_thresholds.py
Needs no server. test_review_mode_reads_no_test_file needs the dev score files in results/.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from secondread import agent as A  # noqa: E402
from secondread import metrics as M  # noqa: E402

V1 = A.ScoringModel("v1", "LoRA v1", True, "v1")


def _cal(data: dict) -> A.Calibration:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
    return A.Calibration(Path(f.name))


NEW = {"default_operating_point": "review", "fallback": 0.5,
       "operating_points": {"review": {"target_recall": 0.6}, "f1": {}},
       "adapters": {"v1": {"sha256": "abc", "review": {"thresholds": {"en-ru": 0.2, "en-es": 0.45}},
                           "f1": {"thresholds": {"en-ru": 0.16, "en-es": 0.70}}},
                    "base": {"sha256": None, "f1": {"thresholds": {"en-ru": 0.37}}}}}
LEGACY = {"fallback": 0.5, "method": "F1-optimal ...",
          "adapters": {"v1": {"sha256": "abc", "thresholds": {"en-ru": 0.16, "en-es": 0.70}}}}


def test_max_threshold_at_recall():
    y, s = [1, 0, 1, 1, 0], [0.9, 0.8, 0.7, 0.3, 0.1]
    assert M.max_threshold_at_recall(y, s, 0.6) == 0.7      # needs 2 of 3 Majors
    assert M.max_threshold_at_recall(y, s, 0.34) == 0.7     # 1 of 3 is 0.333 < 0.34
    assert M.max_threshold_at_recall(y, s, 1 / 3) == 0.9    # exactly 1 of 3 qualifies
    assert M.max_threshold_at_recall(y, s, 1.0) == 0.3
    assert M.max_threshold_at_recall([1, 0, 1], [0.9, 0.9, 0.5], 0.5) == 0.9  # a tie group is flagged whole
    assert math.isinf(M.max_threshold_at_recall([0, 0], [0.2, 0.1], 0.6))
    for bad in (0.0, 1.5):
        try:
            M.max_threshold_at_recall(y, s, bad)
        except ValueError:
            continue
        raise AssertionError(f"target {bad} accepted")


def test_exact_target_qualifies():
    # 95 Majors, as en-es dev: 0.6 * 95 = 57 must count as reaching 0.60
    y = [1] * 95 + [0] * 5
    s = [1.0 - i / 100 for i in range(100)]
    t = M.max_threshold_at_recall(y, s, 0.6)
    assert M.prf_at(y, s, t)["recall"] >= 0.6 and abs(t - s[56]) < 1e-12


def test_highest_threshold_is_maximal():
    import numpy as np
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500) * 0.5 + 0.4 * y, 3)  # coarse scores, many ties
    for target in (0.3, 0.6, 0.9):
        t = M.max_threshold_at_recall(y, s, target)
        assert M.prf_at(y, s, t)["recall"] >= target
        higher = np.unique(s)[np.unique(s) > t]
        assert not len(higher) or M.prf_at(y, s, higher.min())["recall"] < target


def test_calibration_points():
    c = _cal(NEW)
    assert c.default_point == "review" and c.default_from_file
    assert c.threshold("v1", "en-es", "review") == 0.45 and c.threshold("v1", "en-es", "f1") == 0.70
    assert c.threshold("base", "en-ru", "review") is None and c.threshold("base", "en-ru", "f1") == 0.37
    assert c.targets["review"] == 0.6 and c.pairs == ["en-es", "en-ru"]
    old = _cal(LEGACY)  # the one-point format is the F1 point; with no review block, F1 is the default
    assert old.threshold("v1", "en-ru", "f1") == 0.16 and old.threshold("v1", "en-ru", "review") is None
    assert old.default_point == "f1" and not old.default_from_file


def test_agent_thresholds():
    ag = A.ReviewAgent(calibration=_cal(NEW), operating_point="review")
    assert ag.thresholds_for(V1, ["en-ru", "en-es"]) == ({"en-es": 0.45, "en-ru": 0.2}, "dev_review")
    assert ag.thresholds_for(V1, ["en-ru"], point="f1") == ({"en-ru": 0.16}, "dev_f1")
    assert ag.thresholds_for(V1, ["en-ru"], 0.3) == ({"en-ru": 0.3}, "request")
    assert ag.thresholds_for(V1, ["en-ru", "en-de"]) == ({"en-de": 0.5, "en-ru": 0.2}, "dev_review+fallback")
    base = A.ScoringModel("base", "base", False, None)
    assert ag.thresholds_for(base, ["en-ru"]) == ({"en-ru": 0.5}, "fallback")  # no review block for base
    assert A.ReviewAgent(calibration=_cal(NEW), operating_point="f1").thresholds_for(V1, ["en-es"])[0] == {"en-es": 0.70}
    for bad in ("recall", "F1 "):
        try:
            A.ReviewAgent(calibration=_cal(NEW), operating_point=bad)
        except A.ReviewError as e:
            assert e.code == "bad_operating_point"
            continue
        raise AssertionError(f"operating point {bad!r} accepted")


def test_shipped_file():
    data = json.loads((ROOT / "secondread/thresholds.json").read_text())
    c = A.Calibration(ROOT / "secondread/thresholds.json")
    assert data["default_operating_point"] == "review" == c.default_point
    target = data["operating_points"]["review"]["target_recall"]
    v1 = data["adapters"]["v1"]
    assert set(v1["review"]["thresholds"]) == set(v1["f1"]["thresholds"]) == {"en-ru", "en-es"}
    for lp, thr in v1["review"]["thresholds"].items():
        st = v1["review"]["stats"][lp]
        assert "test" not in st, "the review point must not carry test numbers"
        assert st["dev"]["recall"] >= target and thr <= st["exact_threshold"]
        assert c.threshold("v1", lp, "review") == thr


def test_review_mode_reads_no_test_file():
    import pick_thresholds as P
    opened = []
    real = M.load_scored

    def spy(path):
        opened.append(str(path))
        assert ".test." not in Path(path).name and "_test." not in Path(path).name, f"opened {path}"
        return real(path)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t.json"
        out.write_text((ROOT / "secondread/thresholds.json").read_text())
        argv, P.M.load_scored = sys.argv, spy
        try:
            sys.argv = ["pick_thresholds.py", "--target-recall", "0.6", "--n-boot", "20", "--adapters", "v1",
                        "--out", str(out)]
            P.main()
        finally:
            sys.argv, P.M.load_scored = argv, real
        got = json.loads(out.read_text())
        shipped = json.loads((ROOT / "secondread/thresholds.json").read_text())
    assert any("dev" in p for p in opened)
    # the threshold does not depend on the bootstrap; the f1 block is kept as it was
    assert got["adapters"]["v1"]["review"]["thresholds"] == shipped["adapters"]["v1"]["review"]["thresholds"]
    assert got["adapters"]["v1"]["f1"] == shipped["adapters"]["v1"]["f1"]


if __name__ == "__main__":
    import traceback
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"ok    {n}")
        except Exception:
            failed += 1
            print(f"FAIL  {n}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
