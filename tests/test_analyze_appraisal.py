"""Tests for appraisal analysis functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_appraisal import (
    successful_results,
    convergence_trace,
    experiments_to_best,
    waste_ratio,
    cell_coverage,
    information_gain_analysis,
)


def _make_result(cell, val_bpb=None, error=None, scores=None, source="test"):
    r = {"cell": cell, "val_bpb": val_bpb, "error": error, "source": source}
    if scores:
        r["scores"] = scores
    return r


def test_successful_results_filters_errors():
    results = [
        _make_result(0, val_bpb=1.05),
        _make_result(1, val_bpb=None, error="OOM"),
        _make_result(2, val_bpb=1.04),
    ]
    assert len(successful_results(results)) == 2


def test_convergence_trace():
    results = [
        _make_result(0, val_bpb=1.06),
        _make_result(1, error="fail"),
        _make_result(2, val_bpb=1.04),
        _make_result(3, val_bpb=1.05),  # worse than 1.04
        _make_result(4, val_bpb=1.03),
    ]
    trace = convergence_trace(results)
    assert trace == [1.06, 1.04, 1.04, 1.03]


def test_experiments_to_best():
    results = [
        _make_result(0, val_bpb=1.06),
        _make_result(1, error="fail"),
        _make_result(2, val_bpb=1.04),
    ]
    assert experiments_to_best(results, 1.05) == 3  # 3rd experiment (index 2) reaches it
    assert experiments_to_best(results, 1.03) is None


def test_waste_ratio():
    results = [
        _make_result(0, val_bpb=1.06),  # first = new best, not waste
        _make_result(1, error="fail"),   # waste
        _make_result(2, val_bpb=1.07),  # worse = waste
        _make_result(3, val_bpb=1.04),  # new best, not waste
    ]
    assert waste_ratio(results) == 0.5


def test_cell_coverage():
    results = [_make_result(0), _make_result(1), _make_result(0), _make_result(5)]
    cov = cell_coverage(results, n_cells=10)
    assert cov["unique_cells"] == 3
    assert cov["coverage_pct"] == 30.0


def test_information_gain_analysis():
    results = [
        _make_result(0, val_bpb=1.06, scores={"epistemic": 6.0, "pragmatic": -1.0}),
        _make_result(1, val_bpb=1.04, scores={"epistemic": 4.5, "pragmatic": -0.3}),
        _make_result(2, val_bpb=1.03, scores={"epistemic": 3.0, "pragmatic": -0.1}),
    ]
    pairs = information_gain_analysis(results)
    assert len(pairs) == 2
    assert pairs[0]["from_cell"] == 0
    assert pairs[0]["to_cell"] == 1
    assert pairs[0]["val_bpb_improvement"] > 0  # 1.06 → 1.04 = improvement
