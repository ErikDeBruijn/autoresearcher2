"""Tests for executors (without real SSH)."""
import pytest
from autoresearcher2.v3.executors import make_dry_run_executor
from autoresearcher2.v3.proposal import Proposal


def test_dry_run_executor():
    execute = make_dry_run_executor()
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.04"},
    )
    result = execute(p)
    assert result["metrics"]["dry_run"] is True
    assert "raw_log" in result


def test_dry_run_with_various_types():
    execute = make_dry_run_executor()
    for itype in ("config_change", "probe", "code_change", "replication"):
        p = Proposal(
            intent="test", rationale="test", expected_learning="test",
            intervention_type=itype,
            intervention_spec={"x": "1"},
        )
        result = execute(p)
        assert result is not None
