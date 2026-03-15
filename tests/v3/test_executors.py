"""Tests for executors (without real SSH)."""
import pytest
from autoresearcher2.v3.executors import make_dry_run_executor, make_shell_executor
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


def test_shell_executor_runs_local_command():
    execute = make_shell_executor(
        command_template="echo 'score: 42.5'",
        metric_patterns={"score": r"score:\s+([\d.]+)"},
    )
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"lr": "0.01"},
    )
    result = execute(p)
    assert result["metrics"]["score"] == 42.5
    assert result["compute_cost"] > 0
    assert "score: 42.5" in result["raw_log"]


def test_shell_executor_passes_spec_as_env():
    execute = make_shell_executor(
        command_template="printenv DEPTH",
        metric_patterns={},
    )
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="probe",
        intervention_spec={"DEPTH": "8"},
    )
    result = execute(p)
    assert "8" in result["raw_log"]


def test_shell_executor_raises_on_failure():
    execute = make_shell_executor(command_template="exit 1")
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={},
    )
    with pytest.raises(RuntimeError, match="Command failed"):
        execute(p)
