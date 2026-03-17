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


def test_code_change_does_not_inject_signal_alarm(tmp_path):
    """Written .py files must not have signal.alarm injected by the executor.

    The subprocess.run timeout= parameter is the correct timeout mechanism.
    Injecting signal.alarm into user scripts is fragile and Python-only.
    """
    py_content = (
        "import numpy as np\n"
        "import os\n"
        "\n"
        "result = np.array([1, 2, 3])\n"
        "print(f'score: {result.sum()}')\n"
    )
    work_dir = str(tmp_path / "workdir")
    import os
    os.makedirs(work_dir)

    execute = make_shell_executor(
        command_template=f"cd {work_dir} && python3 train.py",
        metric_patterns={"score": r"score:\s+([\d.]+)"},
        work_dir=work_dir,
        timeout=300,
    )
    p = Proposal(
        intent="test no signal injection",
        rationale="signal.alarm is fragile",
        expected_learning="files are written without signal.alarm",
        intervention_type="code_change",
        intervention_spec={"file_changes": {"train.py": py_content}},
    )

    # We expect the command itself may fail (numpy not installed, etc.)
    # but we only care about what was written to the file
    try:
        execute(p)
    except (RuntimeError, Exception):
        pass

    written = open(os.path.join(work_dir, "train.py")).read()
    assert "signal.alarm" not in written, (
        "Executor should not inject signal.alarm into user scripts; "
        "subprocess.run timeout= is the correct timeout mechanism"
    )
