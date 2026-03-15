"""Tests for code_change intervention type in the executor.

code_change lets the LLM propose structural changes to training scripts,
not just parameter tweaks. The executor writes file contents from
intervention_spec["file_changes"] into the per-GPU working directory.
"""
import os
import stat
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.executors import make_trainpy_executor, make_dry_run_executor


@pytest.fixture
def train_env(tmp_path):
    """Set up a fake training environment with base and per-GPU dirs."""
    base = tmp_path / "autoresearch"
    base.mkdir()
    # Minimal train.py that prints a val_bpb
    (base / "train.py").write_text(
        'print("val_bpb: 1.100")\nprint("num_steps: 100")\n'
    )
    gpu_dir = tmp_path / "autoresearch_gpu0"
    gpu_dir.mkdir()
    (gpu_dir / "train.py").write_text(
        'print("val_bpb: 1.100")\nprint("num_steps: 100")\n'
    )
    return {"base": str(base), "gpu_dir": str(gpu_dir), "tmp": tmp_path}


def test_code_change_writes_files(train_env):
    """code_change writes file_changes to the per-GPU train_dir."""
    new_content = 'import math\nprint("val_bpb: 1.050")\nprint("num_steps: 200")\n'
    proposal = Proposal(
        intent="test code change",
        rationale="structural change",
        expected_learning="whether code changes work",
        intervention_type="code_change",
        intervention_spec={"file_changes": {"train.py": new_content}},
    )

    # Create executor with local=True so it uses bash directly
    with patch("autoresearcher2.v3.executors._start_cost_job", return_value=None), \
         patch("autoresearcher2.v3.executors._stop_cost_job", return_value=None):
        executor = make_trainpy_executor(
            remote_dir=str(train_env["base"]),
            cuda_device="0",
            local=True,
        )
        result = executor(proposal)

    # The file should have been written with new content
    gpu_train = os.path.join(str(train_env["base"]) + "_gpu0", "train.py")
    written = open(gpu_train).read()
    assert "import math" in written
    assert "1.050" in written
    assert result["metrics"]["val_bpb"] == 1.05


def test_code_change_no_file_changes_raises():
    """code_change without file_changes key raises ValueError."""
    proposal = Proposal(
        intent="bad code change",
        rationale="missing file_changes",
        expected_learning="error handling",
        intervention_type="code_change",
        intervention_spec={"some_param": "value"},
    )

    with patch("autoresearcher2.v3.executors._start_cost_job", return_value=None), \
         patch("autoresearcher2.v3.executors._stop_cost_job", return_value=None):
        executor = make_trainpy_executor(
            remote_dir="/tmp/test_autoresearch",
            cuda_device="0",
            local=True,
        )
        with pytest.raises(ValueError, match="file_changes"):
            executor(proposal)


def test_code_change_sanitizes_filename(train_env):
    """Filenames with path traversal are sanitized."""
    proposal = Proposal(
        intent="test path traversal",
        rationale="security",
        expected_learning="filenames are safe",
        intervention_type="code_change",
        intervention_spec={
            "file_changes": {"../../../etc/passwd": "harmless content"}
        },
    )

    with patch("autoresearcher2.v3.executors._start_cost_job", return_value=None), \
         patch("autoresearcher2.v3.executors._stop_cost_job", return_value=None):
        executor = make_trainpy_executor(
            remote_dir=str(train_env["base"]),
            cuda_device="0",
            local=True,
        )
        # Should not crash — filename gets sanitized to something safe
        # But training will fail since train.py wasn't modified
        try:
            executor(proposal)
        except RuntimeError:
            pass  # Expected: training fails, but no path traversal

    # Verify no file was written outside train_dir
    assert not os.path.exists("/tmp/test_etc_passwd")

    # The sanitized file should be in the gpu dir
    gpu_dir = str(train_env["base"]) + "_gpu0"
    # ".." becomes "__", "/" becomes "_"
    sanitized = "______etc_passwd"
    assert os.path.exists(os.path.join(gpu_dir, sanitized))


def test_code_change_multiple_files(train_env):
    """code_change can write multiple files at once."""
    proposal = Proposal(
        intent="multi-file change",
        rationale="test",
        expected_learning="multi-file works",
        intervention_type="code_change",
        intervention_spec={
            "file_changes": {
                "train.py": 'print("val_bpb: 1.020")\n',
                "config.py": 'LR = 0.04\nDEPTH = 8\n',
            }
        },
    )

    with patch("autoresearcher2.v3.executors._start_cost_job", return_value=None), \
         patch("autoresearcher2.v3.executors._stop_cost_job", return_value=None):
        executor = make_trainpy_executor(
            remote_dir=str(train_env["base"]),
            cuda_device="0",
            local=True,
        )
        result = executor(proposal)

    gpu_dir = str(train_env["base"]) + "_gpu0"
    assert os.path.exists(os.path.join(gpu_dir, "config.py"))
    config_content = open(os.path.join(gpu_dir, "config.py")).read()
    assert "LR = 0.04" in config_content
    assert result["metrics"]["val_bpb"] == 1.02


def test_config_change_still_works(train_env):
    """config_change intervention type still works after code_change addition."""
    proposal = Proposal(
        intent="test config change",
        rationale="regression test",
        expected_learning="old behavior preserved",
        intervention_type="config_change",
        intervention_spec={"DEPTH": "8"},
    )

    with patch("autoresearcher2.v3.executors._start_cost_job", return_value=None), \
         patch("autoresearcher2.v3.executors._stop_cost_job", return_value=None):
        executor = make_trainpy_executor(
            remote_dir=str(train_env["base"]),
            cuda_device="0",
            local=True,
        )
        result = executor(proposal)

    # Should still parse val_bpb from the original train.py output
    assert result["metrics"]["val_bpb"] == 1.1


def test_unsupported_type_still_dry_runs(train_env):
    """Unknown intervention types still return dry-run result."""
    proposal = Proposal(
        intent="unknown type",
        rationale="test",
        expected_learning="test",
        intervention_type="quantum_experiment",
        intervention_spec={"qubit": "1"},
    )

    with patch("autoresearcher2.v3.executors._start_cost_job", return_value=None), \
         patch("autoresearcher2.v3.executors._stop_cost_job", return_value=None):
        executor = make_trainpy_executor(
            remote_dir=str(train_env["base"]),
            cuda_device="0",
            local=True,
        )
        result = executor(proposal)

    assert result["metrics"]["unsupported"] is True


def test_code_change_with_single_quotes(train_env):
    """code_change handles file content containing single quotes."""
    content_with_quotes = "name = 'hello world'\nprint(f'val_bpb: 1.080')\n"
    proposal = Proposal(
        intent="test quotes",
        rationale="edge case",
        expected_learning="quoting works",
        intervention_type="code_change",
        intervention_spec={"file_changes": {"train.py": content_with_quotes}},
    )

    with patch("autoresearcher2.v3.executors._start_cost_job", return_value=None), \
         patch("autoresearcher2.v3.executors._stop_cost_job", return_value=None):
        executor = make_trainpy_executor(
            remote_dir=str(train_env["base"]),
            cuda_device="0",
            local=True,
        )
        result = executor(proposal)

    gpu_dir = str(train_env["base"]) + "_gpu0"
    written = open(os.path.join(gpu_dir, "train.py")).read()
    assert "name = 'hello world'" in written
    assert result["metrics"]["val_bpb"] == 1.08
