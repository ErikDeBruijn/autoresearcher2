"""Tests for the _apply_code_changes helper extracted from executors."""
import base64
import pytest

from autoresearcher2.v3.executors import _apply_code_changes


def test_apply_diff(tmp_path):
    """_apply_code_changes applies a unified diff via patch."""
    calls = []

    def mock_run_cmd(cmd):
        calls.append(cmd)
        return 0, "patched ok"

    spec = {"diff": "--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-old\n+new\n"}
    _apply_code_changes(mock_run_cmd, spec, str(tmp_path))

    # Should have made two calls: write the diff file, then patch
    assert len(calls) == 2
    assert "/tmp/_ar_patch.diff" in calls[0]
    assert "patch -p1" in calls[1]
    assert str(tmp_path) in calls[1]


def test_apply_file_changes(tmp_path):
    """_apply_code_changes writes file_changes via base64 one-liner."""
    calls = []

    def mock_run_cmd(cmd):
        calls.append(cmd)
        return 0, "ok"

    spec = {"file_changes": {"train.py": "print('hello')\n", "config.py": "LR=0.01\n"}}
    _apply_code_changes(mock_run_cmd, spec, str(tmp_path))

    assert len(calls) == 2
    # Both calls should write files in work_dir
    for call in calls:
        assert str(tmp_path) in call
        assert "base64" in call


def test_raises_without_diff_or_file_changes():
    """_apply_code_changes raises ValueError when neither diff nor file_changes."""
    def mock_run_cmd(cmd):
        return 0, ""

    with pytest.raises(ValueError, match="diff.*file_changes"):
        _apply_code_changes(mock_run_cmd, {"some_param": "value"}, "/tmp/work")


def test_apply_diff_patch_failure():
    """_apply_code_changes raises RuntimeError when patch fails."""
    call_count = [0]

    def mock_run_cmd(cmd):
        call_count[0] += 1
        if call_count[0] == 1:
            return 0, "ok"  # write diff file
        return 1, "patch: FAILED"  # patch command fails

    spec = {"diff": "bad diff"}
    with pytest.raises(RuntimeError, match="patch failed"):
        _apply_code_changes(mock_run_cmd, spec, "/tmp/work")


def test_apply_file_changes_write_failure():
    """_apply_code_changes raises RuntimeError when file write fails."""
    def mock_run_cmd(cmd):
        return 1, "permission denied"

    spec = {"file_changes": {"train.py": "content"}}
    with pytest.raises(RuntimeError, match="Failed to write"):
        _apply_code_changes(mock_run_cmd, spec, "/tmp/work")


def test_sanitizes_filenames():
    """_apply_code_changes sanitizes path traversal in filenames."""
    calls = []

    def mock_run_cmd(cmd):
        calls.append(cmd)
        return 0, "ok"

    spec = {"file_changes": {"../../../etc/passwd": "harmless"}}
    _apply_code_changes(mock_run_cmd, spec, "/tmp/work")

    assert len(calls) == 1
    # Path traversal should be sanitized: no ".." in the target path
    assert "______etc_passwd" in calls[0]
    assert "../" not in calls[0]
