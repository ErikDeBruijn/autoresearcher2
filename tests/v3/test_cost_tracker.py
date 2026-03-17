"""Tests for the cost_tracker module."""
import json
from unittest.mock import patch, MagicMock
import pytest

from autoresearcher2.v3.cost_tracker import (
    _start_cost_job,
    _stop_cost_job,
    with_cost_tracking,
)


def _mock_urlopen(response_data):
    """Create a mock urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_with_cost_tracking_returns_original_when_cuda_device_none():
    """with_cost_tracking returns the original executor when cuda_device is None."""
    def inner(proposal):
        return {"metrics": {}}

    result = with_cost_tracking(inner, cuda_device=None)
    assert result is inner


def test_with_cost_tracking_returns_original_when_cuda_device_non_numeric():
    """with_cost_tracking returns the original executor when cuda_device is non-numeric."""
    def inner(proposal):
        return {"metrics": {}}

    result = with_cost_tracking(inner, cuda_device="cpu")
    assert result is inner


@patch("autoresearcher2.v3.cost_tracker._stop_cost_job")
@patch("autoresearcher2.v3.cost_tracker._start_cost_job")
def test_with_cost_tracking_wraps_when_cuda_device_is_digit(mock_start, mock_stop):
    """with_cost_tracking wraps the executor when cuda_device is '0'."""
    mock_start.return_value = "job_1"
    mock_stop.return_value = {"energy_kwh": 0.1, "cost_eur": 0.02, "avg_power_w": 200}

    def inner(proposal):
        return {"metrics": {"score": 1.0}}

    wrapped = with_cost_tracking(inner, cuda_device="0")
    assert wrapped is not inner

    # Verify it actually calls start/stop when invoked
    from autoresearcher2.v3.proposal import Proposal
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"x": "1"},
    )
    result = wrapped(p)
    assert result["energy_kwh"] == 0.1
    mock_start.assert_called_once_with(gpu=0, label=p.id)
    mock_stop.assert_called_once_with("job_1")


@patch("autoresearcher2.v3.cost_tracker.urllib.request.urlopen")
def test_start_cost_job_returns_none_on_connection_failure(mock_urlopen):
    """_start_cost_job returns None when the HTTP request fails."""
    mock_urlopen.side_effect = ConnectionRefusedError("connection refused")
    result = _start_cost_job(gpu=0, label="test")
    assert result is None


@patch("autoresearcher2.v3.cost_tracker.urllib.request.urlopen")
def test_stop_cost_job_returns_none_on_connection_failure(mock_urlopen):
    """_stop_cost_job returns None when the HTTP request fails."""
    mock_urlopen.side_effect = ConnectionRefusedError("connection refused")
    result = _stop_cost_job("abc123")
    assert result is None
