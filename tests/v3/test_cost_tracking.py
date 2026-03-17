"""Tests for cost tracking on observations."""
import json
from unittest.mock import patch, MagicMock
import pytest
from autoresearcher2.v3.observation import Observation
from autoresearcher2.v3.cost_tracker import _start_cost_job, _stop_cost_job


def test_observation_with_cost_fields(store):
    """Observations can store energy and cost data from gpu-cost-tracker."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"MATRIX_LR": "0.08"},
        outcome_success=True,
        outcome_metrics={"val_bpb": 1.05},
        wall_time_s=300.0,
        energy_kwh=0.12,
        cost_eur=0.03,
        avg_power_w=280.0,
    )
    store.save_observation(obs)
    loaded = store.load_observation(obs.id)
    assert loaded.energy_kwh == 0.12
    assert loaded.cost_eur == 0.03
    assert loaded.avg_power_w == 280.0


def test_observation_without_cost_fields(store):
    """Backwards compat: observations without cost fields still work."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        wall_time_s=60.0,
    )
    store.save_observation(obs)
    loaded = store.load_observation(obs.id)
    assert loaded.energy_kwh is None
    assert loaded.cost_eur is None
    assert loaded.avg_power_w is None


def test_cost_in_to_dict():
    """Cost fields appear in serialized dict."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        energy_kwh=0.5,
        cost_eur=0.10,
        avg_power_w=350.0,
    )
    d = obs.to_dict()
    assert d["energy_kwh"] == 0.5
    assert d["cost_eur"] == 0.10
    assert d["avg_power_w"] == 350.0


def test_cost_from_dict():
    """Cost fields survive serialization roundtrip."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        energy_kwh=0.5,
        cost_eur=0.10,
        avg_power_w=350.0,
    )
    d = obs.to_dict()
    loaded = Observation.from_dict(d)
    assert loaded.energy_kwh == 0.5
    assert loaded.cost_eur == 0.10
    assert loaded.avg_power_w == 350.0


def _mock_urlopen(response_data):
    """Create a mock urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


@patch("autoresearcher2.v3.cost_tracker.urllib.request.urlopen")
def test_start_cost_job(mock_urlopen):
    """_start_cost_job calls the API and returns job_id."""
    mock_urlopen.return_value = _mock_urlopen({"job_id": "abc123"})
    result = _start_cost_job(gpu=0, label="test_proposal")
    assert result == "abc123"
    mock_urlopen.assert_called_once()


@patch("autoresearcher2.v3.cost_tracker.urllib.request.urlopen")
def test_stop_cost_job(mock_urlopen):
    """_stop_cost_job returns cost data from the API."""
    cost_data = {"energy_kwh": 0.15, "cost_eur": 0.04, "duration_s": 300, "avg_power_w": 290}
    mock_urlopen.return_value = _mock_urlopen(cost_data)
    result = _stop_cost_job("abc123")
    assert result["energy_kwh"] == 0.15
    assert result["cost_eur"] == 0.04
    assert result["avg_power_w"] == 290


@patch("autoresearcher2.v3.cost_tracker.urllib.request.urlopen")
def test_start_cost_job_failure_returns_none(mock_urlopen):
    """Cost tracker failure is non-fatal — returns None."""
    mock_urlopen.side_effect = Exception("connection refused")
    result = _start_cost_job(gpu=0, label="test")
    assert result is None


def test_worker_passes_cost_to_observation(store):
    """Worker passes cost fields from executor result to observation."""
    from autoresearcher2.v3.worker import Worker
    from autoresearcher2.v3.proposal import Proposal

    # Create a proposal in the store
    p = Proposal(
        intent="test cost passthrough",
        rationale="test",
        expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        status="todo",
    )
    store.save_proposal(p)

    def fake_executor(proposal):
        return {
            "metrics": {"val_bpb": 1.0},
            "energy_kwh": 0.25,
            "cost_eur": 0.06,
            "avg_power_w": 300.0,
        }

    worker = Worker(store, execute_fn=fake_executor, worker_id="test_worker")
    result = worker.tick()

    assert result is not None
    assert result["energy_kwh"] == 0.25
    assert result["cost_eur"] == 0.06
    assert result["avg_power_w"] == 300.0

    # Verify it's persisted in the store
    obs = store.load_observation(result["id"])
    assert obs.energy_kwh == 0.25
    assert obs.cost_eur == 0.06


@patch("autoresearcher2.v3.cost_tracker._stop_cost_job")
@patch("autoresearcher2.v3.cost_tracker._start_cost_job")
def test_with_cost_tracking_wraps_executor(mock_start, mock_stop):
    """with_cost_tracking wraps an executor and merges cost fields into result."""
    from autoresearcher2.v3.cost_tracker import with_cost_tracking
    from autoresearcher2.v3.proposal import Proposal

    mock_start.return_value = "job_42"
    mock_stop.return_value = {
        "energy_kwh": 0.15,
        "cost_eur": 0.04,
        "avg_power_w": 290.0,
    }

    def inner_executor(proposal):
        return {"metrics": {"score": 1.0}, "raw_log": "ok"}

    wrapped = with_cost_tracking(inner_executor, cuda_device="1")
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"x": "1"},
    )
    result = wrapped(p)

    # Inner result preserved
    assert result["metrics"]["score"] == 1.0
    assert result["raw_log"] == "ok"

    # Cost fields merged in
    assert result["energy_kwh"] == 0.15
    assert result["cost_eur"] == 0.04
    assert result["avg_power_w"] == 290.0

    mock_start.assert_called_once_with(gpu=1, label=p.id)
    mock_stop.assert_called_once_with("job_42")


@patch("autoresearcher2.v3.cost_tracker._stop_cost_job")
@patch("autoresearcher2.v3.cost_tracker._start_cost_job")
def test_with_cost_tracking_handles_tracker_failure(mock_start, mock_stop):
    """with_cost_tracking still returns executor result when tracker fails."""
    from autoresearcher2.v3.cost_tracker import with_cost_tracking
    from autoresearcher2.v3.proposal import Proposal

    mock_start.return_value = None  # tracker unavailable

    def inner_executor(proposal):
        return {"metrics": {"score": 2.0}}

    wrapped = with_cost_tracking(inner_executor, cuda_device="1")
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"x": "1"},
    )
    result = wrapped(p)

    assert result["metrics"]["score"] == 2.0
    assert "energy_kwh" not in result
    mock_stop.assert_not_called()


@patch("autoresearcher2.v3.cost_tracker._stop_cost_job")
@patch("autoresearcher2.v3.cost_tracker._start_cost_job")
def test_with_cost_tracking_no_cuda_device(mock_start, mock_stop):
    """with_cost_tracking with no cuda_device skips cost tracking entirely."""
    from autoresearcher2.v3.cost_tracker import with_cost_tracking
    from autoresearcher2.v3.proposal import Proposal

    def inner_executor(proposal):
        return {"metrics": {"score": 3.0}}

    wrapped = with_cost_tracking(inner_executor, cuda_device=None)
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"x": "1"},
    )
    result = wrapped(p)

    assert result["metrics"]["score"] == 3.0
    mock_start.assert_not_called()
    mock_stop.assert_not_called()


@patch("autoresearcher2.v3.cost_tracker._stop_cost_job")
@patch("autoresearcher2.v3.cost_tracker._start_cost_job")
def test_with_cost_tracking_propagates_executor_exception(mock_start, mock_stop):
    """with_cost_tracking stops cost job even when executor raises."""
    from autoresearcher2.v3.cost_tracker import with_cost_tracking
    from autoresearcher2.v3.proposal import Proposal

    mock_start.return_value = "job_99"

    def failing_executor(proposal):
        raise RuntimeError("train.py failed")

    wrapped = with_cost_tracking(failing_executor, cuda_device="0")
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change",
        intervention_spec={"x": "1"},
    )

    with pytest.raises(RuntimeError, match="train.py failed"):
        wrapped(p)

    # Cost job should still be stopped to avoid leaking jobs
    mock_stop.assert_called_once_with("job_99")
