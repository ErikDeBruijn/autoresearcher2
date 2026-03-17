"""Tests for Observation — reality contact records."""
import json
import pytest
from autoresearcher2.v3.observation import Observation


def test_create_observation():
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.04"},
        outcome_metrics={"val_bpb": 0.97, "outcome": 1.03},
        outcome_success=True,
        wall_time_s=312.5,
        compute_cost=0.5,
    )
    assert obs.id.startswith("obs_")
    assert obs.outcome_success is True
    assert obs.wall_time_s == 312.5


def test_observation_failed():
    obs = Observation(
        intervention_type="code_change",
        intervention_spec={"patch": "diff..."},
        outcome_success=False,
        error="SyntaxError in patched file",
    )
    assert obs.outcome_success is False
    assert obs.error == "SyntaxError in patched file"
    assert obs.outcome_metrics is None


def test_observation_serialize_roundtrip():
    obs = Observation(
        intervention_type="probe",
        intervention_spec={"run_steps": 100},
        outcome_metrics={"loss": 2.3},
        outcome_success=True,
        wall_time_s=30.0,
    )
    data = obs.to_dict()
    obs2 = Observation.from_dict(data)
    assert obs2.id == obs.id
    assert obs2.outcome_metrics == {"loss": 2.3}


def test_observation_save_load(tmp_path):
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_metrics={"y": 0.5},
        outcome_success=True,
        wall_time_s=60.0,
    )
    path = tmp_path / f"{obs.id}.json"
    obs.save(path)
    obs2 = Observation.load(path)
    assert obs2.id == obs.id
    assert obs2.wall_time_s == 60.0


def test_observation_with_raw_log():
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        wall_time_s=60.0,
        raw_log="epoch 1: loss=2.3\nepoch 2: loss=1.8\n",
    )
    assert "epoch 1" in obs.raw_log


def test_observation_immutable_id():
    """Observation ID should not change after creation."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
    )
    original_id = obs.id
    data = obs.to_dict()
    obs2 = Observation.from_dict(data)
    assert obs2.id == original_id


def test_observation_artifact_paths():
    """Observations can store artifact paths (videos, models)."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"game": "Breakout"},
        outcome_metrics={"mean_reward": 3.6},
        outcome_success=True,
        wall_time_s=300.0,
    )
    obs.artifact_paths = {"video": "/tmp/breakout.mp4", "model": "/tmp/model.zip"}
    data = obs.to_dict()
    assert data["artifact_paths"]["video"] == "/tmp/breakout.mp4"
    obs2 = Observation.from_dict(data)
    assert obs2.artifact_paths["video"] == "/tmp/breakout.mp4"
    assert obs2.artifact_paths["model"] == "/tmp/model.zip"


def test_observation_artifact_paths_default_empty():
    """Artifact paths default to empty dict."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
    )
    assert obs.artifact_paths == {}
    data = obs.to_dict()
    obs2 = Observation.from_dict(data)
    assert obs2.artifact_paths == {}
