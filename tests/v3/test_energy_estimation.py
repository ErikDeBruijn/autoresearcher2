"""Tests for energy estimation logic.

When observations don't have cost tracking data (energy_kwh is None),
the stats should estimate energy based on wall_time and average power
from tracked observations. This ensures total cost display is meaningful
even when gpu-cost-tracker was added after experiments started.
"""
import pytest
from autoresearcher2.v3.observation import Observation


def _make_obs(store, wall_time_s, energy_kwh=None, cost_eur=None, avg_power_w=None, success=True):
    """Helper to create and save an observation."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=success,
        wall_time_s=wall_time_s,
        energy_kwh=energy_kwh,
        cost_eur=cost_eur,
        avg_power_w=avg_power_w,
    )
    store.save_observation(obs)
    return obs


def test_tracked_observations_sum_exactly(store):
    """Observations with cost data should sum to exact totals."""
    _make_obs(store, wall_time_s=300, energy_kwh=0.10, cost_eur=0.023, avg_power_w=400)
    _make_obs(store, wall_time_s=300, energy_kwh=0.12, cost_eur=0.028, avg_power_w=420)

    observations = store.list_observations()
    total_energy = sum(o.energy_kwh for o in observations if o.energy_kwh)
    total_cost = sum(o.cost_eur for o in observations if o.cost_eur)

    assert abs(total_energy - 0.22) < 0.001
    assert abs(total_cost - 0.051) < 0.001


def test_untracked_observations_have_none_cost(store):
    """Observations without cost tracking have None for energy fields."""
    _make_obs(store, wall_time_s=300)
    obs = store.list_observations()[0]
    assert obs.energy_kwh is None
    assert obs.cost_eur is None
    assert obs.avg_power_w is None


def test_mixed_tracked_untracked_estimation():
    """
    Energy estimation logic: use avg power from tracked runs
    to estimate untracked runs' energy.

    This mirrors the logic in web/api.py get_stats().
    """
    # Simulate the estimation algorithm
    tracked_observations = [
        {"wall_time_s": 300, "energy_kwh": 0.034, "cost_eur": 0.008, "avg_power_w": 386},
        {"wall_time_s": 310, "energy_kwh": 0.035, "cost_eur": 0.008, "avg_power_w": 390},
    ]
    untracked_observations = [
        {"wall_time_s": 300, "energy_kwh": None},
        {"wall_time_s": 315, "energy_kwh": None},
        {"wall_time_s": 305, "energy_kwh": None},
    ]

    # Algorithm from api.py
    total_energy = 0.0
    total_cost = 0.0
    avg_power_samples = []
    untracked_wall_time = 0.0

    for obs in tracked_observations:
        total_energy += obs["energy_kwh"]
        total_cost += obs["cost_eur"]
        avg_power_samples.append(obs["avg_power_w"])

    for obs in untracked_observations:
        untracked_wall_time += obs["wall_time_s"]

    if untracked_wall_time > 0 and avg_power_samples:
        avg_power_w = sum(avg_power_samples) / len(avg_power_samples)
        estimated_kwh = (avg_power_w * untracked_wall_time) / 3_600_000
        total_energy += estimated_kwh
        total_cost += estimated_kwh * 0.23

    # 2 tracked: 0.069 kWh
    # 3 untracked: 388W avg * 920s / 3600000 = ~0.099 kWh
    # Total should be ~0.168 kWh
    assert total_energy > 0.15
    assert total_energy < 0.20
    assert total_cost > 0.03  # not just tracked cost

    # Without estimation, we'd only have 0.069 kWh — much too low
    tracked_only = sum(o["energy_kwh"] for o in tracked_observations)
    assert total_energy > tracked_only * 2  # estimation at least doubles it


def test_no_tracked_runs_no_estimation():
    """If no runs have power data, no estimation is possible."""
    untracked_wall_time = 1000.0
    avg_power_samples = []

    estimated = 0.0
    if untracked_wall_time > 0 and avg_power_samples:
        # This branch should NOT execute
        avg_power_w = sum(avg_power_samples) / len(avg_power_samples)
        estimated = (avg_power_w * untracked_wall_time) / 3_600_000

    assert estimated == 0.0


def test_observation_project_id_in_to_dict(store):
    """Observation.to_dict() includes project_id for per-project energy tracking."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        wall_time_s=300,
        energy_kwh=0.1,
    )
    obs.project_id = "proj_abc"
    d = obs.to_dict()
    assert d["project_id"] == "proj_abc"


def test_observation_project_id_none_in_to_dict():
    """Observation without project_id serializes as None."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
    )
    d = obs.to_dict()
    assert d["project_id"] is None
