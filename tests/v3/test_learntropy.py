"""Tests for learntropy calculation."""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.learntropy import compute_learntropy


def make_wm_with_beliefs(n=3):
    wm = WorldModel()
    for i in range(n):
        wm.add_belief(claim=f"belief {i}", confidence=0.5, evidence_for=[f"obs_{i}"])
    return wm


def test_empty_delta_is_zero():
    wm = make_wm_with_beliefs()
    assert compute_learntropy(wm, wm, {}) == 0.0


def test_no_change_delta_is_zero():
    wm = make_wm_with_beliefs()
    delta = {"beliefs_added": [], "beliefs_revised": [], "tensions_added": []}
    assert compute_learntropy(wm, wm, delta) == 0.0


def test_adding_belief_increases_learntropy():
    wm_before = make_wm_with_beliefs(3)
    wm_after = make_wm_with_beliefs(3)
    wm_after.add_belief(claim="new", confidence=0.6, evidence_for=["obs_new"])

    delta = {"beliefs_added": [{"claim": "new", "confidence": 0.6}]}
    score = compute_learntropy(wm_before, wm_after, delta)
    assert score > 0.0


def test_revising_belief_with_large_shift():
    wm_before = make_wm_with_beliefs(3)
    wm_after = make_wm_with_beliefs(3)

    belief_id = wm_before.beliefs[0]["id"]
    delta = {
        "beliefs_revised": [
            {"id": belief_id, "new_confidence": 0.95, "reason": "strong evidence"},
        ],
    }
    score = compute_learntropy(wm_before, wm_after, delta)
    assert score > 0.2  # Large confidence shift (0.5 → 0.95)


def test_retiring_belief_is_highly_informative():
    wm_before = make_wm_with_beliefs(3)
    wm_after = make_wm_with_beliefs(2)

    delta = {"beliefs_retired": [{"id": wm_before.beliefs[2]["id"], "reason": "contradicted"}]}
    score = compute_learntropy(wm_before, wm_after, delta)
    assert score > 0.1  # Retirement bonus


def test_tension_changes_increase_learntropy():
    wm_before = make_wm_with_beliefs(3)
    wm_before.add_tension(belief_ids=["B0", "B1"], nature="conflict", salience=0.8)

    delta = {
        "tensions_added": [{"beliefs": ["B1", "B2"], "nature": "new conflict", "salience": 0.5}],
        "tensions_resolved": [{"id": wm_before.tensions[0]["id"], "resolution": "resolved"}],
    }
    score = compute_learntropy(wm_before, wm_before, delta)
    assert score > 0.0


def test_massive_change_caps_at_one():
    wm_before = make_wm_with_beliefs(1)
    delta = {
        "beliefs_added": [{"claim": f"new {i}", "confidence": 0.5} for i in range(10)],
        "beliefs_revised": [{"id": wm_before.beliefs[0]["id"], "new_confidence": 0.99}],
        "beliefs_retired": [{"id": "some_id", "reason": "wrong"}],
        "tensions_added": [{"beliefs": ["a", "b"], "nature": "x", "salience": 0.8}] * 5,
        "cost_beliefs_updated": {"config_change": {"wall_time_s": 999}},
    }
    score = compute_learntropy(wm_before, wm_before, delta)
    assert score <= 1.0


def test_cost_update_adds_small_learntropy():
    wm = make_wm_with_beliefs(3)
    delta = {"cost_beliefs_updated": {"probe": {"wall_time_s": 45}}}
    score = compute_learntropy(wm, wm, delta)
    assert 0.05 < score < 0.3  # Small but nonzero
