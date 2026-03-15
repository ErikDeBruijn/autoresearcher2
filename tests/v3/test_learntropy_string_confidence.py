"""Tests for learntropy handling of string confidence values.

LLMs sometimes return confidence as a string ("0.8") instead of float (0.8).
compute_learntropy must handle this gracefully without crashing.
This was a real production bug: every planner cycle crashed with
TypeError: unsupported operand type(s) for -: 'str' and 'str'
"""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.learntropy import compute_learntropy, _find_belief_confidence


def make_wm_with_string_confidence():
    """Create a WorldModel where beliefs have string confidence values."""
    wm = WorldModel()
    wm.add_belief(claim="lr=0.04 is optimal", confidence=0.7, evidence_for=["obs_1"])
    # Simulate LLM corruption: overwrite confidence with a string
    wm.beliefs[0]["confidence"] = "0.7"
    return wm


def test_string_confidence_in_before_wm():
    """compute_learntropy doesn't crash when wm_before has string confidence."""
    wm_before = make_wm_with_string_confidence()
    wm_after = WorldModel()
    bid = wm_before.beliefs[0]["id"]

    delta = {
        "beliefs_revised": [
            {"id": bid, "new_confidence": 0.9, "reason": "confirmed"},
        ],
    }
    # This would crash with: TypeError: unsupported operand type(s) for -: 'float' and 'str'
    score = compute_learntropy(wm_before, wm_after, delta)
    assert score > 0.0


def test_string_confidence_in_delta():
    """compute_learntropy doesn't crash when delta has string new_confidence."""
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.5, evidence_for=["obs_1"])
    bid = wm.beliefs[0]["id"]

    delta = {
        "beliefs_revised": [
            {"id": bid, "new_confidence": "0.9", "reason": "string from LLM"},
        ],
    }
    # This would crash with: TypeError: unsupported operand type(s) for -: 'str' and 'float'
    score = compute_learntropy(wm, wm, delta)
    assert score > 0.0


def test_both_string_confidences():
    """Both old and new confidence are strings — the original crash."""
    wm = make_wm_with_string_confidence()
    bid = wm.beliefs[0]["id"]

    delta = {
        "beliefs_revised": [
            {"id": bid, "new_confidence": "0.95", "reason": "LLM returned strings"},
        ],
    }
    # This was the exact crash: TypeError: unsupported operand type(s) for -: 'str' and 'str'
    score = compute_learntropy(wm, wm, delta)
    assert score > 0.0


def test_non_numeric_confidence_skipped():
    """Non-numeric confidence values are gracefully skipped."""
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.5, evidence_for=[])
    bid = wm.beliefs[0]["id"]

    delta = {
        "beliefs_revised": [
            {"id": bid, "new_confidence": "high", "reason": "LLM returned text"},
        ],
    }
    # Should not crash, should produce a score (just without the confidence shift component)
    score = compute_learntropy(wm, wm, delta)
    assert isinstance(score, float)


def test_find_belief_confidence_returns_float():
    """_find_belief_confidence coerces string values to float."""
    wm = make_wm_with_string_confidence()
    bid = wm.beliefs[0]["id"]
    conf = _find_belief_confidence(wm, bid)
    assert isinstance(conf, float)
    assert conf == 0.7


def test_find_belief_confidence_non_numeric():
    """_find_belief_confidence returns None for non-numeric confidence."""
    wm = WorldModel()
    wm.add_belief(claim="test", confidence="high", evidence_for=[])
    bid = wm.beliefs[0]["id"]
    conf = _find_belief_confidence(wm, bid)
    assert conf is None
