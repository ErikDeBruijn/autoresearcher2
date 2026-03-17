"""Tests for Proposal — rationale-first experiment proposals."""
import json
import pytest
from autoresearcher2.v3.proposal import Proposal


def test_create_proposal():
    p = Proposal(
        intent="Test whether lr generalizes across depths",
        rationale="lr=0.04 works at depth=8, untested at depth=10",
        expected_learning="Clarifies lr×depth interaction",
        intervention_type="config_change",
        intervention_spec={"DEPTH": "10", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.2"},
        estimated_cost={"cost_to_test": "~5 min GPU", "cheaper_probe": None},
    )
    assert p.id.startswith("prop_")
    assert p.status == "backlog"
    assert p.intent == "Test whether lr generalizes across depths"
    assert p.critic is None


def test_proposal_rationale_first_fields():
    """Rationale-first: intent, rationale, expected_learning come before intervention."""
    p = Proposal(
        intent="Explore untested region",
        rationale="High uncertainty in this area",
        expected_learning="Whether depth matters at high lr",
        intervention_type="config_change",
        intervention_spec={"x": "1"},
    )
    d = p.to_dict()
    keys = list(d.keys())
    # intent, rationale, expected_learning should appear before intervention_type
    assert keys.index("intent") < keys.index("intervention_type")
    assert keys.index("rationale") < keys.index("intervention_type")
    assert keys.index("expected_learning") < keys.index("intervention_type")


def test_proposal_serialize_roundtrip():
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    data = p.to_dict()
    p2 = Proposal.from_dict(data)
    assert p2.id == p.id
    assert p2.intent == p.intent
    assert p2.status == "backlog"



def test_proposal_project_id_roundtrip():
    """project_id survives serialization roundtrip."""
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.project_id = "proj_abc123"
    data = p.to_dict()
    assert data["project_id"] == "proj_abc123"
    p2 = Proposal.from_dict(data)
    assert p2.project_id == "proj_abc123"


def test_proposal_set_critic_decision():
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.set_critic_decision(decision="accept", rank=1, rationale="High value, low cost")
    assert p.critic["decision"] == "accept"
    assert p.critic["rank"] == 1
    assert p.critic["rationale"] == "High value, low cost"


def test_proposal_reject():
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.set_critic_decision(decision="reject", rank=None, rationale="Redundant with obs_003")
    assert p.critic["decision"] == "reject"


def test_proposal_various_intervention_types():
    """Proposals can be any type, not just config changes."""
    types = ["config_change", "code_change", "schema_extension", "probe", "replication", "other"]
    for t in types:
        p = Proposal(
            intent="test", rationale="test", expected_learning="test",
            intervention_type=t, intervention_spec={"detail": t},
        )
        assert p.intervention_type == t


def test_proposal_promote_to_todo():
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    assert p.status == "backlog"
    p.set_critic_decision(decision="accept", rank=1, rationale="good")
    p.promote("todo")
    assert p.status == "todo"


def test_proposal_complete():
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.promote("todo")
    p.promote("running")
    p.complete(observation_id="obs_001")
    assert p.status == "done"
    assert p.observation_id == "obs_001"
