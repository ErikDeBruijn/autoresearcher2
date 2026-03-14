"""Integration test: full orientation → generator → critic cycle.

Tests v3.0 design criteria:
1. Generator produces diverse proposals
2. Rationale-first ordering
3. Critic ranking is defensible
4. Cost awareness
5. Not limited to parameter grid
6. World model update is structured
"""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.observation import Observation
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.orientation import orient
from autoresearcher2.v3.generator import generate_proposals
from autoresearcher2.v3.critic import critique_proposals


class MockLLM:
    """Mock LLM that tracks calls and returns canned responses per role."""

    def __init__(self):
        self.calls = []

    def orientation_response(self, prompt):
        self.calls.append(("orientation", prompt))
        return {
            "beliefs_added": [
                {"claim": "WD=0.1 at lr=0.04 improves outcome", "confidence": 0.6,
                 "evidence_for": ["obs_new"]},
            ],
            "beliefs_revised": [
                {"id": "B1", "new_confidence": 0.85, "new_evidence_for": ["obs_new"],
                 "reason": "Another data point confirms lr=0.04 is strong"},
            ],
            "tensions_added": [
                {"beliefs": ["B1", "B3"], "nature": "lr=0.04 is strong but WD also matters",
                 "salience": "medium"},
            ],
            "cost_beliefs_updated": {"config_change": {"wall_time_s": 310}},
        }

    def generator_response(self, prompt):
        self.calls.append(("generator", prompt))
        return {
            "proposals": [
                {
                    "intent": "Resolve tension between lr dominance and WD effect",
                    "rationale": "T1 is unresolved, WD at different lr values not tested",
                    "expected_learning": "Whether WD effect is lr-dependent or independent",
                    "intervention_type": "config_change",
                    "intervention_spec": {"DEPTH": "8", "MATRIX_LR": "0.08", "WEIGHT_DECAY": "0.1"},
                    "estimated_cost": {"cost_to_test": "~5 min GPU", "cheaper_probe": None},
                },
                {
                    "intent": "Quick probe: is depth=10 viable?",
                    "rationale": "B2 has very low confidence (0.45), one probe could double evidence",
                    "expected_learning": "Whether depth=10 trains successfully at all",
                    "intervention_type": "probe",
                    "intervention_spec": {"DEPTH": "10", "MATRIX_LR": "0.04", "run_steps": 100},
                    "estimated_cost": {"cost_to_test": "~30s", "cheaper_probe": None},
                },
                {
                    "intent": "Try different optimizer to test if results generalize",
                    "rationale": "All experiments use same optimizer, this is a blind spot",
                    "expected_learning": "Whether our findings are optimizer-specific",
                    "intervention_type": "code_change",
                    "intervention_spec": {"change": "replace Adam with AdamW", "file": "train.py"},
                    "estimated_cost": {"cost_to_test": "~10 min GPU", "cheaper_probe": "Just check if it compiles first"},
                },
                {
                    "intent": "Confirm best config is reproducible",
                    "rationale": "Only tested once",
                    "expected_learning": "Variance of our best result",
                    "intervention_type": "replication",
                    "intervention_spec": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.2"},
                    "estimated_cost": {"cost_to_test": "~5 min GPU", "cheaper_probe": None},
                },
                {
                    "intent": "Extend schema with batch_size factor",
                    "rationale": "Never tested batch_size, could be dominant and we're blind to it",
                    "expected_learning": "Whether batch_size should be in our search space",
                    "intervention_type": "schema_extension",
                    "intervention_spec": {"new_factor": "BATCH_SIZE", "levels": ["32", "64", "128"]},
                    "estimated_cost": {"cost_to_test": "3 runs × ~5 min", "cheaper_probe": "One run with batch=128 as probe"},
                },
            ]
        }

    def critic_response(self, prompt):
        self.calls.append(("critic", prompt))
        return {
            "rankings": [
                {"proposal_id": None, "decision": "accept", "rank": 1,
                 "rationale": "Cheapest probe, tests weakest belief (B2)"},
                {"proposal_id": None, "decision": "accept", "rank": 2,
                 "rationale": "Directly resolves highest-salience tension (T1)"},
                {"proposal_id": None, "decision": "deprioritize", "rank": 3,
                 "rationale": "Good idea but expensive and not urgent"},
                {"proposal_id": None, "decision": "reject", "rank": None,
                 "rationale": "Replication adds confidence but teaches nothing new"},
                {"proposal_id": None, "decision": "deprioritize", "rank": 4,
                 "rationale": "Schema extension is valuable but premature"},
            ]
        }


def make_initial_world_model():
    wm = WorldModel()
    wm.add_belief(
        claim="lr=0.04 produces best outcomes",
        confidence=0.7,
        evidence_for=["obs_001", "obs_002"],
    )
    wm.add_belief(
        claim="DEPTH > 8 has diminishing returns",
        confidence=0.45,
        evidence_for=["obs_003"],
    )
    wm.add_belief(
        claim="WEIGHT_DECAY has minimal effect",
        confidence=0.3,
        evidence_for=["obs_002"],
        evidence_against=["obs_004"],
    )
    wm.cost_beliefs = {
        "config_change": {"wall_time_s": 300, "compute_cost": 0.5},
        "probe": {"wall_time_s": 30, "compute_cost": 0.05},
    }
    return wm


def test_full_cycle():
    """One complete OODA cycle: observe → orient → decide (generate + critique)."""
    mock = MockLLM()
    wm = make_initial_world_model()

    # Step 1: New observation arrives (Observe)
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.1"},
        outcome_metrics={"val_bpb": 0.97, "outcome": 1.03},
        outcome_success=True,
        wall_time_s=310.0,
        compute_cost=0.5,
    )

    # Step 2: Orientation (Orient) — LLM updates world model
    delta = orient(wm, obs, mock.orientation_response)
    assert wm.version == 1
    assert len(wm.beliefs) == 4  # 3 original + 1 added
    assert wm.beliefs[0]["confidence"] == 0.85  # B1 revised up
    assert len(wm.tensions) == 1  # New tension added
    assert wm.cost_beliefs["config_change"]["wall_time_s"] == 310

    # Step 3: Generator (Decide) — LLM proposes experiments
    proposals = generate_proposals(wm, n_proposals=5, llm_call_fn=mock.generator_response)
    assert len(proposals) == 5

    # Verify rationale-first: every proposal has intent before intervention
    for p in proposals:
        assert p.intent
        assert p.rationale
        assert p.expected_learning
        d = p.to_dict()
        keys = list(d.keys())
        assert keys.index("intent") < keys.index("intervention_type")

    # Verify diversity: not all config_change
    types = {p.intervention_type for p in proposals}
    assert len(types) >= 3  # config_change, probe, code_change, replication, schema_extension

    # Step 4: Critic (Decide) — LLM ranks and selects
    # Need to fix proposal IDs in critic response
    for i, ranking in enumerate(mock.critic_response.__code__.co_consts):
        pass  # We'll match by position instead

    # Set proposal IDs in the mock response
    def critic_with_ids(prompt):
        response = mock.critic_response(prompt)
        # The critic response was already recorded, modify the last one
        # Match proposal IDs to rankings
        for i, ranking in enumerate(response["rankings"]):
            if i < len(proposals):
                ranking["proposal_id"] = proposals[i].id
        return response

    # Re-call critic (remove the duplicate call from mock)
    mock.calls = [c for c in mock.calls if c[0] != "critic"]
    accepted = critique_proposals(wm, proposals, n_select=2, llm_call_fn=critic_with_ids)

    assert len(accepted) == 2
    # Both should have critic decisions
    for p in accepted:
        assert p.critic is not None
        assert p.critic["decision"] == "accept"

    # Verify 3 LLM calls were made: orientation, generator, critic
    call_types = [c[0] for c in mock.calls]
    assert "orientation" in call_types
    assert "generator" in call_types
    assert "critic" in call_types


def test_world_model_update_is_structured():
    """Test criterion 6: delta updates are structured, not free text."""
    mock = MockLLM()
    wm = make_initial_world_model()
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"DEPTH": "10", "MATRIX_LR": "0.04"},
        outcome_metrics={"outcome": 0.85},
        outcome_success=True,
        wall_time_s=600.0,
    )

    delta = orient(wm, obs, mock.orientation_response)

    # Delta should be a dict with structured fields, not free text
    assert isinstance(delta, dict)
    assert any(k in delta for k in (
        "beliefs_added", "beliefs_revised", "beliefs_retired",
        "tensions_added", "tensions_resolved", "cost_beliefs_updated",
    ))


def test_cost_awareness_in_prompts():
    """Both generator and critic prompts should reference cost information."""
    wm = make_initial_world_model()

    from autoresearcher2.v3.generator import build_generator_prompt
    from autoresearcher2.v3.critic import build_critic_prompt

    gen_prompt = build_generator_prompt(wm, n_proposals=5)
    assert "cost" in gen_prompt.lower()
    assert "300" in gen_prompt  # wall_time_s from cost_beliefs

    proposals = [
        Proposal(intent="test", rationale="test", expected_learning="test",
                 intervention_type="probe", intervention_spec={"x": "1"},
                 estimated_cost={"cost_to_test": "~30s"}),
    ]
    critic_prompt = build_critic_prompt(wm, proposals, n_select=1)
    assert "cost" in critic_prompt.lower()
