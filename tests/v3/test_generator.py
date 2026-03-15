"""Tests for generator — rationale-first proposal generation."""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.generator import generate_proposals, build_generator_prompt, DomainConfig, ATARI_DOMAIN, GENERIC_DOMAIN


def make_rich_world_model():
    """World model with beliefs, tensions, and cost data to reason about."""
    wm = WorldModel()
    wm.add_belief(
        claim="learning_rate=0.04 produces best outcomes",
        confidence=0.82,
        evidence_for=["obs_001", "obs_007"],
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
        evidence_against=["obs_009"],
    )
    wm.add_tension(
        belief_ids=["B1", "B3"],
        nature="If WD has minimal effect, why did obs_009 show big improvement with WD=0.1?",
        salience="high",
    )
    wm.cost_beliefs = {
        "config_change": {"wall_time_s": 300, "compute_cost": 0.5},
        "probe": {"wall_time_s": 30, "compute_cost": 0.05},
    }
    return wm


def test_generator_prompt_includes_world_model():
    wm = make_rich_world_model()
    prompt = build_generator_prompt(wm, n_proposals=5)
    assert "learning_rate=0.04" in prompt
    assert "WEIGHT_DECAY has minimal effect" in prompt
    assert "tension" in prompt.lower() or "T1" in prompt


def test_generator_prompt_mentions_cost():
    wm = make_rich_world_model()
    prompt = build_generator_prompt(wm, n_proposals=5)
    assert "cost" in prompt.lower()
    assert "probe" in prompt.lower()


def test_generator_produces_proposals():
    wm = make_rich_world_model()

    def mock_llm(prompt):
        return {
            "proposals": [
                {
                    "intent": "Test whether WD=0.1 effect is real or noise",
                    "rationale": "B3 has low confidence and contradictory evidence",
                    "expected_learning": "Whether WD significantly affects outcome at optimal lr",
                    "intervention_type": "config_change",
                    "intervention_spec": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.1"},
                    "estimated_cost": {"cost_to_test": "~5 min GPU", "cheaper_probe": None},
                },
                {
                    "intent": "Quick probe: does depth=10 crash or just slow down?",
                    "rationale": "B2 is underexplored, one data point is weak evidence",
                    "expected_learning": "Whether depth=10 is viable at all",
                    "intervention_type": "probe",
                    "intervention_spec": {"DEPTH": "10", "MATRIX_LR": "0.04", "run_steps": 100},
                    "estimated_cost": {"cost_to_test": "~30s", "cheaper_probe": None},
                },
                {
                    "intent": "Replicate best result for confidence",
                    "rationale": "B1 is high confidence but based on only 2 observations",
                    "expected_learning": "Whether lr=0.04 result is reproducible",
                    "intervention_type": "replication",
                    "intervention_spec": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.2"},
                    "estimated_cost": {"cost_to_test": "~5 min GPU", "cheaper_probe": None},
                },
            ]
        }

    proposals = generate_proposals(wm, n_proposals=3, llm_call_fn=mock_llm)

    assert len(proposals) == 3
    # Rationale-first: every proposal has intent and rationale
    for p in proposals:
        assert p.intent
        assert p.rationale
        assert p.expected_learning
        assert p.intervention_type
        assert p.intervention_spec
    # Diverse types
    types = {p.intervention_type for p in proposals}
    assert len(types) >= 2


def test_generator_handles_llm_failure():
    wm = make_rich_world_model()

    def failing_llm(prompt):
        raise RuntimeError("SSH failed")

    proposals = generate_proposals(wm, n_proposals=5, llm_call_fn=failing_llm)
    assert proposals == []


def test_generator_handles_malformed_response():
    wm = make_rich_world_model()

    def bad_llm(prompt):
        return {"not_proposals": "garbage"}

    proposals = generate_proposals(wm, n_proposals=5, llm_call_fn=bad_llm)
    assert proposals == []


def test_domain_config_changes_prompt():
    wm = make_rich_world_model()

    default_prompt = build_generator_prompt(wm, n_proposals=3)
    assert "NanoGPT" in default_prompt
    assert "DEPTH" in default_prompt

    atari_prompt = build_generator_prompt(wm, n_proposals=3, domain=ATARI_DOMAIN)
    assert "Atari" in atari_prompt
    assert "learning_rate" in atari_prompt
    assert "NanoGPT" not in atari_prompt

    generic_prompt = build_generator_prompt(wm, n_proposals=3, domain=GENERIC_DOMAIN)
    assert "NanoGPT" not in generic_prompt
    assert "Atari" not in generic_prompt
    assert "optimize a target metric" in generic_prompt


def test_custom_domain_config():
    wm = make_rich_world_model()
    outreach = DomainConfig(
        name="cold outreach",
        description="We optimize B2B cold email campaigns.",
        intervention_types="config_change (modify email parameters) or probe (small A/B test)",
        parameters="subject_line, tone, cta_position, personalization_level",
        diversity_hint="Mix of config_change and probe across different email elements",
    )
    prompt = build_generator_prompt(wm, n_proposals=3, domain=outreach)
    assert "cold email" in prompt
    assert "subject_line" in prompt
    assert "NanoGPT" not in prompt


def test_generate_proposals_with_domain():
    wm = make_rich_world_model()

    def mock_llm(prompt):
        assert "Atari" in prompt
        return {
            "proposals": [
                {
                    "intent": "Test PPO vs DQN on Breakout",
                    "rationale": "No evidence yet on algorithm choice",
                    "expected_learning": "Which algorithm family works best",
                    "intervention_type": "config_change",
                    "intervention_spec": {"game": "Breakout", "algorithm": "PPO"},
                    "estimated_cost": {"cost_to_test": "~30 min GPU"},
                }
            ]
        }

    proposals = generate_proposals(wm, n_proposals=1, llm_call_fn=mock_llm, domain=ATARI_DOMAIN)
    assert len(proposals) == 1
    assert proposals[0].intervention_spec["game"] == "Breakout"
