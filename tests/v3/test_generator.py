"""Tests for generator — rationale-first proposal generation."""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.generator import (
    generate_proposals, build_generator_prompt, DomainConfig,
    domain_config_from_project,
)


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
        salience=0.8,
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

    # Default (no domain) should use generic defaults
    default_prompt = build_generator_prompt(wm, n_proposals=3)
    assert "experiment" in default_prompt.lower() or "research" in default_prompt.lower()

    atari = DomainConfig(
        name="Atari Breakout",
        description="We train RL agents on ALE/Breakout-v5.",
        intervention_types="config_change or probe",
        parameters="learning_rate, n_envs, total_timesteps",
        diversity_hint="Mix config_change and code_change",
    )
    atari_prompt = build_generator_prompt(wm, n_proposals=3, domain=atari)
    assert "Breakout" in atari_prompt
    assert "learning_rate" in atari_prompt

    generic = DomainConfig(
        name="generic optimization",
        description="We run experiments to optimize a target metric.",
        intervention_types="config_change (modify parameters) or probe (cheap test)",
        parameters="any key-value pairs relevant to the domain",
        diversity_hint="Mix of config_change (full run) and probe (quick test)",
    )
    generic_prompt = build_generator_prompt(wm, n_proposals=3, domain=generic)
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
        assert "Breakout" in prompt
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

    atari_domain = DomainConfig(
        name="Atari Breakout",
        description="We train RL agents on ALE/Breakout-v5.",
        intervention_types="config_change or probe",
        parameters="learning_rate, n_envs, total_timesteps",
        diversity_hint="Mix config_change and code_change",
    )
    proposals = generate_proposals(wm, n_proposals=1, llm_call_fn=mock_llm, domain=atari_domain)
    assert len(proposals) == 1
    assert proposals[0].intervention_spec["game"] == "Breakout"


def test_domain_config_from_project_with_config():
    """domain_config_from_project reads domain_config dict from project metadata."""
    project = {
        "id": "proj-1",
        "name": "Protein Folding",
        "description": "Predict protein 3D structures",
        "domain_config": {
            "name": "Protein folding optimization",
            "description": "We predict protein 3D structures from amino acid sequences.",
            "intervention_types": "config_change (modify folding parameters) or probe (small test run)",
            "parameters": "temperature, num_iterations, model_size",
            "diversity_hint": "Mix of config_change and probe",
            "hardware": "4x A100 GPUs",
        },
    }
    dc = domain_config_from_project(project)
    assert isinstance(dc, DomainConfig)
    assert dc.name == "Protein folding optimization"
    assert "amino acid" in dc.description
    assert "temperature" in dc.parameters
    assert dc.hardware == "4x A100 GPUs"


def test_domain_config_from_project_without_config():
    """domain_config_from_project returns generic defaults when project has no domain_config."""
    project = {"id": "proj-2", "name": "My Experiment", "description": "Some research"}
    dc = domain_config_from_project(project)
    assert isinstance(dc, DomainConfig)
    # Should use generic defaults, not NanoGPT-specific content
    assert "NanoGPT" not in dc.name
    assert "NanoGPT" not in dc.description
    assert "DEPTH" not in dc.parameters
    assert "MATRIX_LR" not in dc.parameters


def test_domain_config_from_project_generates_proposals():
    """Full round-trip: project metadata -> domain config -> generator -> proposals."""
    wm = make_rich_world_model()
    project = {
        "id": "proj-bio",
        "name": "Drug Discovery",
        "domain_config": {
            "name": "Drug binding affinity",
            "description": "We optimize molecular binding affinity scores.",
            "intervention_types": "config_change (modify docking parameters) or probe (quick screen)",
            "parameters": "ligand_type, receptor_site, scoring_function, num_conformers",
            "diversity_hint": "Mix docking parameter tweaks and quick screening probes",
        },
    }
    dc = domain_config_from_project(project)

    def mock_llm(prompt):
        assert "binding affinity" in prompt
        assert "ligand_type" in prompt
        assert "NanoGPT" not in prompt
        return {
            "proposals": [
                {
                    "intent": "Test alternative scoring function",
                    "rationale": "Current scoring may miss key interactions",
                    "expected_learning": "Whether Vina scoring outperforms default",
                    "intervention_type": "config_change",
                    "intervention_spec": {"scoring_function": "vina", "num_conformers": 100},
                    "estimated_cost": {"cost_to_test": "~10 min CPU"},
                }
            ]
        }

    proposals = generate_proposals(wm, n_proposals=1, llm_call_fn=mock_llm, domain=dc)
    assert len(proposals) == 1
    assert proposals[0].intervention_spec["scoring_function"] == "vina"
