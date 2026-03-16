"""Cross-domain validation: v3 system works for any optimization domain.

Tests that the generator-critic loop handles:
1. LLM training (config_change with hyperparameters)
2. Atari game optimization (config_change with game settings)
3. Cold outreach optimization (config_change with message parameters)
4. Mixed intervention types (code_change, probe, schema_extension)
"""
import pytest
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker
from autoresearcher2.v3.world_model import WorldModel


def make_domain_llm(domain_name, proposals_spec):
    """Create a mock LLM that generates domain-specific proposals."""
    call_count = [0]

    def llm_call(prompt):
        call_count[0] += 1
        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            return {
                "beliefs_added": [
                    {"claim": f"{domain_name}: latest result informative",
                     "confidence": 0.6, "evidence_for": ["obs_new"]},
                ],
                "cost_beliefs_updated": proposals_spec.get("cost_update", {}),
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            return {"proposals": proposals_spec["proposals"]}
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}

    return llm_call


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "research.db")
    s.init()
    yield s
    s.close()


def run_domain_cycle(store, domain_llm, execute_fn, cycles=2):
    """Run planner+worker cycles and return final state."""
    planner = Planner(store, llm_call_fn=domain_llm, min_queue_size=3, n_proposals=2, n_select=2)
    worker = Worker(store, execute_fn=execute_fn)

    for _ in range(cycles):
        planner.tick()
        while worker.tick() is not None:
            pass

    return {
        "done": store.count_proposals("done"),
        "observations": len(store.list_observations()),
        "world_model_version": store.load_world_model().version,
        "beliefs": len(store.load_world_model().beliefs),
    }


def test_llm_training_domain(store):
    """Domain: NanoGPT hyperparameter optimization."""
    llm = make_domain_llm("gpt", {
        "proposals": [
            {
                "intent": "Test higher learning rate",
                "rationale": "Current lr=0.04 might be suboptimal for depth=8",
                "expected_learning": "lr sensitivity at depth=8",
                "intervention_type": "config_change",
                "intervention_spec": {"DEPTH": "8", "MATRIX_LR": "0.08", "WEIGHT_DECAY": "0.1"},
                "estimated_cost": {"cost_to_test": "~5 min GPU"},
            },
            {
                "intent": "Quick probe depth=10",
                "rationale": "Untested region",
                "expected_learning": "Whether depth=10 trains at all",
                "intervention_type": "probe",
                "intervention_spec": {"DEPTH": "10", "MATRIX_LR": "0.04", "run_steps": 100},
                "estimated_cost": {"cost_to_test": "~30s"},
            },
        ],
        "cost_update": {"config_change": {"wall_time_s": 300}},
    })

    def execute(proposal):
        return {"metrics": {"val_bpb": 1.05}, "compute_cost": 0.50}

    result = run_domain_cycle(store, llm, execute)
    assert result["done"] >= 2
    assert result["observations"] >= 2
    assert result["world_model_version"] > 0


def test_atari_domain(store):
    """Domain: Atari game score optimization."""
    llm = make_domain_llm("atari", {
        "proposals": [
            {
                "intent": "Test PPO with higher learning rate on Breakout",
                "rationale": "Default lr might be too conservative for this game",
                "expected_learning": "lr sensitivity in Breakout",
                "intervention_type": "config_change",
                "intervention_spec": {"game": "Breakout", "learning_rate": "5e-4", "network_size": "medium"},
                "estimated_cost": {"cost_to_test": "~30 min GPU"},
            },
            {
                "intent": "Try DQN instead of PPO on Pong",
                "rationale": "PPO might not be optimal for simpler games",
                "expected_learning": "Algorithm choice effect on simple games",
                "intervention_type": "code_change",
                "intervention_spec": {"algorithm": "DQN", "game": "Pong"},
                "estimated_cost": {"cost_to_test": "~20 min GPU"},
            },
        ],
        "cost_update": {"config_change": {"wall_time_s": 1800}},
    })

    def execute(proposal):
        return {"metrics": {"mean_reward": 42.5, "max_reward": 100}}

    result = run_domain_cycle(store, llm, execute)
    assert result["done"] >= 2
    assert result["observations"] >= 2


def test_cold_outreach_domain(store):
    """Domain: Cold outreach message optimization."""
    llm = make_domain_llm("outreach", {
        "proposals": [
            {
                "intent": "Test shorter subject line",
                "rationale": "Current 8-word subjects might get truncated on mobile",
                "expected_learning": "Subject length effect on open rate",
                "intervention_type": "config_change",
                "intervention_spec": {"subject_length": "short", "tone": "casual", "cta_position": "top"},
                "estimated_cost": {"cost_to_test": "100 emails, ~$5"},
            },
            {
                "intent": "A/B test personalization depth",
                "rationale": "Unknown if deep personalization converts better",
                "expected_learning": "ROI of personalization effort",
                "intervention_type": "config_change",
                "intervention_spec": {"personalization": "deep", "template": "v2"},
                "estimated_cost": {"cost_to_test": "100 emails, ~$5"},
            },
        ],
        "cost_update": {},
    })

    def execute(proposal):
        return {"metrics": {"open_rate": 0.35, "reply_rate": 0.08, "meeting_rate": 0.02}}

    result = run_domain_cycle(store, llm, execute)
    assert result["done"] >= 2


def test_mixed_intervention_types(store):
    """All intervention types work through the pipeline."""
    llm = make_domain_llm("mixed", {
        "proposals": [
            {
                "intent": "Config change",
                "rationale": "test",
                "expected_learning": "test",
                "intervention_type": "config_change",
                "intervention_spec": {"x": "1"},
                "estimated_cost": {},
            },
            {
                "intent": "Code change",
                "rationale": "test",
                "expected_learning": "test",
                "intervention_type": "code_change",
                "intervention_spec": {"file": "train.py", "change": "add dropout"},
                "estimated_cost": {},
            },
        ],
        "cost_update": {},
    })

    types_executed = []

    def execute(proposal):
        types_executed.append(proposal.intervention_type)
        return {"metrics": {"result": 1.0}}

    run_domain_cycle(store, llm, execute, cycles=1)
    assert "config_change" in types_executed
    assert "code_change" in types_executed
