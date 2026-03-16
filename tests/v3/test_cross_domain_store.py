"""Cross-domain E2E test with Store (SQLite) + DomainConfig.

Verifies the full v4 pipeline works for non-NanoGPT domains:
generator with DomainConfig → critic → worker → orientation → repeat.
"""
import pytest
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker
from autoresearcher2.v3.generator import DomainConfig


def make_domain_llm(domain_proposals):
    """Mock LLM that returns domain-specific proposals."""
    def llm_call(prompt):
        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            return {
                "beliefs_added": [
                    {"claim": "Latest result informative", "confidence": 0.6,
                     "evidence_for": ["obs_new"]},
                ],
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            return {"proposals": domain_proposals}
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}
    return llm_call


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    s.init()
    yield s
    s.close()


def test_atari_full_cycle_with_store(store):
    """Full OODA cycle for Atari domain using Store backend."""
    llm = make_domain_llm([
        {
            "intent": "Test PPO on Breakout with higher lr",
            "rationale": "Default lr conservative",
            "expected_learning": "lr sensitivity",
            "intervention_type": "config_change",
            "intervention_spec": {"game": "Breakout", "learning_rate": "5e-4"},
            "estimated_cost": {"cost_to_test": "30 min"},
        },
        {
            "intent": "Try DQN on Pong",
            "rationale": "Algorithm diversity",
            "expected_learning": "Algorithm effect",
            "intervention_type": "config_change",
            "intervention_spec": {"game": "Pong", "algorithm": "DQN"},
            "estimated_cost": {"cost_to_test": "20 min"},
        },
    ])

    def execute(proposal):
        return {"metrics": {"mean_reward": 42.5}, "compute_cost": 0.5}

    planner = Planner(store, llm_call_fn=llm, min_queue_size=3, min_todo=2,
                      n_proposals=2, n_select=2, domain=DomainConfig(name="Atari Breakout", description="We train RL agents on ALE/Breakout-v5.", intervention_types="config_change or probe", parameters="learning_rate, n_envs", diversity_hint="Mix config_change and probe"))
    worker = Worker(store, execute_fn=execute, worker_id="test_worker")

    # Two cycles: generate → critique → execute → orient
    for _ in range(2):
        planner.tick()
        while worker.tick() is not None:
            pass

    assert store.count_proposals("done") + store.count_proposals("reviewed") >= 2
    assert len(store.list_observations()) >= 2
    wm = store.load_world_model()
    assert wm.version > 0
    assert len(wm.beliefs) > 0


def test_custom_domain_full_cycle(store):
    """Full cycle with a custom domain (cold outreach)."""
    outreach = DomainConfig(
        name="cold outreach",
        description="We optimize B2B cold email campaigns.",
        intervention_types="config_change (email params) or probe (small A/B test)",
        parameters="subject_line, tone, cta_position",
        diversity_hint="Mix of config_change and probe",
    )

    llm = make_domain_llm([
        {
            "intent": "Short subject line A/B test",
            "rationale": "Mobile truncation",
            "expected_learning": "Subject length effect",
            "intervention_type": "probe",
            "intervention_spec": {"subject_length": "short", "tone": "casual"},
            "estimated_cost": {"cost_to_test": "$5"},
        },
        {
            "intent": "Personalization depth test",
            "rationale": "Unknown ROI",
            "expected_learning": "Personalization value",
            "intervention_type": "config_change",
            "intervention_spec": {"personalization": "deep", "template": "v2"},
            "estimated_cost": {"cost_to_test": "$5"},
        },
    ])

    def execute(proposal):
        return {"metrics": {"open_rate": 0.35, "reply_rate": 0.08}}

    planner = Planner(store, llm_call_fn=llm, min_queue_size=3, min_todo=2,
                      n_proposals=2, n_select=2, domain=outreach)
    worker = Worker(store, execute_fn=execute, worker_id="outreach_worker")

    planner.tick()
    while worker.tick() is not None:
        pass
    planner.tick()

    obs = store.list_observations()
    assert len(obs) >= 2
    assert obs[0].outcome_metrics["open_rate"] == 0.35


def test_domain_config_reaches_prompt(store):
    """Verify DomainConfig actually changes what the LLM sees."""
    prompts_seen = []

    def capture_llm(prompt):
        prompts_seen.append(prompt)
        if "Generate" in prompt:
            return {"proposals": []}
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}

    planner = Planner(store, llm_call_fn=capture_llm, min_queue_size=3,
                      min_todo=2, n_proposals=2, domain=DomainConfig(name="Atari Breakout", description="We train RL agents on ALE/Breakout-v5.", intervention_types="config_change or probe", parameters="learning_rate, n_envs", diversity_hint="Mix config_change and probe"))
    planner.tick()

    # The generator prompt should mention CartPole
    gen_prompts = [p for p in prompts_seen if "Generate" in p]
    assert len(gen_prompts) > 0
    assert "Breakout" in gen_prompts[0]
    assert "NanoGPT" not in gen_prompts[0]
