import numpy as np
from autoresearcher2.toy.pomdp import ToyPOMDP
from autoresearcher2.toy.active_inference import ActiveInferenceAgent


def test_pomdp_creation():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    assert pomdp.n_states == 27
    assert pomdp.n_observations == 5  # outcome bands


def test_agent_selects_action():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)
    action = agent.select_action()
    assert 0 <= action < pomdp.n_states


def test_agent_updates_beliefs():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)
    beliefs_before = agent.beliefs.copy()
    action = agent.select_action()
    obs = pomdp.observe(action)
    agent.update(action, obs)
    assert not np.allclose(agent.beliefs, beliefs_before)


def test_epistemic_decreases_over_time():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)

    epistemic_values = []
    for _ in range(50):
        efe = agent.compute_efe_all_actions()
        avg_epistemic = np.mean([e["epistemic"] for e in efe.values()])
        epistemic_values.append(avg_epistemic)
        action = agent.select_action()
        obs = pomdp.observe(action)
        agent.update(action, obs)

    first_quarter = np.mean(epistemic_values[:12])
    last_quarter = np.mean(epistemic_values[-12:])
    assert last_quarter < first_quarter


def test_canonical_efe_decomposition():
    """EFE should decompose into pragmatic + epistemic."""
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)
    efe = agent.compute_efe_all_actions()
    for action, components in efe.items():
        assert "pragmatic" in components
        assert "epistemic" in components
        assert "total" in components
        np.testing.assert_almost_equal(
            components["total"],
            components["pragmatic"] + components["epistemic"],
            decimal=10,
        )
