"""Tests for the autoresearch-style baseline.

This baseline mirrors Karpathy's autoresearch approach: the LLM sees a flat
results log (config → val_bpb) and suggests what to try next. No Bayesian model,
no factor importances, no appraisal signals. Fair comparison substrate.
"""

from unittest.mock import patch

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.llm.autoresearch_baseline import (
    build_flat_prompt,
    AutoresearchLLMAgent,
)


def make_schema():
    return InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )


def test_flat_prompt_contains_results_but_no_factor_importances():
    """The flat prompt should show results but NOT factor importances or appraisal."""
    schema = make_schema()
    history = [
        {"config": {"optimizer": "adam", "lr": "low"}, "val_bpb": 1.05},
        {"config": {"optimizer": "sgd", "lr": "high"}, "val_bpb": 1.08},
    ]
    prompt = build_flat_prompt(schema, history)

    # Should contain results
    assert "1.05" in prompt
    assert "1.08" in prompt
    assert "adam" in prompt

    # Should NOT contain structured Bayesian signals
    assert "factor importance" not in prompt.lower()
    assert "learntropy" not in prompt.lower()
    assert "appraisal" not in prompt.lower()
    assert "surprise" not in prompt.lower()
    assert "epistemic" not in prompt.lower()
    assert "coverage gap" not in prompt.lower()


def test_flat_prompt_shows_schema():
    schema = make_schema()
    prompt = build_flat_prompt(schema, [])
    assert "optimizer" in prompt
    assert "adam" in prompt
    assert "sgd" in prompt


def test_flat_prompt_with_empty_history():
    schema = make_schema()
    prompt = build_flat_prompt(schema, [])
    assert "No experiments run yet" in prompt


def test_agent_select_next_uses_llm():
    """Agent should call LLM and return a valid cell."""
    schema = make_schema()
    agent = AutoresearchLLMAgent(schema)

    # Mock the LLM call to return a suggestion
    mock_response = '[{"config": {"optimizer": "adam", "lr": "high"}, "reasoning": "test"}]'
    with patch(
        "autoresearcher2.llm.autoresearch_baseline._call_claude_flat",
        return_value=mock_response,
    ):
        cell = agent.select_next()

    assert 0 <= cell < schema.n_cells


def test_agent_observe_builds_history():
    """Agent should accumulate results in a flat log."""
    schema = make_schema()
    agent = AutoresearchLLMAgent(schema)
    agent.observe(0, 0.95)  # outcome = 2.0 - val_bpb
    agent.observe(1, 0.92)

    assert len(agent.history) == 2
    assert agent.history[0]["val_bpb"] == 1.05  # 2.0 - 0.95
    assert agent.history[1]["val_bpb"] == 1.08  # 2.0 - 0.92


def test_agent_falls_back_to_random_on_llm_failure():
    """If LLM fails, agent should pick a random valid cell."""
    schema = make_schema()
    agent = AutoresearchLLMAgent(schema, seed=42)

    with patch(
        "autoresearcher2.llm.autoresearch_baseline._call_claude_flat",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        cell = agent.select_next()

    assert 0 <= cell < schema.n_cells


def test_agent_first_experiment_is_random():
    """With no history, agent should pick randomly (no LLM call needed)."""
    schema = make_schema()
    agent = AutoresearchLLMAgent(schema, seed=42)

    # Should not call LLM for the first experiment
    with patch(
        "autoresearcher2.llm.autoresearch_baseline._call_claude_flat",
        side_effect=AssertionError("Should not be called"),
    ):
        cell = agent.select_next()

    assert 0 <= cell < schema.n_cells
