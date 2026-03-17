"""Tests for domain-agnostic chat agent prompt and project creation.

Ensures the chat system prompt doesn't hardcode NanoGPT-specific terms
when the active project uses a different domain.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from api import _build_chat_system_prompt


def test_chat_prompt_no_nanogpt_for_custom_domain(store):
    """When project is a non-ML domain, prompt must not contain 'NanoGPT' or 'val_bpb'."""
    # Create a biology experiment project
    pid = store.create_project(
        name="Enzyme Optimization",
        description="We optimize enzyme activity through mutation experiments.",
        domain_config={
            "name": "Enzyme optimization",
            "description": "Optimize enzyme activity via mutations.",
            "target_metric": "activity_score",
            "optimize": "maximize",
            "intervention_types": "config_change, probe",
            "parameters": "mutation_site, temperature, pH",
        },
    )

    prompt = _build_chat_system_prompt(store)

    assert "NanoGPT" not in prompt
    assert "val_bpb" not in prompt
    assert "Enzyme" in prompt or "enzyme" in prompt


def test_chat_prompt_uses_project_target_metric(store):
    """The prompt should reference the project's target_metric, not hardcoded val_bpb."""
    store.create_project(
        name="Solar Cell Efficiency",
        description="Optimize photovoltaic cell parameters.",
        domain_config={
            "name": "Solar cell optimization",
            "target_metric": "efficiency_pct",
            "optimize": "maximize",
            "intervention_types": "config_change",
            "parameters": "material, thickness, doping",
        },
    )

    prompt = _build_chat_system_prompt(store)

    assert "efficiency_pct" in prompt
    assert "val_bpb" not in prompt


def test_chat_prompt_no_hardcoded_domain_list(store):
    """The prompt should not list 'nanogpt, atari-rl, generic' as the only domains."""
    store.create_project(
        name="Test Project",
        description="A test.",
        domain_config={
            "name": "test",
            "target_metric": "score",
            "intervention_types": "config_change",
            "parameters": "x, y",
        },
    )

    prompt = _build_chat_system_prompt(store)

    # Should not have the old hardcoded domain list
    assert "Domains: nanogpt, atari-rl, generic" not in prompt


def test_observation_formatting_uses_project_metric(store):
    """Observation lines in the prompt should use the project's target metric, not val_bpb."""
    from autoresearcher2.v3.observation import Observation
    import time

    pid = store.create_project(
        name="RL Project",
        description="RL agent training.",
        domain_config={
            "name": "RL training",
            "target_metric": "mean_reward",
            "optimize": "maximize",
            "intervention_types": "config_change",
            "parameters": "lr, gamma",
        },
    )

    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"lr": "0.001"},
        outcome_metrics={"mean_reward": 250.5, "episodes": 100},
        outcome_success=True,
    )
    store.save_observation(obs, project_id=pid)

    prompt = _build_chat_system_prompt(store)

    # Should show the actual metric, not val_bpb
    assert "mean_reward" in prompt
    assert "val_bpb=?" not in prompt


def test_domain_configs_accept_custom_dict(store):
    """Project creation should accept any domain_config dict, not just known templates."""
    pid = store.create_project(
        name="Custom Domain",
        description="Something totally custom.",
        domain_config={
            "name": "Custom experiments",
            "target_metric": "custom_score",
            "optimize": "minimize",
            "intervention_types": "probe, config_change",
            "parameters": "alpha, beta, gamma",
            "custom_field": "anything goes",
        },
    )

    project = store.get_project(pid)
    assert project["domain_config"]["custom_field"] == "anything goes"
    assert project["domain_config"]["target_metric"] == "custom_score"
