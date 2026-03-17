"""Tests for chat-driven project creation.

When the chat LLM emits a CREATE_PROJECT command in its response,
the backend detects it, creates the project in the Store, and
replaces the command block with a success message.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

# Import from web/api.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from api import _execute_chat_commands


def test_no_command_passthrough():
    """Response without commands passes through unchanged."""
    text = "The current val_bpb is 1.05, which is great progress!"
    with patch("api.get_store") as mock:
        result = _execute_chat_commands(text)
    assert result == text
    mock.assert_not_called()


def test_create_project_with_code_fence(store):
    """CREATE_PROJECT in code fence is detected and executed."""
    response = """I'll create a research project for you.

```command
CREATE_PROJECT {"name": "Optimization Study", "description": "Optimize target parameters", "parameters": "learning_rate,batch_size,depth"}
```

This will set up the project with the specified parameters."""

    with patch("api.get_store", return_value=store):
        result = _execute_chat_commands(response)

    assert "Project created" in result
    assert "Optimization Study" in result
    assert "```command" not in result  # Command block replaced

    # Verify project was actually created in store
    projects = store.list_projects()
    assert len(projects) == 1
    assert projects[0]["name"] == "Optimization Study"
    assert projects[0]["description"] == "Optimize target parameters"
    cfg = projects[0]["domain_config"]
    assert cfg["parameters"] == "learning_rate,batch_size,depth"


def test_create_project_without_code_fence(store):
    """CREATE_PROJECT without code fence is also detected."""
    response = 'Here we go: CREATE_PROJECT {"name": "Quick Test", "domain_type": "generic"}'

    with patch("api.get_store", return_value=store):
        result = _execute_chat_commands(response)

    assert "Project created" in result
    projects = store.list_projects()
    assert len(projects) == 1
    assert projects[0]["name"] == "Quick Test"


def test_no_domain_config_creates_project_with_none(store):
    """When no domain_config is provided, project is created with domain_config=None."""
    response = '```command\nCREATE_PROJECT {"name": "Test Project", "domain_type": "generic"}\n```'

    with patch("api.get_store", return_value=store):
        _execute_chat_commands(response)

    projects = store.list_projects()
    assert projects[0]["domain_config"] is None


def test_invalid_json_handled():
    """Invalid JSON in command adds error message instead of crashing."""
    response = 'CREATE_PROJECT {invalid json here}'

    with patch("api.get_store") as mock:
        result = _execute_chat_commands(response)

    assert "Failed to parse" in result


def test_surrounding_text_preserved(store):
    """Text before and after the command is preserved."""
    response = """Great choice! Let me set that up.

```command
CREATE_PROJECT {"name": "Test", "domain_type": "generic"}
```

You can now start adding proposals to this project."""

    with patch("api.get_store", return_value=store):
        result = _execute_chat_commands(response)

    assert "Great choice!" in result
    assert "start adding proposals" in result
    assert "Project created" in result


def test_full_domain_config_dict(store):
    """_execute_chat_commands works with a full domain_config dict (no template lookup)."""
    response = '''```command
CREATE_PROJECT {"name": "Custom Experiment", "description": "Optimize enzyme activity", "domain_config": {"name": "Enzyme optimization", "description": "Optimize enzyme activity via mutations.", "target_metric": "activity_score", "optimize": "maximize", "intervention_types": "config_change, probe", "parameters": "mutation_site, temperature, pH"}}
```'''

    with patch("api.get_store", return_value=store):
        result = _execute_chat_commands(response)

    assert "Project created" in result
    assert "Custom Experiment" in result
    projects = store.list_projects()
    assert len(projects) == 1
    cfg = projects[0]["domain_config"]
    assert cfg["name"] == "Enzyme optimization"
    assert cfg["target_metric"] == "activity_score"
    assert cfg["optimize"] == "maximize"


def test_top_level_target_metric_promoted_to_domain_config(store):
    """target_metric and optimize at top level are promoted into domain_config."""
    response = 'CREATE_PROJECT {"name": "Plant Growth", "target_metric": "growth_cm", "optimize": "maximize", "parameters": "light,water"}'

    with patch("api.get_store", return_value=store):
        result = _execute_chat_commands(response)

    assert "Project created" in result
    projects = store.list_projects()
    assert len(projects) == 1
    cfg = projects[0]["domain_config"]
    assert cfg["target_metric"] == "growth_cm"
    assert cfg["optimize"] == "maximize"
    assert cfg["parameters"] == "light,water"


def test_domain_config_target_metric_not_overwritten(store):
    """Top-level target_metric does not overwrite domain_config's own target_metric."""
    response = '''```command
CREATE_PROJECT {"name": "Test", "target_metric": "wrong", "domain_config": {"target_metric": "correct", "parameters": "x"}}
```'''

    with patch("api.get_store", return_value=store):
        _execute_chat_commands(response)

    projects = store.list_projects()
    assert projects[0]["domain_config"]["target_metric"] == "correct"
