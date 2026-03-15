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

from api import _execute_chat_commands, DOMAIN_CONFIGS
from autoresearcher2.v3.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "research.db")
    s.init()
    yield s
    s.close()


def test_no_command_passthrough():
    """Response without commands passes through unchanged."""
    text = "The current val_bpb is 1.05, which is great progress!"
    with patch("api.get_store") as mock:
        result = _execute_chat_commands(text)
    assert result == text
    mock.assert_not_called()


def test_create_project_with_code_fence(store):
    """CREATE_PROJECT in code fence is detected and executed."""
    response = """I'll create an Atari project for you.

```command
CREATE_PROJECT {"name": "Atari Breakout", "description": "Train RL agents on Breakout", "domain_type": "atari-rl", "parameters": "game,learning_rate,network_size"}
```

This will set up the project with RL-focused parameters."""

    with patch("api.get_store", return_value=store):
        result = _execute_chat_commands(response)

    assert "Project created" in result
    assert "Atari Breakout" in result
    assert "```command" not in result  # Command block replaced

    # Verify project was actually created in store
    projects = store.list_projects()
    assert len(projects) == 1
    assert projects[0]["name"] == "Atari Breakout"
    assert projects[0]["description"] == "Train RL agents on Breakout"
    cfg = projects[0]["domain_config"]
    assert cfg["parameters"] == "game,learning_rate,network_size"


def test_create_project_without_code_fence(store):
    """CREATE_PROJECT without code fence is also detected."""
    response = 'Here we go: CREATE_PROJECT {"name": "Quick Test", "domain_type": "generic"}'

    with patch("api.get_store", return_value=store):
        result = _execute_chat_commands(response)

    assert "Project created" in result
    projects = store.list_projects()
    assert len(projects) == 1
    assert projects[0]["name"] == "Quick Test"


def test_domain_config_applied(store):
    """Domain type maps to correct DomainConfig."""
    response = '```command\nCREATE_PROJECT {"name": "GPT Test", "domain_type": "nanogpt"}\n```'

    with patch("api.get_store", return_value=store):
        _execute_chat_commands(response)

    projects = store.list_projects()
    cfg = projects[0]["domain_config"]
    assert "NanoGPT" in cfg["name"]
    assert "config_change" in cfg["intervention_types"]


def test_custom_parameters_override(store):
    """Custom parameters override domain defaults."""
    response = '```command\nCREATE_PROJECT {"name": "Custom", "domain_type": "nanogpt", "parameters": "lr,depth,width"}\n```'

    with patch("api.get_store", return_value=store):
        _execute_chat_commands(response)

    projects = store.list_projects()
    cfg = projects[0]["domain_config"]
    assert cfg["parameters"] == "lr,depth,width"


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


def test_domain_configs_all_have_code_change():
    """All domain configs include code_change as an intervention type."""
    for name, cfg in DOMAIN_CONFIGS.items():
        assert "code_change" in cfg["intervention_types"], f"{name} missing code_change"
