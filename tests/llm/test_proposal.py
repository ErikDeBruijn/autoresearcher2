# tests/llm/test_proposal.py
"""Tests for LLM proposal module — parsing and validation only.

These tests do NOT call claude -p. They test the prompt building,
response parsing, and config validation logic.
"""

import json

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.llm.proposal import (
    _build_prompt,
    _extract_json_array,
    _parse_response,
    _valid_config,
)


def make_schema():
    return InterventionSchema(
        factors={
            "DEPTH": ["6", "8", "10"],
            "MATRIX_LR": ["0.02", "0.04", "0.08"],
        }
    )


def test_build_prompt_contains_schema_and_history():
    schema = make_schema()
    history = [
        {
            "config": {"DEPTH": "8", "MATRIX_LR": "0.04"},
            "val_bpb": 1.032,
            "outcome": 0.968,
            "cell_index": 4,
            "appraisal": {"surprise": 0.5, "learntropy": 0.3},
        }
    ]
    prompt = _build_prompt(schema, history, {"DEPTH": 0.15, "MATRIX_LR": 0.08})
    assert "DEPTH" in prompt
    assert "MATRIX_LR" in prompt
    assert "0.968" in prompt or "1.032" in prompt
    assert "9 " in prompt or "Total cells: 9" in prompt


def test_extract_json_array_clean():
    text = '[{"config": {"DEPTH": "8"}, "reasoning": "test"}]'
    result = _extract_json_array(text)
    assert result is not None
    assert len(result) == 1
    assert result[0]["config"]["DEPTH"] == "8"


def test_extract_json_array_with_surrounding_text():
    text = 'Here are my suggestions:\n[{"config": {"DEPTH": "8"}, "reasoning": "test"}]\nHope this helps!'
    result = _extract_json_array(text)
    assert result is not None
    assert len(result) == 1


def test_extract_json_array_no_array():
    assert _extract_json_array("no json here") is None


def test_valid_config_correct():
    schema = make_schema()
    assert _valid_config({"DEPTH": "8", "MATRIX_LR": "0.04"}, schema) is True


def test_valid_config_missing_factor():
    schema = make_schema()
    assert _valid_config({"DEPTH": "8"}, schema) is False


def test_valid_config_invalid_level():
    schema = make_schema()
    assert _valid_config({"DEPTH": "8", "MATRIX_LR": "0.99"}, schema) is False


def test_valid_config_extra_factor():
    schema = make_schema()
    assert _valid_config({"DEPTH": "8", "MATRIX_LR": "0.04", "EXTRA": "x"}, schema) is False


def test_parse_response_with_claude_json_wrapper():
    schema = make_schema()
    # claude --output-format json wraps content in {"result": "..."}
    inner = json.dumps([
        {"config": {"DEPTH": "8", "MATRIX_LR": "0.04"}, "reasoning": "good"},
        {"config": {"DEPTH": "6", "MATRIX_LR": "0.02"}, "reasoning": "explore"},
    ])
    raw = json.dumps({"result": inner})
    result = _parse_response(raw, schema)
    assert len(result) == 2
    assert result[0]["cell"] == schema.config_to_cell({"DEPTH": "8", "MATRIX_LR": "0.04"})
    assert result[0]["reasoning"] == "good"


def test_parse_response_filters_invalid_configs():
    schema = make_schema()
    inner = json.dumps([
        {"config": {"DEPTH": "8", "MATRIX_LR": "0.04"}, "reasoning": "valid"},
        {"config": {"DEPTH": "99", "MATRIX_LR": "0.04"}, "reasoning": "invalid level"},
    ])
    raw = json.dumps({"result": inner})
    result = _parse_response(raw, schema)
    assert len(result) == 1


def test_parse_response_garbage_returns_empty():
    schema = make_schema()
    result = _parse_response("this is not json at all", schema)
    assert result == []
