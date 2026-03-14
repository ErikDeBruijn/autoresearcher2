"""Tests for v2.0 research step proposal parsing."""

import json

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.llm.research_proposal import _parse_steps, _build_v2_prompt


SCHEMA = InterventionSchema(factors={
    "DEPTH": ["6", "8", "10"],
    "MATRIX_LR": ["0.02", "0.04", "0.10"],
    "WEIGHT_DECAY": ["0.05", "0.20", "0.50"],
})


class TestParseSteps:
    def test_parse_experiment_step(self):
        raw = json.dumps([{
            "type": "experiment",
            "payload": {"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
            "reasoning": "test",
        }])
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 1
        assert steps[0].type == "experiment"

    def test_parse_analysis_step(self):
        raw = json.dumps([{
            "type": "analysis",
            "payload": {
                "question": "Mean outcome?",
                "code": "import json\nprint(json.dumps({'mean': 0.5}))",
            },
            "reasoning": "test",
        }])
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 1
        assert steps[0].type == "analysis"

    def test_parse_mixed_steps(self):
        raw = json.dumps([
            {
                "type": "experiment",
                "payload": {"config": {"DEPTH": "6", "MATRIX_LR": "0.02", "WEIGHT_DECAY": "0.05"}},
                "reasoning": "explore",
            },
            {
                "type": "analysis",
                "payload": {"question": "test", "code": "import json\nprint('ok')"},
                "reasoning": "analyze",
            },
            {
                "type": "hypothesis",
                "payload": {
                    "claim": "DEPTH matters",
                    "proposed_test": "import json\nprint(json.dumps({'p': 0.01}))",
                },
                "reasoning": "test claim",
            },
        ])
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 3
        types = [s.type for s in steps]
        assert "experiment" in types
        assert "analysis" in types
        assert "hypothesis" in types

    def test_parse_claude_json_wrapper(self):
        """claude --output-format json wraps response in {"result": "..."}."""
        inner = json.dumps([{
            "type": "experiment",
            "payload": {"config": {"DEPTH": "10", "MATRIX_LR": "0.10", "WEIGHT_DECAY": "0.50"}},
            "reasoning": "test",
        }])
        raw = json.dumps({"result": inner})
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 1

    def test_skip_invalid_step(self):
        raw = json.dumps([
            {"type": "experiment", "payload": {}, "reasoning": "bad"},  # missing config
            {
                "type": "experiment",
                "payload": {"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
                "reasoning": "good",
            },
        ])
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 1
        assert steps[0].reasoning == "good"

    def test_skip_blocked_analysis(self):
        raw = json.dumps([{
            "type": "analysis",
            "payload": {"code": "import os\nos.system('rm -rf /')"},
            "reasoning": "evil",
        }])
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 0

    def test_max_three_steps(self):
        raw = json.dumps([
            {
                "type": "experiment",
                "payload": {"config": {"DEPTH": str(d), "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
                "reasoning": f"step {i}",
            }
            for i, d in enumerate(["6", "8", "10", "6"])
        ])
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 3

    def test_empty_response(self):
        assert _parse_steps("", SCHEMA) == []
        assert _parse_steps("no json here", SCHEMA) == []

    def test_json_with_surrounding_prose(self):
        raw = 'Here are my suggestions:\n' + json.dumps([{
            "type": "experiment",
            "payload": {"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
            "reasoning": "test",
        }]) + '\nHope this helps!'
        steps = _parse_steps(raw, SCHEMA)
        assert len(steps) == 1


class TestBuildPrompt:
    def test_prompt_contains_schema(self):
        history = [{"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}, "outcome": 0.96}]
        prompt = _build_v2_prompt(SCHEMA, history, {"DEPTH": 0.5, "MATRIX_LR": 0.3})
        assert "DEPTH" in prompt
        assert "MATRIX_LR" in prompt
        assert "experiment" in prompt.lower()
        assert "analysis" in prompt.lower()
        assert "hypothesis" in prompt.lower()

    def test_prompt_includes_analysis_results(self):
        history = [{"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}, "outcome": 0.96}]
        analysis = [{"question": "What is the mean?", "result": {"output": {"mean": 0.96}}}]
        prompt = _build_v2_prompt(SCHEMA, history, {}, analysis_results=analysis)
        assert "What is the mean?" in prompt

    def test_prompt_mentions_data_variable(self):
        history = [{"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}, "outcome": 0.96}]
        prompt = _build_v2_prompt(SCHEMA, history, {})
        assert "DATA" in prompt
