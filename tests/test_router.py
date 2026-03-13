"""Tests for the v2.0 research step router."""

from unittest.mock import MagicMock, patch

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.router import ResearchStep, StepRouter, StepResult


SCHEMA = InterventionSchema(factors={
    "DEPTH": ["6", "8", "10"],
    "MATRIX_LR": ["0.02", "0.04", "0.10"],
    "WEIGHT_DECAY": ["0.05", "0.20", "0.50"],
})


class TestResearchStep:
    def test_valid_experiment(self):
        step = ResearchStep(
            type="experiment",
            payload={"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
        )
        assert step.validate(SCHEMA) == []

    def test_experiment_missing_config(self):
        step = ResearchStep(type="experiment", payload={})
        errors = step.validate(SCHEMA)
        assert len(errors) == 1
        assert "config" in errors[0].lower()

    def test_experiment_wrong_factors(self):
        step = ResearchStep(
            type="experiment",
            payload={"config": {"DEPTH": "8", "LR": "0.04"}},
        )
        errors = step.validate(SCHEMA)
        assert len(errors) == 1
        assert "factors" in errors[0].lower() or "don't match" in errors[0].lower()

    def test_valid_analysis(self):
        step = ResearchStep(
            type="analysis",
            payload={
                "question": "What is the mean outcome?",
                "code": "import numpy as np\nprint(np.mean(DATA))",
            },
        )
        assert step.validate(SCHEMA) == []

    def test_analysis_missing_code(self):
        step = ResearchStep(type="analysis", payload={"question": "test"})
        errors = step.validate(SCHEMA)
        assert any("code" in e.lower() for e in errors)

    def test_analysis_blocked_import(self):
        step = ResearchStep(
            type="analysis",
            payload={"code": "import subprocess\nsubprocess.run(['ls'])"},
        )
        errors = step.validate(SCHEMA)
        assert any("subprocess" in e for e in errors)

    def test_valid_hypothesis(self):
        step = ResearchStep(
            type="hypothesis",
            payload={
                "claim": "DEPTH is the most important factor",
                "proposed_test": "import json\nprint(json.dumps({'result': True}))",
            },
        )
        assert step.validate(SCHEMA) == []

    def test_hypothesis_missing_claim(self):
        step = ResearchStep(
            type="hypothesis",
            payload={"proposed_test": "import json\nprint('ok')"},
        )
        errors = step.validate(SCHEMA)
        assert any("claim" in e.lower() for e in errors)

    def test_hypothesis_missing_test(self):
        step = ResearchStep(
            type="hypothesis",
            payload={"claim": "test claim"},
        )
        errors = step.validate(SCHEMA)
        assert any("proposed_test" in e.lower() for e in errors)

    def test_valid_schema_change(self):
        step = ResearchStep(
            type="schema_change",
            payload={"changes": [{"factor": "MATRIX_LR", "operation": "refine_levels"}]},
        )
        assert step.validate(SCHEMA) == []

    def test_schema_change_missing_changes(self):
        step = ResearchStep(type="schema_change", payload={})
        errors = step.validate(SCHEMA)
        assert any("changes" in e.lower() for e in errors)

    def test_unknown_type(self):
        step = ResearchStep(type="magic", payload={})
        errors = step.validate(SCHEMA)
        assert any("unknown" in e.lower() for e in errors)


class TestStepRouter:
    def setup_method(self):
        self.env = MagicMock()
        self.router = StepRouter(
            schema=SCHEMA,
            env=self.env,
            ssh_host="test@host",
            ssh_key="/tmp/key",
        )

    def test_experiment_success(self):
        self.env.run.return_value = 0.9664  # 2.0 - 1.0336
        step = ResearchStep(
            type="experiment",
            payload={"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
        )
        result = self.router.execute(step, [])
        assert result.success
        assert result.step_type == "experiment"
        assert result.payload["outcome"] == 0.9664
        self.env.run.assert_called_once()

    def test_experiment_env_failure(self):
        self.env.run.side_effect = RuntimeError("SSH failed")
        step = ResearchStep(
            type="experiment",
            payload={"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
        )
        result = self.router.execute(step, [])
        assert not result.success
        assert "SSH failed" in result.payload["error"]

    def test_validation_failure_blocks_execution(self):
        step = ResearchStep(type="experiment", payload={})
        result = self.router.execute(step, [])
        assert not result.success
        assert "errors" in result.payload
        self.env.run.assert_not_called()

    @patch("autoresearcher2.research.router.run_in_sandbox")
    def test_analysis_success(self, mock_sandbox):
        mock_sandbox.return_value = {
            "success": True,
            "output": {"mean": 0.95},
            "stdout": '{"mean": 0.95}',
            "stderr": "",
        }
        step = ResearchStep(
            type="analysis",
            payload={
                "question": "What is the mean?",
                "code": "import json\nprint(json.dumps({'mean': 0.95}))",
            },
        )
        result = self.router.execute(step, [{"outcome": 0.95}])
        assert result.success
        assert result.payload["output"] == {"mean": 0.95}
        assert len(self.router.analysis_log) == 1

    @patch("autoresearcher2.research.router.run_in_sandbox")
    def test_analysis_failure(self, mock_sandbox):
        mock_sandbox.return_value = {
            "success": False,
            "error": "Exit code 1: NameError",
            "stdout": "",
            "stderr": "NameError",
        }
        step = ResearchStep(
            type="analysis",
            payload={"code": "import numpy\nprint(undefined_var)"},
        )
        result = self.router.execute(step, [])
        assert not result.success

    @patch("autoresearcher2.research.router.run_in_sandbox")
    def test_hypothesis_with_code(self, mock_sandbox):
        mock_sandbox.return_value = {
            "success": True,
            "output": {"p_value": 0.03, "reject_null": True},
            "stdout": '{"p_value": 0.03, "reject_null": true}',
            "stderr": "",
        }
        step = ResearchStep(
            type="hypothesis",
            payload={
                "claim": "DEPTH matters most",
                "proposed_test": "import json\nprint(json.dumps({'p_value': 0.03, 'reject_null': True}))",
                "acceptance_threshold": "p < 0.05",
            },
        )
        result = self.router.execute(step, [{"outcome": 0.95}])
        assert result.success
        assert result.payload["test_output"]["p_value"] == 0.03

    def test_hypothesis_without_code(self):
        step = ResearchStep(
            type="hypothesis",
            payload={
                "claim": "DEPTH matters",
                "proposed_test": "Run ANOVA on DEPTH vs outcome",
            },
        )
        result = self.router.execute(step, [])
        assert not result.success
        assert "not code" in result.summary

    def test_schema_change_queued(self):
        step = ResearchStep(
            type="schema_change",
            payload={"changes": [{"factor": "MATRIX_LR", "operation": "add_level", "value": "0.06"}]},
            reasoning="Fine-grained LR exploration around optimum",
        )
        result = self.router.execute(step, [])
        assert result.success
        assert result.payload["status"] == "queued_for_approval"
        assert len(self.router.pending_schema_changes) == 1
        assert self.router.pending_schema_changes[0]["status"] == "pending_approval"

    def test_step_result_dataclass(self):
        r = StepResult(step_type="test", success=True, payload={"k": "v"}, summary="ok")
        assert r.step_type == "test"
        assert r.success
        assert r.payload == {"k": "v"}
        assert r.summary == "ok"
