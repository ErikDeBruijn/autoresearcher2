"""Tests for the v2.0 research loop."""

from unittest.mock import MagicMock, patch

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.core.research_loop_v2 import ResearchLoopV2
from autoresearcher2.research.router import ResearchStep


SCHEMA = InterventionSchema(factors={
    "DEPTH": ["6", "8", "10"],
    "MATRIX_LR": ["0.02", "0.04", "0.10"],
    "WEIGHT_DECAY": ["0.05", "0.20", "0.50"],
})


def make_loop():
    model = BayesianLinearModel(SCHEMA)
    controller = Controller(SCHEMA, model, preferred_outcome=1.0, seed=42)
    memory = MemoryStore()
    env = MagicMock()
    env.run.return_value = 0.9664

    loop = ResearchLoopV2(
        schema=SCHEMA,
        model=model,
        controller=controller,
        memory=memory,
        env=env,
    )
    return loop


class TestResearchLoopV2:
    @patch("autoresearcher2.core.research_loop_v2.propose_research_steps")
    def test_experiment_step_updates_model(self, mock_propose):
        loop = make_loop()
        mock_propose.return_value = [
            ResearchStep(
                type="experiment",
                payload={"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
                reasoning="test",
            )
        ]
        results = loop.run(1)
        assert len(results) == 1
        assert results[0]["outcome"] == 0.9664
        assert len(loop.history) == 1
        assert len(loop.memory._records) == 1

    @patch("autoresearcher2.core.research_loop_v2.propose_research_steps")
    @patch("autoresearcher2.research.router.run_in_sandbox")
    def test_analysis_step_no_model_update(self, mock_sandbox, mock_propose):
        loop = make_loop()
        mock_sandbox.return_value = {
            "success": True, "output": {"mean": 0.95}, "stdout": "", "stderr": ""
        }
        mock_propose.return_value = [
            ResearchStep(
                type="analysis",
                payload={"question": "mean?", "code": "import json\nprint(json.dumps({'mean': 0.95}))"},
                reasoning="test",
            )
        ]
        results = loop.run(1)
        assert len(results) == 0  # no experiment results
        assert len(loop.history) == 0  # model not updated
        assert len(loop.step_log) == 1  # but step was logged
        assert loop.step_log[0]["step"]["type"] == "analysis"

    @patch("autoresearcher2.core.research_loop_v2.propose_research_steps")
    def test_fallback_on_llm_failure(self, mock_propose):
        loop = make_loop()
        mock_propose.return_value = []  # LLM failed
        results = loop.run(1)
        assert len(results) == 1  # Thompson sampling fallback
        assert results[0]["outcome"] == 0.9664

    @patch("autoresearcher2.core.research_loop_v2.propose_research_steps")
    def test_mixed_steps(self, mock_propose):
        loop = make_loop()
        mock_propose.return_value = [
            ResearchStep(
                type="experiment",
                payload={"config": {"DEPTH": "6", "MATRIX_LR": "0.02", "WEIGHT_DECAY": "0.05"}},
                reasoning="explore",
            ),
            ResearchStep(
                type="schema_change",
                payload={"changes": [{"factor": "DEPTH", "operation": "add_level"}]},
                reasoning="expand",
            ),
        ]
        results = loop.run(1)
        assert len(results) == 1  # only the experiment
        assert len(loop.step_log) == 2  # both steps logged
        assert len(loop.router.pending_schema_changes) == 1

    @patch("autoresearcher2.core.research_loop_v2.propose_research_steps")
    def test_multiple_iterations(self, mock_propose):
        loop = make_loop()
        call_count = [0]

        def dynamic_propose(*args, **kwargs):
            call_count[0] += 1
            return [
                ResearchStep(
                    type="experiment",
                    payload={"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
                    reasoning=f"iteration {call_count[0]}",
                )
            ]

        mock_propose.side_effect = dynamic_propose
        results = loop.run(3)
        assert len(results) == 3
        assert len(loop.history) == 3
        assert call_count[0] == 3

    def test_compute_importances_empty(self):
        loop = make_loop()
        assert loop._compute_importances() == {}

    @patch("autoresearcher2.core.research_loop_v2.propose_research_steps")
    def test_compute_importances_after_data(self, mock_propose):
        loop = make_loop()
        mock_propose.return_value = [
            ResearchStep(
                type="experiment",
                payload={"config": {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.20"}},
                reasoning="test",
            )
        ]
        loop.run(1)
        importances = loop._compute_importances()
        assert "DEPTH" in importances
        assert "MATRIX_LR" in importances
        assert "WEIGHT_DECAY" in importances
