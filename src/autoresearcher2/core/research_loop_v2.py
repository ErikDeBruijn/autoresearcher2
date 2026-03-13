"""v2.0 Research loop with research step dispatch.

Replaces the v1.5 loop's fixed cycle (select cell → run → appraise → store)
with a flexible cycle (propose steps → route → collect results → appraise → store).

The LLM can now propose experiments, analyses, hypotheses, or schema changes.
Only experiment steps update the Bayesian model. Analysis and hypothesis steps
enrich the LLM's context for better experiment proposals.
"""

import logging

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.environment import Environment
from autoresearcher2.research.router import StepRouter, ResearchStep, StepResult
from autoresearcher2.llm.research_proposal import propose_research_steps
from autoresearcher2.appraisal.signals import compute_appraisal

logger = logging.getLogger(__name__)


class ResearchLoopV2:
    """v2.0 research loop: propose steps → route → appraise → repeat.

    Each iteration:
    1. LLM proposes 3 research steps (any mix of types)
    2. Router dispatches each step to the right handler
    3. Experiment results update the Bayesian model + get appraised
    4. All results feed back into the next LLM consultation
    """

    def __init__(
        self,
        schema: InterventionSchema,
        model: BayesianLinearModel,
        controller: Controller,
        memory: MemoryStore,
        env: Environment,
        ssh_host: str = "root@dllm-experiment.home",
        ssh_key: str = "~/.ssh/pve03_key",
    ):
        self.schema = schema
        self.model = model
        self.controller = controller
        self.memory = memory
        self.env = env
        self.router = StepRouter(
            schema=schema, env=env, ssh_host=ssh_host, ssh_key=ssh_key
        )
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key

        self.history: list[dict] = []
        self.step_log: list[dict] = []

    def run(self, n_iterations: int) -> list[dict]:
        """Run n iterations of the research loop.

        Each iteration may produce 0-3 experiment results depending on the
        step mix the LLM proposes.
        """
        all_results = []

        for iteration in range(n_iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, n_iterations)

            # Get factor importances from the model
            importances = self._compute_importances()

            # Ask LLM for research steps
            steps = propose_research_steps(
                schema=self.schema,
                history=self.history,
                factor_importances=importances,
                analysis_results=self.router.analysis_log,
                ssh_host=self.ssh_host,
                ssh_key=self.ssh_key,
            )

            if not steps:
                # LLM failed — fall back to Thompson sampling
                logger.warning("LLM returned no steps, falling back to Thompson sampling")
                cell = self.controller.select_next()
                steps = [ResearchStep(
                    type="experiment",
                    payload={"config": self.schema.cell_to_config(cell)},
                    reasoning="Fallback: Thompson sampling (LLM unavailable)",
                )]

            # Execute each step
            for step in steps:
                result = self.router.execute(step, self.history)
                self.step_log.append({
                    "iteration": iteration,
                    "step": {"type": step.type, "payload": step.payload, "reasoning": step.reasoning},
                    "result": {"success": result.success, "summary": result.summary, "payload": result.payload},
                })

                logger.info("  %s: %s", step.type, result.summary)

                # Only experiment results update the model and memory
                if step.type == "experiment" and result.success:
                    experiment_result = self._process_experiment(result)
                    all_results.append(experiment_result)

        return all_results

    def _process_experiment(self, result: StepResult) -> dict:
        """Update Bayesian model and memory with experiment result."""
        cell = result.payload["cell"]
        config = result.payload["config"]
        outcome = result.payload["outcome"]

        snapshot_before = self.model.snapshot()
        self.model.update(cell, outcome)
        snapshot_after = self.model.snapshot()

        appraisal = compute_appraisal(
            self.schema, cell, outcome, snapshot_before, snapshot_after
        )

        self.memory.add(
            cell_index=cell,
            config=config,
            outcome=outcome,
            appraisal=appraisal,
        )

        record = {
            "cell": cell,
            "config": config,
            "outcome": outcome,
            "appraisal": appraisal,
        }
        self.history.append(record)

        return record

    def _compute_importances(self) -> dict[str, float]:
        """Compute factor importances from the Bayesian model weights."""
        if not self.history:
            return {}

        importances = {}
        for i, name in enumerate(self.schema.factor_names):
            n_levels = len(self.schema.factors[name])
            # Sum absolute weight contributions for this factor's one-hot features
            start = sum(len(self.schema.factors[n]) for n in self.schema.factor_names[:i])
            end = start + n_levels
            if end <= len(self.model.mu_w):
                weights = self.model.mu_w[start:end]
                importances[name] = float(max(weights) - min(weights))

        return importances
