"""Research step router for v2.0 agent architecture.

Dispatches research steps proposed by the LLM to the right handler:
- experiment → TrainPyEnvironment (or any Environment)
- analysis   → CodeSandbox (sandbox.py)
- hypothesis → statistical test via sandbox
- schema_change → approval queue (logged, requires human approval)

This replaces the v1.5 pattern of "LLM outputs 3 configs" with
"LLM outputs 3 research steps of any type."
"""

import json
import logging
from dataclasses import dataclass, field

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.environment import Environment
from autoresearcher2.research.sandbox import validate_code, run_in_sandbox

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a research step."""

    step_type: str
    success: bool
    payload: dict  # step-specific results
    summary: str  # one-line human-readable summary


@dataclass
class ResearchStep:
    """A single research step proposed by the LLM."""

    type: str  # "experiment", "analysis", "hypothesis", "schema_change"
    payload: dict
    reasoning: str = ""

    def validate(self, schema: InterventionSchema) -> list[str]:
        """Check step is well-formed. Returns list of errors (empty = valid)."""
        errors = []
        if self.type not in ("experiment", "analysis", "hypothesis", "schema_change"):
            errors.append(f"Unknown step type: {self.type}")
            return errors

        if self.type == "experiment":
            config = self.payload.get("config")
            if not config:
                errors.append("Experiment step requires 'config'")
            elif set(config.keys()) != set(schema.factor_names):
                errors.append(
                    f"Config factors {set(config.keys())} don't match schema {set(schema.factor_names)}"
                )

        elif self.type == "analysis":
            if not self.payload.get("code"):
                errors.append("Analysis step requires 'code'")
            else:
                violations = validate_code(self.payload["code"])
                if violations:
                    errors.extend(violations)

        elif self.type == "hypothesis":
            if not self.payload.get("claim"):
                errors.append("Hypothesis step requires 'claim'")
            if not self.payload.get("proposed_test"):
                errors.append("Hypothesis step requires 'proposed_test'")

        elif self.type == "schema_change":
            if not self.payload.get("changes"):
                errors.append("Schema change step requires 'changes'")

        return errors


class StepRouter:
    """Routes research steps to the appropriate handler."""

    def __init__(
        self,
        schema: InterventionSchema,
        env: Environment,
        ssh_host: str = "root@dllm-experiment.home",
        ssh_key: str = "~/.ssh/pve03_key",
        sandbox_timeout: int = 60,
    ):
        self.schema = schema
        self.env = env
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key
        self.sandbox_timeout = sandbox_timeout
        self.pending_schema_changes: list[dict] = []
        self.analysis_log: list[dict] = []

    def execute(self, step: ResearchStep, history: list[dict]) -> StepResult:
        """Execute a research step and return results."""
        errors = step.validate(self.schema)
        if errors:
            return StepResult(
                step_type=step.type,
                success=False,
                payload={"errors": errors},
                summary=f"Validation failed: {'; '.join(errors)}",
            )

        if step.type == "experiment":
            return self._run_experiment(step)
        elif step.type == "analysis":
            return self._run_analysis(step, history)
        elif step.type == "hypothesis":
            return self._run_hypothesis(step, history)
        elif step.type == "schema_change":
            return self._queue_schema_change(step)
        else:
            return StepResult(
                step_type=step.type,
                success=False,
                payload={},
                summary=f"Unknown step type: {step.type}",
            )

    def _run_experiment(self, step: ResearchStep) -> StepResult:
        """Run an experiment via the Environment."""
        config = step.payload["config"]
        try:
            cell = self.schema.config_to_cell(config)
        except (KeyError, ValueError) as e:
            return StepResult(
                step_type="experiment",
                success=False,
                payload={"error": str(e)},
                summary=f"Invalid config: {e}",
            )

        try:
            outcome = self.env.run(cell)
        except Exception as e:
            return StepResult(
                step_type="experiment",
                success=False,
                payload={"error": str(e), "cell": cell, "config": config},
                summary=f"Experiment failed: {e}",
            )

        return StepResult(
            step_type="experiment",
            success=True,
            payload={
                "cell": cell,
                "config": config,
                "outcome": outcome,
            },
            summary=f"Experiment cell {cell}: outcome={outcome:.4f}",
        )

    def _run_analysis(self, step: ResearchStep, history: list[dict]) -> StepResult:
        """Run analysis code in the sandbox."""
        code = step.payload["code"]
        question = step.payload.get("question", "")
        timeout = step.payload.get("timeout_seconds", self.sandbox_timeout)

        # Inject history as DATA for the analysis code
        data_json = json.dumps(history)

        result = run_in_sandbox(
            code=code,
            ssh_host=self.ssh_host,
            ssh_key=self.ssh_key,
            timeout_seconds=timeout,
            data_json=data_json,
        )

        # Log the analysis
        self.analysis_log.append({
            "question": question,
            "code": code,
            "result": result,
        })

        if result["success"]:
            return StepResult(
                step_type="analysis",
                success=True,
                payload={
                    "question": question,
                    "output": result["output"],
                    "code": code,
                },
                summary=f"Analysis complete: {question[:80]}",
            )
        else:
            return StepResult(
                step_type="analysis",
                success=False,
                payload={
                    "question": question,
                    "error": result["error"],
                    "code": code,
                },
                summary=f"Analysis failed: {result['error'][:80]}",
            )

    def _run_hypothesis(self, step: ResearchStep, history: list[dict]) -> StepResult:
        """Evaluate a hypothesis by running the proposed test in the sandbox."""
        claim = step.payload["claim"]
        test_code = step.payload["proposed_test"]
        threshold = step.payload.get("acceptance_threshold", "p < 0.05")

        # If proposed_test is a description rather than code, wrap it
        if not test_code.strip().startswith(("import", "from", "#", "def", "data")):
            return StepResult(
                step_type="hypothesis",
                success=False,
                payload={"claim": claim, "error": "proposed_test must be executable Python code"},
                summary=f"Hypothesis not testable: proposed_test is not code",
            )

        data_json = json.dumps(history)

        result = run_in_sandbox(
            code=test_code,
            ssh_host=self.ssh_host,
            ssh_key=self.ssh_key,
            timeout_seconds=self.sandbox_timeout,
            data_json=data_json,
        )

        if result["success"]:
            return StepResult(
                step_type="hypothesis",
                success=True,
                payload={
                    "claim": claim,
                    "threshold": threshold,
                    "test_output": result["output"],
                    "code": test_code,
                },
                summary=f"Hypothesis tested: {claim[:60]}",
            )
        else:
            return StepResult(
                step_type="hypothesis",
                success=False,
                payload={
                    "claim": claim,
                    "error": result["error"],
                    "code": test_code,
                },
                summary=f"Hypothesis test failed: {result['error'][:60]}",
            )

    def _queue_schema_change(self, step: ResearchStep) -> StepResult:
        """Queue a schema change for human approval."""
        changes = step.payload["changes"]
        self.pending_schema_changes.append({
            "changes": changes,
            "reasoning": step.reasoning,
            "status": "pending_approval",
        })

        return StepResult(
            step_type="schema_change",
            success=True,
            payload={
                "changes": changes,
                "status": "queued_for_approval",
            },
            summary=f"Schema change queued: {len(changes)} change(s) pending approval",
        )
