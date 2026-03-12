"""Autoresearch-style baseline: LLM with flat results log, no Bayesian model.

This mirrors Karpathy's autoresearch approach for fair comparison.
The LLM sees only: schema + flat results table (config → val_bpb).
No factor importances, no appraisal signals, no coverage gap analysis,
no epistemic/pragmatic scoring. Just "here's what happened, what next?"
"""

import json
import logging
import subprocess
import textwrap

import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.eval.baselines import BaselineAgent

logger = logging.getLogger(__name__)


class AutoresearchLLMAgent(BaselineAgent):
    """LLM-only experiment selection with flat results log."""

    def __init__(
        self,
        schema: InterventionSchema,
        seed: int | None = None,
        ssh_host: str = "root@dllm-experiment.home",
        ssh_key: str = "~/.ssh/pve03_key",
    ):
        super().__init__(schema)
        self.rng = np.random.default_rng(seed)
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key
        self.history: list[dict] = []
        self._pending_suggestions: list[int] = []

    def select_next(self) -> int:
        # Use pending suggestions from previous LLM call
        if self._pending_suggestions:
            return self._pending_suggestions.pop(0)

        # No history yet — pick randomly
        if not self.history:
            return int(self.rng.integers(0, self.schema.n_cells))

        # Ask LLM
        prompt = build_flat_prompt(self.schema, self.history)
        try:
            raw = _call_claude_flat(prompt, self.ssh_host, self.ssh_key)
            suggestions = _parse_flat_response(raw, self.schema)
            if suggestions:
                cell = suggestions[0]
                self._pending_suggestions = suggestions[1:]
                return cell
        except Exception:
            logger.warning("LLM call failed, falling back to random", exc_info=True)

        # Fallback: random
        return int(self.rng.integers(0, self.schema.n_cells))

    def observe(self, cell_index: int, outcome: float) -> None:
        config = self.schema.cell_to_config(cell_index)
        val_bpb = 2.0 - outcome
        self.history.append({
            "config": config,
            "val_bpb": val_bpb,
            "outcome": outcome,
            "cell": cell_index,
        })


def build_flat_prompt(schema: InterventionSchema, history: list[dict]) -> str:
    """Build a prompt with only schema + flat results. No structured signals."""
    schema_lines = []
    for name, levels in schema.factors.items():
        schema_lines.append(f"  {name}: {levels}")
    schema_desc = "\n".join(schema_lines)

    if not history:
        results_section = "No experiments run yet."
    else:
        table_lines = ["config | val_bpb"]
        table_lines.append("------ | ------")
        for rec in sorted(history, key=lambda r: r.get("val_bpb", 99)):
            config = rec.get("config", {})
            val_bpb = rec.get("val_bpb", "?")
            config_str = ", ".join(f"{k}={v}" for k, v in config.items())
            if isinstance(val_bpb, float):
                val_bpb = f"{val_bpb:.4f}"
            table_lines.append(f"{config_str} | {val_bpb}")
        results_section = "\n".join(table_lines)

    return textwrap.dedent(f"""\
        You are optimizing a GPT training pipeline. You can adjust these knobs:

        KNOBS:
        {schema_desc}

        Total configurations: {schema.n_cells}

        RESULTS SO FAR (sorted by val_bpb, lower is better):
        {results_section}

        TASK: Suggest exactly 3 configurations to try next.
        Pick configs that you think will achieve the lowest val_bpb.
        Avoid exact configs already tried.

        Respond with ONLY a JSON array of 3 objects, each with:
        - "config": dict mapping knob names to values
        - "reasoning": one sentence explaining why

        JSON array:""")


def _call_claude_flat(prompt: str, ssh_host: str, ssh_key: str) -> str:
    """Call claude -p on the remote VM via SSH."""
    ssh_cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        ssh_host,
        "claude -p --output-format json 2>/dev/null",
    ]
    result = subprocess.run(
        ssh_cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH/claude failed (rc={result.returncode}): {result.stderr[:200]}"
        )
    return result.stdout


def _parse_flat_response(raw: str, schema: InterventionSchema) -> list[int]:
    """Parse LLM response into a list of cell indices."""
    try:
        outer = json.loads(raw)
        text = outer.get("result", raw)
    except (json.JSONDecodeError, AttributeError):
        text = raw

    suggestions = _extract_json_array(text)
    if not suggestions:
        return []

    cells = []
    for s in suggestions:
        config = s.get("config", {})
        if _valid_config(config, schema):
            cells.append(schema.config_to_cell(config))

    return cells[:3]


def _extract_json_array(text: str) -> list[dict] | None:
    """Find and parse a JSON array from text."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _valid_config(config: dict, schema: InterventionSchema) -> bool:
    """Check config has correct factors with valid levels."""
    if set(config.keys()) != set(schema.factor_names):
        return False
    for name, levels in schema.factors.items():
        if config.get(name) not in levels:
            return False
    return True
