"""v2.0 LLM-based research step proposal.

Replaces the v1.5 pattern of "suggest 3 configs" with "suggest 3 research steps."
The LLM can now propose experiments, analyses, hypotheses, or schema changes.

Uses the same SSH+claude pattern as proposal.py, but with an expanded prompt.
"""

import json
import logging
import subprocess
import textwrap

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.router import ResearchStep

logger = logging.getLogger(__name__)


def propose_research_steps(
    schema: InterventionSchema,
    history: list[dict],
    factor_importances: dict[str, float],
    analysis_results: list[dict] | None = None,
    prior_results: list[dict] | None = None,
    ssh_host: str = "root@dllm-experiment.home",
    ssh_key: str = "~/.ssh/pve03_key",
) -> list[ResearchStep]:
    """Ask the LLM to propose research steps.

    Returns list of ResearchStep objects (may be empty on failure).
    """
    if not history:
        return []

    prompt = _build_v2_prompt(
        schema, history, factor_importances, analysis_results, prior_results
    )

    try:
        raw = _call_claude(prompt, ssh_host, ssh_key)
    except Exception:
        logger.warning("LLM call failed", exc_info=True)
        return []

    return _parse_steps(raw, schema)


def _build_v2_prompt(
    schema: InterventionSchema,
    history: list[dict],
    factor_importances: dict[str, float],
    analysis_results: list[dict] | None = None,
    prior_results: list[dict] | None = None,
) -> str:
    schema_desc = "\n".join(
        f"  {name}: {levels}" for name, levels in schema.factors.items()
    )

    # Results table
    table_lines = ["exp | config | outcome | tokens_M | tok/s | surprise | learntropy"]
    table_lines.append("--- | ------ | ------- | -------- | ----- | -------- | ----------")
    for i, rec in enumerate(history):
        config = rec.get("config", {})
        outcome = rec.get("outcome", "?")
        appraisal = rec.get("appraisal", {})
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        tokens_m = rec.get("tokens_M", "?")
        tok_per_sec = rec.get("tok_per_sec", "?")
        surprise = appraisal.get("surprise", "?")
        learntropy = appraisal.get("learntropy", "?")

        fmt = lambda v, d=4: f"{v:.{d}f}" if isinstance(v, float) else str(v)
        table_lines.append(
            f"{i} | {config_str} | {fmt(outcome)} | {fmt(tokens_m, 0)} | "
            f"{fmt(tok_per_sec, 0)} | {fmt(surprise, 3)} | {fmt(learntropy, 3)}"
        )
    results_table = "\n".join(table_lines)

    # Factor importances
    if factor_importances:
        imp_lines = [f"  {n}: {v:.3f}" for n, v in sorted(factor_importances.items(), key=lambda x: -x[1])]
        importance_desc = "\n".join(imp_lines)
    else:
        importance_desc = "  (not yet computed)"

    # Previous analysis results
    analysis_desc = ""
    if analysis_results:
        analysis_desc = "PREVIOUS ANALYSIS RESULTS:\n"
        for ar in analysis_results[-5:]:  # last 5
            q = ar.get("question", "")
            output = ar.get("result", {}).get("output", "")
            if isinstance(output, dict):
                output = json.dumps(output, indent=2)
            analysis_desc += f"  Q: {q}\n  A: {str(output)[:300]}\n\n"

    return textwrap.dedent(f"""\
        You are a research agent for autoresearcher2, a Bayesian experiment optimization system.
        You have access to four types of research actions:

        1. EXPERIMENT — run a specific configuration and measure the outcome
        2. ANALYSIS — write and execute Python code to analyze existing results
           (numpy, scipy, pandas, statsmodels, sklearn available; DATA variable has history)
        3. HYPOTHESIS — state a testable claim with executable test code
        4. SCHEMA_CHANGE — propose modifying the factor/level grid (requires approval)

        EPISTEMIC GOVERNANCE:
        - Distinguish observation from interpretation
        - Claims require evidence. Label confidence: observed / supported / plausible / speculative
        - Prefer honest uncertainty over polished narrative
        - When proposing analyses, be explicit about what you hope to learn and why

        SCHEMA (factors and levels):
        {schema_desc}
        Total cells: {schema.n_cells}

        TOKEN BUDGET CONFOUND:
        Deeper models process fewer tokens in the same wall-clock time. DEPTH=6 → ~323M tokens,
        DEPTH=8 → ~176M, DEPTH=10 → ~101M. Consider this when interpreting results.

        RESULTS SO FAR:
        {results_table}

        FACTOR IMPORTANCES:
        {importance_desc}

        {analysis_desc}

        TASK: Propose exactly 3 research steps. You may mix types freely.

        Guidelines:
        - If you have < 5 experiments, prefer experiments to build data
        - If you have 5+ experiments, consider an analysis step to understand patterns
        - Use hypothesis steps to make your reasoning explicit and testable
        - Schema changes should be rare and well-justified
        - Analysis code receives history as DATA (list of dicts with config, outcome, etc.)
        - Analysis code must print JSON to stdout

        Respond with ONLY a JSON array of 3 objects:
        ```json
        [
          {{
            "type": "experiment" | "analysis" | "hypothesis" | "schema_change",
            "payload": {{ ... }},
            "reasoning": "one sentence"
          }},
          ...
        ]
        ```

        For experiments: payload = {{"config": {{"FACTOR": "level", ...}}}}
        For analyses: payload = {{"question": "...", "code": "..."}}
        For hypotheses: payload = {{"claim": "...", "proposed_test": "<python code>", "acceptance_threshold": "..."}}
        For schema changes: payload = {{"changes": [{{"factor": "...", "operation": "...", ...}}]}}

        JSON array:""")


def _call_claude(prompt: str, ssh_host: str, ssh_key: str) -> str:
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
        timeout=180,  # longer timeout for richer responses
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"SSH/claude failed (rc={result.returncode}): {result.stderr[:200]}"
        )

    return result.stdout


def _parse_steps(raw: str, schema: InterventionSchema) -> list[ResearchStep]:
    """Parse LLM response into validated ResearchStep objects."""
    try:
        outer = json.loads(raw)
        text = outer.get("result", raw)
    except (json.JSONDecodeError, AttributeError):
        text = raw

    steps_data = _extract_json_array(text)
    if not steps_data:
        logger.warning("Could not parse research steps from LLM response")
        return []

    steps = []
    for s in steps_data:
        step_type = s.get("type", "")
        payload = s.get("payload", {})
        reasoning = s.get("reasoning", "")

        step = ResearchStep(type=step_type, payload=payload, reasoning=reasoning)
        errors = step.validate(schema)
        if errors:
            logger.debug("Skipping invalid step: %s — %s", step_type, errors)
            continue

        steps.append(step)

    return steps[:3]


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
