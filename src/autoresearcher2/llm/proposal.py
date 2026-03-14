"""LLM-based experiment proposal via claude -p on a remote VM.

The LLM is a suggestion engine, not the decision maker. The controller
still makes final decisions via Thompson sampling. If the LLM fails or
returns garbage, we return an empty list and the caller falls back.
"""

import json
import logging
import subprocess
import textwrap

from autoresearcher2.core.schema import InterventionSchema

logger = logging.getLogger(__name__)


def propose_experiments(
    schema: InterventionSchema,
    history: list[dict],  # [{config, val_bpb, outcome, appraisal, tokens_M, tok_per_sec}, ...]
    factor_importances: dict[str, float],
    prior_results: list[dict] | None = None,  # results from earlier approaches
    ssh_host: str = "root@dllm-experiment.home",
    ssh_key: str = "~/.ssh/pve03_key",
) -> list[dict]:
    """Ask Claude to analyze results and suggest next experiments.

    Returns list of {cell: int, config: dict, reasoning: str}.
    Returns empty list if LLM fails (caller falls back to Thompson sampling).
    """
    if not history:
        return []

    prompt = _build_prompt(schema, history, factor_importances, prior_results)

    try:
        raw = _call_claude(prompt, ssh_host, ssh_key)
    except Exception:
        logger.warning("LLM call failed, falling back to Thompson sampling", exc_info=True)
        return []

    return _parse_response(raw, schema)


def _build_prompt(
    schema: InterventionSchema,
    history: list[dict],
    factor_importances: dict[str, float],
    prior_results: list[dict] | None = None,
) -> str:
    # Schema description
    schema_lines = []
    for name, levels in schema.factors.items():
        schema_lines.append(f"  {name}: {levels}")
    schema_desc = "\n".join(schema_lines)

    # Results table — now includes tokens_M and tok_per_sec
    table_lines = ["cell | config | outcome | tokens_M | tok/s | surprise | learntropy"]
    table_lines.append("---- | ------ | ------- | -------- | ----- | -------- | ----------")
    for rec in history:
        config = rec.get("config", {})
        outcome = rec.get("outcome", rec.get("val_bpb", "?"))
        appraisal = rec.get("appraisal", {})
        surprise = appraisal.get("surprise", "?")
        learntropy = appraisal.get("learntropy", "?")
        cell = rec.get("cell_index", "?")
        tokens_m = rec.get("tokens_M", "?")
        tok_per_sec = rec.get("tok_per_sec", "?")

        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        if isinstance(outcome, float):
            outcome = f"{outcome:.4f}"
        if isinstance(surprise, float):
            surprise = f"{surprise:.3f}"
        if isinstance(learntropy, float):
            learntropy = f"{learntropy:.3f}"
        if isinstance(tokens_m, float):
            tokens_m = f"{tokens_m:.0f}"
        if isinstance(tok_per_sec, float):
            tok_per_sec = f"{tok_per_sec:.0f}"

        table_lines.append(
            f"{cell} | {config_str} | {outcome} | {tokens_m} | {tok_per_sec} | {surprise} | {learntropy}"
        )
    results_table = "\n".join(table_lines)

    # Factor importances
    importance_lines = []
    for name, imp in sorted(factor_importances.items(), key=lambda x: -x[1]):
        importance_lines.append(f"  {name}: {imp:.3f}")
    importance_desc = "\n".join(importance_lines) if importance_lines else "  (not yet computed)"

    # High-signal experiments (belief-changing results the model learned most from)
    high_signal_desc = _format_high_signal_experiments(history)

    # Coverage gaps (factor levels never or rarely tested)
    coverage_desc = _format_coverage_gaps(schema, history)

    # Prior approach results (what other methods already found)
    prior_desc = _format_prior_results(prior_results) if prior_results else ""

    # Epistemic context from project governance documents
    epistemic_context = _epistemic_context()

    return textwrap.dedent(f"""\
        You are an experiment advisor for an automated Bayesian research system
        (autoresearcher2). The system maintains a structured model of factor effects
        and updates beliefs after each experiment. Your role: suggest experiments
        that the Bayesian model would not choose on its own — especially exploring
        untested regions and following up on surprising results.

        {epistemic_context}

        SCHEMA (factors and their levels):
        {schema_desc}

        Total cells: {schema.n_cells}

        IMPORTANT CONFOUND — TOKEN BUDGET VARIES BY CONFIG:
        All experiments run for the same wall-clock time (~5 minutes), but deeper
        models process fewer tokens. DEPTH=6 trains ~323M tokens, DEPTH=8 ~176M,
        DEPTH=10 ~101M. Lower val_bpb for shallower models may partly reflect more
        training data, not inherently better architecture. The tokens_M and tok/s
        columns below let you assess this.

        RESULTS SO FAR (this approach):
        {results_table}

        FACTOR IMPORTANCES (higher = more influential on outcome):
        {importance_desc}

        {high_signal_desc}

        {coverage_desc}

        {prior_desc}

        TASK: Suggest exactly 3 experiment configurations to run next.

        Guidelines:
        - At least 1 suggestion should explore an UNTRIED factor level or region
        - At least 1 should exploit or refine the best-performing region
        - If high-signal experiments exist, consider follow-ups that test whether
          the surprising finding generalizes
        - Avoid exact configs already run (see results table)
        - Consider the token budget confound when interpreting results
        - If prior approach results are available, use them as additional signal
          but do not simply repeat their best configs — look for gaps in what they explored

        Respond with ONLY a JSON array of 3 objects, each with:
        - "config": dict mapping factor names to level values
        - "reasoning": one sentence explaining why

        JSON array:""")


def _format_high_signal_experiments(history: list[dict]) -> str:
    """Extract experiments where the model learned the most."""
    scored = []
    for rec in history:
        appraisal = rec.get("appraisal", {})
        learntropy = appraisal.get("learntropy", 0)
        theory_conflict = appraisal.get("theory_conflict", 0)
        score = max(learntropy, theory_conflict)
        if score > 0.001:
            scored.append((score, rec))

    if not scored:
        return "HIGH-SIGNAL EXPERIMENTS: None yet (model has not been surprised)."

    scored.sort(key=lambda x: -x[0])
    lines = ["HIGH-SIGNAL EXPERIMENTS (results that changed the model's beliefs):"]
    for score, rec in scored[:3]:
        config = rec.get("config", {})
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        appraisal = rec.get("appraisal", {})
        outcome = rec.get("outcome", "?")
        if isinstance(outcome, float):
            outcome = f"{outcome:.4f}"
        surprise = appraisal.get("surprise", 0)
        learntropy = appraisal.get("learntropy", 0)
        breadth = appraisal.get("prediction_impact_breadth", 0)
        lines.append(
            f"  {config_str} → outcome={outcome} "
            f"(surprise={surprise:.3f}, learntropy={learntropy:.3f}, "
            f"predictions changed: {int(breadth)} cells)"
        )

    return "\n".join(lines)


def _format_coverage_gaps(schema: InterventionSchema, history: list[dict]) -> str:
    """Identify factor levels that have never or rarely been tested."""
    # Count how often each factor level appears
    level_counts: dict[str, dict[str, int]] = {}
    for name in schema.factor_names:
        level_counts[name] = {level: 0 for level in schema.factors[name]}

    for rec in history:
        config = rec.get("config", {})
        for name, value in config.items():
            if name in level_counts and value in level_counts[name]:
                level_counts[name][value] += 1

    # Find untried and underexplored levels
    untried = []
    underexplored = []
    n_experiments = len(history)
    for name, counts in level_counts.items():
        n_levels = len(counts)
        expected = n_experiments / n_levels if n_levels > 0 else 0
        for level, count in counts.items():
            if count == 0:
                untried.append(f"  {name}={level}: NEVER TESTED")
            elif expected > 0 and count < expected * 0.3:
                underexplored.append(f"  {name}={level}: only {count}× (expected ~{expected:.0f})")

    if not untried and not underexplored:
        return "COVERAGE: All factor levels have been tested."

    lines = ["COVERAGE GAPS (factor levels with insufficient data):"]
    lines.extend(untried)
    lines.extend(underexplored)
    return "\n".join(lines)


def _format_prior_results(prior_results: list[dict]) -> str:
    """Summarize results from earlier approaches (random, bayesian, etc.)."""
    if not prior_results:
        return ""

    lines = [
        "PRIOR APPROACH RESULTS (from earlier methods on the same schema):",
        "These runs already completed. Use them as additional signal — they show",
        "which regions of the space have been explored and what was found.",
        "",
    ]
    for prior in prior_results:
        name = prior.get("approach", "unknown")
        results = prior.get("results", [])
        successful = [r for r in results if r.get("error") is None]
        if not successful:
            lines.append(f"  {name}: no successful experiments")
            continue

        best = min(r["val_bpb"] for r in successful)
        mean = sum(r["val_bpb"] for r in successful) / len(successful)
        unique_cells = len(set(r.get("cell", r.get("cell_index", -1)) for r in successful))

        # Top 3 configs
        top3 = sorted(successful, key=lambda r: r["val_bpb"])[:3]
        top3_str = "; ".join(
            f"{','.join(f'{k}={v}' for k,v in r['config'].items())}={r['val_bpb']:.4f}"
            for r in top3
        )

        lines.append(
            f"  {name}: {len(successful)} experiments, {unique_cells} unique cells, "
            f"best={best:.6f}, mean={mean:.6f}"
        )
        lines.append(f"    Top 3: {top3_str}")

    return "\n".join(lines)


def _epistemic_context() -> str:
    """Condensed epistemic governance context from CONSTITUTION and CHARTER."""
    return textwrap.dedent("""\
        EPISTEMIC GOVERNANCE (from project Constitution and Charter):
        - Distinguish observation from interpretation. Do not convert excitement into evidence.
        - Claims require concrete evidence. Label confidence: observed / supported / plausible / speculative.
        - Prefer honest uncertainty over polished narrative. Say what you don't know.
        - Watch for confirmation bias, overgeneralization from thin evidence, and proxy gaming.
        - Exploration and surprise are valuable, but must remain accountable to reality.
        - The goal is not just better results, but better judgment about what the results mean.
        - When reasoning about experiment suggestions, be explicit about your uncertainty.""")


def _call_claude(prompt: str, ssh_host: str, ssh_key: str) -> str:
    """Call claude -p on the remote VM via SSH."""
    # We pass the prompt via stdin to avoid shell escaping issues
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


def _parse_response(raw: str, schema: InterventionSchema) -> list[dict]:
    """Parse LLM JSON response into validated suggestions."""
    try:
        # claude --output-format json wraps the response in a JSON object
        outer = json.loads(raw)
        # The actual text content is in the "result" field
        text = outer.get("result", raw)
    except (json.JSONDecodeError, AttributeError):
        text = raw

    # Extract JSON array from the text (might have surrounding prose)
    suggestions = _extract_json_array(text)
    if not suggestions:
        logger.warning("Could not parse LLM suggestions from response")
        return []

    # Validate and enrich each suggestion
    valid = []
    for s in suggestions:
        config = s.get("config", {})
        reasoning = s.get("reasoning", "")

        # Validate config against schema
        if not _valid_config(config, schema):
            logger.debug("Skipping invalid config: %s", config)
            continue

        cell = schema.config_to_cell(config)
        valid.append({
            "cell": cell,
            "config": config,
            "reasoning": reasoning,
        })

    return valid[:3]


def _extract_json_array(text: str) -> list[dict] | None:
    """Find and parse a JSON array from text that might contain other content."""
    # Try parsing the whole thing first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find the first [ ... ] in the text
    start = text.find("[")
    if start == -1:
        return None

    # Find matching closing bracket
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
    """Check that config has exactly the right factors with valid levels."""
    if set(config.keys()) != set(schema.factor_names):
        return False
    for name, levels in schema.factors.items():
        if config.get(name) not in levels:
            return False
    return True
