"""Minimal v2.0 test runner — pure LLM-driven experiments on GPU 0.

This is a test harness to validate the v2.0 architecture before building
the full research step router. It runs a short experiment loop where the
LLM proposes all experiments (no Bayesian model).
"""

import argparse
import json
import logging
import subprocess
import textwrap
import time
import uuid

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__).info

SCHEMA = InterventionSchema(factors={
    "DEPTH": ["6", "8", "10"],
    "MATRIX_LR": ["0.02", "0.04", "0.08"],
    "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
})

RUN_ID = uuid.uuid4().hex[:8]


def build_prompt(history: list[dict], n_suggestions: int = 3) -> str:
    schema_lines = []
    for name, levels in SCHEMA.factors.items():
        schema_lines.append(f"  {name}: {levels}")
    schema_desc = "\n".join(schema_lines)

    if history:
        table_lines = ["cell | config | val_bpb | tokens_M | tok/s"]
        table_lines.append("---- | ------ | ------- | -------- | -----")
        for rec in history:
            config = rec["config"]
            config_str = ", ".join(f"{k}={v}" for k, v in config.items())
            val_bpb = f"{rec['val_bpb']:.6f}" if rec.get("val_bpb") else "FAILED"
            tokens_m = f"{rec['tokens_M']:.0f}" if rec.get("tokens_M") else "?"
            tok_s = f"{rec['tok_per_sec']:.0f}" if rec.get("tok_per_sec") else "?"
            table_lines.append(f"{rec['cell']} | {config_str} | {val_bpb} | {tokens_m} | {tok_s}")
        results_section = "\n".join(table_lines)
    else:
        results_section = "(No experiments run yet. You are choosing the first batch.)"

    return textwrap.dedent(f"""\
        You are the sole experiment advisor for an automated ML research system.
        There is no Bayesian model — you make all decisions about what to try next.

        Your goal: find the configuration that minimizes val_bpb (validation bits-per-byte)
        for a small GPT language model.

        SCHEMA (factors and levels):
        {schema_desc}

        Total cells: {SCHEMA.n_cells}

        IMPORTANT: All experiments run for ~5 minutes wall-clock time. Deeper models
        process fewer tokens in that time. DEPTH=6 trains ~323M tokens, DEPTH=8 ~176M,
        DEPTH=10 ~101M. Consider this when interpreting results.

        RESULTS SO FAR:
        {results_section}

        Suggest exactly {n_suggestions} experiment configurations to run next.
        Balance exploration (untried regions) with exploitation (refining good regions).

        Respond with ONLY a JSON array of {n_suggestions} objects, each with:
        - "config": dict mapping factor names to level values (as strings)
        - "reasoning": one sentence explaining why

        JSON array:""")


def call_llm(prompt: str, ssh_host: str, ssh_key: str) -> list[dict]:
    ssh_cmd = [
        "ssh", "-i", ssh_key,
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
        ssh_host,
        "claude -p --output-format json 2>/dev/null",
    ]
    result = subprocess.run(ssh_cmd, input=prompt, capture_output=True,
                            text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"LLM call failed (rc={result.returncode}): {result.stderr[:200]}")

    # Parse response
    try:
        outer = json.loads(result.stdout)
        text = outer.get("result", result.stdout)
    except (json.JSONDecodeError, AttributeError):
        text = result.stdout

    # Extract JSON array
    start = text.find("[")
    if start == -1:
        raise RuntimeError(f"No JSON array in LLM response: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                suggestions = json.loads(text[start:i + 1])
                break
    else:
        raise RuntimeError("Malformed JSON array in LLM response")

    # Validate configs
    valid = []
    for s in suggestions:
        config = s.get("config", {})
        if set(config.keys()) != set(SCHEMA.factor_names):
            continue
        if all(config[k] in SCHEMA.factors[k] for k in SCHEMA.factor_names):
            cell = SCHEMA.config_to_cell(config)
            valid.append({"cell": cell, "config": config, "reasoning": s.get("reasoning", "")})
    return valid


def run_experiment(env: TrainPyEnvironment, cell: int, config: dict) -> dict:
    start = time.time()
    try:
        outcome = env.run(cell)
        val_bpb = 2.0 - outcome
        wall_time = time.time() - start
        metadata = getattr(env, "last_run_metadata", {})
        tokens_M = metadata.get("total_tokens_M")
        tok_per_sec = round(tokens_M * 1e6 / wall_time, 1) if tokens_M and wall_time > 0 else None
        return {
            "cell": cell, "config": config, "outcome": outcome, "val_bpb": val_bpb,
            "wall_time_s": round(wall_time, 1), "tokens_M": tokens_M,
            "tok_per_sec": tok_per_sec, "error": None,
        }
    except Exception as e:
        wall_time = time.time() - start
        # Retry once
        log(f"    attempt 1 failed: {str(e)[:100]}... retrying")
        time.sleep(5)
        try:
            start2 = time.time()
            outcome = env.run(cell)
            val_bpb = 2.0 - outcome
            wall_time2 = time.time() - start2
            metadata = getattr(env, "last_run_metadata", {})
            tokens_M = metadata.get("total_tokens_M")
            tok_per_sec = round(tokens_M * 1e6 / wall_time2, 1) if tokens_M and wall_time2 > 0 else None
            log(f"    succeeded on retry (intermittent failure)")
            return {
                "cell": cell, "config": config, "outcome": outcome, "val_bpb": val_bpb,
                "wall_time_s": round(wall_time + wall_time2, 1), "tokens_M": tokens_M,
                "tok_per_sec": tok_per_sec, "error": None, "retries": 1,
            }
        except Exception as e2:
            return {
                "cell": cell, "config": config, "outcome": None, "val_bpb": None,
                "wall_time_s": round(time.time() - start, 1), "tokens_M": None,
                "tok_per_sec": None, "error": str(e2), "retries": 1,
                "failure_type": "reproducible" if str(e) == str(e2) else "different_errors",
            }


def main():
    parser = argparse.ArgumentParser(description="v2.0 test runner — pure LLM on GPU 0")
    parser.add_argument("--n-experiments", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--ssh-host", default="root@dllm-experiment.home")
    parser.add_argument("--ssh-key", default="~/.ssh/pve03_key")
    args = parser.parse_args()

    env = TrainPyEnvironment(
        schema=SCHEMA,
        ssh_host=args.ssh_host,
        ssh_key=args.ssh_key,
        remote_dir="~/github.com/karpathy/autoresearch-v2",
        cuda_device="0",
    )

    out_file = f"artifacts/v2-test/run_{RUN_ID}.json"
    import os
    os.makedirs("artifacts/v2-test", exist_ok=True)

    history = []
    experiment_count = 0

    log(f"=== v2.0 Test Run {RUN_ID} ===")
    log(f"Budget: {args.n_experiments} experiments, batch size {args.batch_size}")
    log(f"GPU: 0, remote dir: autoresearch-v2")

    while experiment_count < args.n_experiments:
        remaining = args.n_experiments - experiment_count
        batch_size = min(args.batch_size, remaining)

        log(f"--- LLM consultation ({experiment_count}/{args.n_experiments} done) ---")
        prompt = build_prompt(history, n_suggestions=batch_size)

        try:
            suggestions = call_llm(prompt, args.ssh_host, args.ssh_key)
        except Exception as e:
            log(f"LLM failed: {e}")
            break

        if not suggestions:
            log("LLM returned no valid suggestions, stopping")
            break

        for s in suggestions:
            if experiment_count >= args.n_experiments:
                break
            cell, config = s["cell"], s["config"]
            log(f"  [{experiment_count+1}/{args.n_experiments}] cell {cell} {config}")
            log(f"    reasoning: {s['reasoning']}")

            result = run_experiment(env, cell, config)
            result["source"] = "llm_v2"
            result["experiment"] = experiment_count
            result["reasoning"] = s["reasoning"]

            if result["error"] is None:
                log(f"    val_bpb={result['val_bpb']:.6f} tokens={result['tokens_M']}M")
            else:
                log(f"    FAILED: {result['error'][:100]}")

            history.append(result)
            experiment_count += 1

            # Save after each experiment
            with open(out_file, "w") as f:
                json.dump({"run_id": RUN_ID, "version": "2.0-test",
                           "experiments": history}, f, indent=2)

    # Summary
    successful = [r for r in history if r.get("val_bpb")]
    log(f"=== Run complete: {len(successful)}/{len(history)} successful ===")
    if successful:
        best = min(successful, key=lambda r: r["val_bpb"])
        log(f"Best val_bpb: {best['val_bpb']:.6f} — {best['config']}")


if __name__ == "__main__":
    main()
