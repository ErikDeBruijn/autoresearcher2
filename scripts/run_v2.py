"""v2.0 Research Agent — cumulative code changes to train.py.

The LLM sees the current train.py, proposes a code change (as a unified diff),
the change is applied, training runs, and if val_bpb improves the change is
kept. If it worsens, train.py is reverted to the previous best.

This is the Karpathy autoresearch model with epistemic governance.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import textwrap
import time
import uuid

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__).info

RUN_ID = uuid.uuid4().hex[:8]

DEFAULT_SSH_HOST = "root@dllm-experiment.local"
DEFAULT_SSH_KEY = "~/.ssh/pve03_key"
DEFAULT_REMOTE_DIR = "~/github.com/karpathy/autoresearch-v2"
DEFAULT_CUDA_DEVICE = "0"


def ssh(cmd: str, ssh_host: str, ssh_key: str, timeout: int = 30,
        stdin: str | None = None) -> tuple[int, str]:
    result = subprocess.run(
        ["ssh", "-i", ssh_key,
         "-o", "ConnectTimeout=10",
         "-o", "ServerAliveInterval=15",
         "-o", "ServerAliveCountMax=2",
         ssh_host, cmd],
        input=stdin, capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def get_train_py(ssh_host: str, ssh_key: str, remote_dir: str) -> str:
    """Read current train.py from the remote."""
    rc, content = ssh(f"cat {remote_dir}/train.py", ssh_host, ssh_key, timeout=15)
    if rc != 0:
        raise RuntimeError(f"Failed to read train.py: {content}")
    return content


def apply_diff(diff_text: str, ssh_host: str, ssh_key: str, remote_dir: str) -> tuple[bool, str]:
    """Apply a unified diff to train.py on the remote. Returns (success, message)."""
    # Write diff to a temp file and apply with patch
    diff_id = uuid.uuid4().hex[:8]
    diff_file = f"/tmp/patch_{diff_id}.diff"

    rc, out = ssh(f"cat > {diff_file}", ssh_host, ssh_key, stdin=diff_text)
    if rc != 0:
        return False, f"Failed to write diff: {out}"

    rc, out = ssh(
        f"cd {remote_dir} && patch -p0 --forward < {diff_file} 2>&1; echo EXIT:$?",
        ssh_host, ssh_key, timeout=15
    )
    ssh(f"rm -f {diff_file}", ssh_host, ssh_key)

    if "EXIT:0" in out:
        return True, out
    else:
        # Revert any partial patch
        ssh(f"cd {remote_dir} && git checkout train.py", ssh_host, ssh_key)
        return False, f"Patch failed: {out}"


def apply_full_file(new_content: str, ssh_host: str, ssh_key: str, remote_dir: str) -> tuple[bool, str]:
    """Write a complete new train.py to the remote."""
    # Verify it's valid Python first
    try:
        compile(new_content, "train.py", "exec")
    except SyntaxError as e:
        return False, f"Syntax error in proposed train.py: {e}"

    rc, out = ssh(f"cat > {remote_dir}/train.py", ssh_host, ssh_key, stdin=new_content)
    if rc != 0:
        return False, f"Failed to write train.py: {out}"
    return True, "File written successfully"


def run_training(ssh_host: str, ssh_key: str, remote_dir: str,
                 cuda_device: str, poll_interval: int = 15,
                 max_wait: int = 600) -> tuple[float | None, str, dict]:
    """Run train.py and return (val_bpb, full_output, metadata)."""
    job_id = uuid.uuid4().hex[:8]
    out_file = f"/tmp/train_{job_id}.out"
    done_file = f"/tmp/train_{job_id}.done"

    ssh(
        f"nohup bash -c '"
        f"cd {remote_dir} && "
        f"CUDA_VISIBLE_DEVICES={cuda_device} uv run train.py "
        f"> {out_file} 2>&1; echo $? > {done_file}"
        f"' &>/dev/null &",
        ssh_host, ssh_key,
    )

    start = time.time()
    while time.time() - start < max_wait:
        time.sleep(poll_interval)
        try:
            rc, check = ssh(f"cat {done_file} 2>/dev/null", ssh_host, ssh_key)
            if rc == 0 and check.strip():
                break
        except (subprocess.TimeoutExpired, Exception):
            continue
    else:
        ssh(f"pkill -f 'train.py' 2>/dev/null; rm -f {out_file} {done_file}",
            ssh_host, ssh_key)
        return None, "Timed out", {}

    train_rc = int(check.strip())
    _, out = ssh(f"cat {out_file}", ssh_host, ssh_key, timeout=30)
    ssh(f"rm -f {out_file} {done_file}", ssh_host, ssh_key)

    if train_rc != 0:
        return None, f"train.py failed (exit {train_rc}): {out[-500:]}", {}

    # Parse val_bpb
    match = re.search(r"val_bpb:\s+([\d.]+)", out)
    if not match:
        return None, f"Could not parse val_bpb from output", {}

    val_bpb = float(match.group(1))

    # Parse metadata
    meta = {}
    for key, pattern in {
        "total_tokens_M": r"total_tokens_M:\s+([\d.]+)",
        "num_steps": r"num_steps:\s+(\d+)",
        "num_params_M": r"num_params_M:\s+([\d.]+)",
        "mfu_percent": r"mfu_percent:\s+([\d.]+)",
        "training_seconds": r"training_seconds:\s+([\d.]+)",
    }.items():
        m = re.search(pattern, out)
        if m:
            meta[key] = float(m.group(1))

    return val_bpb, out, meta


def ask_llm_for_change(train_py: str, history: list[dict],
                       ssh_host: str, ssh_key: str) -> dict | None:
    """Ask the LLM to propose a change to train.py.

    Returns dict with 'change_type' ('diff' or 'full_file'), 'content', 'reasoning'.
    """
    history_text = ""
    if history:
        lines = ["Previous iterations (most recent last):"]
        for h in history[-10:]:  # last 10 to keep prompt manageable
            status = f"val_bpb={h['val_bpb']:.6f}" if h.get('val_bpb') else f"FAILED: {h.get('error', '?')[:80]}"
            kept = "KEPT" if h.get("kept") else "REVERTED"
            lines.append(f"  #{h['iteration']}: {status} [{kept}] — {h.get('reasoning', '?')}")
        history_text = "\n".join(lines)

    prompt = textwrap.dedent(f"""\
        You are an ML research agent optimizing a GPT language model training script.
        Your goal: minimize val_bpb (validation bits-per-byte). Lower is better.

        The training script runs for a fixed time budget (~5 minutes). After training,
        it evaluates on a validation set and reports val_bpb.

        CURRENT train.py:
        ```python
        {train_py}
        ```

        {history_text}

        RULES:
        - Propose ONE change to train.py that you believe will lower val_bpb
        - The change can be anything: hyperparameters, architecture, optimizer, schedule, etc.
        - Keep changes focused — one idea per iteration so we can attribute improvements
        - The script must remain a valid single-file Python program
        - Do not remove the final eval/printing section (lines starting with "val_bpb:")
        - Do not change the data loading (prepare.py is fixed)

        RESPOND with a JSON object containing:
        - "reasoning": why you think this change will help (1-2 sentences)
        - "change_description": short label for what you're changing
        - "full_file": the complete new train.py content (as a string)

        Respond with ONLY the JSON object, no other text.

        JSON:""")

    ssh_cmd = [
        "ssh", "-i", ssh_key,
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
        ssh_host,
        "claude -p --output-format json 2>/dev/null",
    ]

    result = subprocess.run(ssh_cmd, input=prompt, capture_output=True,
                            text=True, timeout=300)
    if result.returncode != 0:
        log(f"LLM call failed (rc={result.returncode})")
        return None

    # Parse response
    try:
        outer = json.loads(result.stdout)
        text = outer.get("result", result.stdout)
    except (json.JSONDecodeError, AttributeError):
        text = result.stdout

    # Extract JSON object
    try:
        # Find first { ... } block
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    proposal = json.loads(text[start:i + 1])
                    return proposal
    except (json.JSONDecodeError, Exception) as e:
        log(f"Failed to parse LLM response: {e}")
        return None

    return None


def main():
    parser = argparse.ArgumentParser(description="v2.0 — cumulative LLM-driven code changes")
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--cuda-device", default=DEFAULT_CUDA_DEVICE)
    args = parser.parse_args()

    os.makedirs("artifacts/v2", exist_ok=True)
    out_file = f"artifacts/v2/run_{RUN_ID}.json"

    log(f"=== v2.0 Run {RUN_ID} ===")
    log(f"Iterations: {args.n_iterations}, GPU: {args.cuda_device}")
    log(f"Remote: {args.remote_dir}")

    # Run baseline first (unmodified train.py)
    log("--- Baseline run (unmodified train.py) ---")
    ssh(f"cd {args.remote_dir} && git checkout train.py", args.ssh_host, args.ssh_key)
    baseline_bpb, baseline_out, baseline_meta = run_training(
        args.ssh_host, args.ssh_key, args.remote_dir, args.cuda_device)

    if baseline_bpb is None:
        log(f"Baseline failed: {baseline_out[:200]}")
        return

    log(f"Baseline val_bpb: {baseline_bpb:.6f}")
    best_bpb = baseline_bpb

    history = [{
        "iteration": 0,
        "val_bpb": baseline_bpb,
        "metadata": baseline_meta,
        "change": "baseline (unmodified)",
        "reasoning": "baseline run",
        "kept": True,
        "error": None,
    }]

    for iteration in range(1, args.n_iterations + 1):
        log(f"--- Iteration {iteration}/{args.n_iterations} (best so far: {best_bpb:.6f}) ---")

        # Get current train.py
        current_train_py = get_train_py(args.ssh_host, args.ssh_key, args.remote_dir)

        # Ask LLM for a change
        log("  Consulting LLM...")
        proposal = ask_llm_for_change(current_train_py, history,
                                       args.ssh_host, args.ssh_key)
        if not proposal:
            log("  LLM returned no proposal, skipping iteration")
            history.append({
                "iteration": iteration, "val_bpb": None, "metadata": {},
                "change": "LLM failed", "reasoning": "", "kept": False,
                "error": "LLM returned no proposal",
            })
            continue

        reasoning = proposal.get("reasoning", "?")
        change_desc = proposal.get("change_description", "unknown change")
        log(f"  Proposal: {change_desc}")
        log(f"  Reasoning: {reasoning}")

        # Apply the change
        new_content = proposal.get("full_file", "")
        if not new_content:
            log("  No full_file in proposal, skipping")
            history.append({
                "iteration": iteration, "val_bpb": None, "metadata": {},
                "change": change_desc, "reasoning": reasoning, "kept": False,
                "error": "No full_file in proposal",
            })
            continue

        success, msg = apply_full_file(new_content, args.ssh_host, args.ssh_key, args.remote_dir)
        if not success:
            log(f"  Failed to apply change: {msg}")
            history.append({
                "iteration": iteration, "val_bpb": None, "metadata": {},
                "change": change_desc, "reasoning": reasoning, "kept": False,
                "error": msg,
            })
            continue

        # Run training
        log("  Running training...")
        val_bpb, train_out, meta = run_training(
            args.ssh_host, args.ssh_key, args.remote_dir, args.cuda_device)

        if val_bpb is None:
            log(f"  Training failed: {train_out[:200]}")
            # Revert
            ssh(f"cd {args.remote_dir} && git checkout train.py", args.ssh_host, args.ssh_key)
            # Re-apply best version if we had changes
            if len(history) > 1:
                last_kept = [h for h in history if h.get("kept") and h.get("train_py")]
                if last_kept:
                    apply_full_file(last_kept[-1]["train_py"],
                                    args.ssh_host, args.ssh_key, args.remote_dir)
            history.append({
                "iteration": iteration, "val_bpb": None, "metadata": meta,
                "change": change_desc, "reasoning": reasoning, "kept": False,
                "error": train_out[:300],
            })
            continue

        # Compare with best
        improved = val_bpb < best_bpb
        delta = best_bpb - val_bpb

        if improved:
            log(f"  val_bpb={val_bpb:.6f} — IMPROVED by {delta:.6f}! Keeping change.")
            best_bpb = val_bpb
            # Save the working train.py content
            kept_train_py = get_train_py(args.ssh_host, args.ssh_key, args.remote_dir)
        else:
            log(f"  val_bpb={val_bpb:.6f} — worse by {-delta:.6f}. Reverting.")
            # Revert to git baseline
            ssh(f"cd {args.remote_dir} && git checkout train.py", args.ssh_host, args.ssh_key)
            # Re-apply best version
            last_kept = [h for h in history if h.get("kept") and h.get("train_py")]
            if last_kept:
                apply_full_file(last_kept[-1]["train_py"],
                                args.ssh_host, args.ssh_key, args.remote_dir)
            kept_train_py = None

        history.append({
            "iteration": iteration,
            "val_bpb": val_bpb,
            "metadata": meta,
            "change": change_desc,
            "reasoning": reasoning,
            "kept": improved,
            "delta": delta,
            "error": None,
            "train_py": kept_train_py,  # only stored for kept changes
        })

        # Save progress (without full train.py content to keep file manageable)
        save_history = []
        for h in history:
            save_h = {k: v for k, v in h.items() if k != "train_py"}
            save_h["has_train_py"] = h.get("train_py") is not None
            save_history.append(save_h)

        with open(out_file, "w") as f:
            json.dump({"run_id": RUN_ID, "version": "2.0",
                        "best_val_bpb": best_bpb, "iterations": save_history}, f, indent=2)

    # Summary
    log(f"=== Run complete ===")
    log(f"Baseline: {history[0]['val_bpb']:.6f}")
    log(f"Best:     {best_bpb:.6f}")
    kept_count = sum(1 for h in history[1:] if h.get("kept"))
    log(f"Kept {kept_count}/{args.n_iterations} changes")

    # Reset remote to git state
    ssh(f"cd {args.remote_dir} && git checkout train.py", args.ssh_host, args.ssh_key)


if __name__ == "__main__":
    main()
