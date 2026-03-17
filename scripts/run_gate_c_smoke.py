"""Gate C: Minimal Real Substrate Smoke Test

Runs 3 real train.py experiments on dllm-experiment VM with different configs.
Patches knobs via sed, runs train.py, parses val_bpb from output.
No LLM, no fancy orchestration — just: patch, run, parse, log.

Usage:
    uv run python scripts/run_gate_c_smoke.py

Requirements:
    - SSH access: ssh -i ~/.ssh/pve03_key root@dllm-experiment.local
    - Data shards present on VM (~/.cache/autoresearch/data/)
    - GPU available on VM
"""

import json
import subprocess
import time
import re
from pathlib import Path

VM_HOST = "root@dllm-experiment.local"
SSH_KEY = "~/.ssh/pve03_key"
REMOTE_DIR = "~/github.com/karpathy/autoresearch"

# 3 configs to test: vary DEPTH and MATRIX_LR (small schema, fast signal)
CONFIGS = [
    {"DEPTH": 6, "MATRIX_LR": 0.04, "label": "shallow_default_lr"},
    {"DEPTH": 8, "MATRIX_LR": 0.04, "label": "default"},
    {"DEPTH": 8, "MATRIX_LR": 0.08, "label": "default_high_lr"},
]


def ssh_cmd(cmd: str, timeout: int = 600) -> tuple[int, str]:
    """Run command on VM via SSH. Returns (returncode, output)."""
    full_cmd = [
        "ssh", "-i", SSH_KEY, "-o", "ConnectTimeout=10",
        VM_HOST, cmd,
    ]
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def patch_and_run(config: dict) -> dict:
    """Patch train.py knobs, run training, parse results."""
    label = config["label"]
    print(f"\n--- Running: {label} ---")
    print(f"  Config: {config}")

    # Reset train.py to git HEAD first
    rc, out = ssh_cmd(f"cd {REMOTE_DIR} && git checkout train.py")
    if rc != 0:
        return {"label": label, "error": f"git checkout failed: {out}"}

    # Patch each knob via sed
    for knob, value in config.items():
        if knob == "label":
            continue
        # Match line like: DEPTH = 8
        sed_cmd = f"sed -i 's/^{knob} = .*/{knob} = {value}/' {REMOTE_DIR}/train.py"
        rc, out = ssh_cmd(sed_cmd)
        if rc != 0:
            return {"label": label, "error": f"sed failed for {knob}: {out}"}

    # Verify patches
    verify_cmd = " && ".join(
        f"grep '^{k} = ' {REMOTE_DIR}/train.py"
        for k in config if k != "label"
    )
    rc, out = ssh_cmd(verify_cmd)
    print(f"  Patched: {out.strip()}")

    # Run training (use GPU 1 to avoid contention)
    print(f"  Training started...")
    start = time.time()
    rc, out = ssh_cmd(
        f"cd {REMOTE_DIR} && CUDA_VISIBLE_DEVICES=1 uv run train.py 2>&1",
        timeout=900,  # 15 min max
    )
    wall_time = time.time() - start
    print(f"  Finished in {wall_time:.0f}s (exit code {rc})")

    # Parse results
    result = {
        "label": label,
        "config": {k: v for k, v in config.items() if k != "label"},
        "exit_code": rc,
        "wall_time_s": round(wall_time, 1),
    }

    if rc != 0:
        # Check for FAIL
        if "FAIL" in out:
            result["error"] = "Training diverged (NaN/loss>100)"
        else:
            result["error"] = f"Non-zero exit: {out[-500:]}"
        print(f"  ERROR: {result['error']}")
        return result

    # Parse val_bpb and training_seconds from output
    val_bpb_match = re.search(r"val_bpb:\s+([\d.]+)", out)
    train_time_match = re.search(r"training_seconds:\s+([\d.]+)", out)

    if val_bpb_match:
        result["val_bpb"] = float(val_bpb_match.group(1))
        print(f"  val_bpb: {result['val_bpb']:.6f}")
    else:
        result["error"] = "Could not parse val_bpb"
        print(f"  ERROR: could not parse val_bpb from output")
        # Save last 500 chars for debugging
        result["output_tail"] = out[-500:]

    if train_time_match:
        result["training_seconds"] = float(train_time_match.group(1))

    return result


def main():
    start = time.time()
    print("=" * 60)
    print("GATE C: Real Substrate Smoke Test")
    print("=" * 60)

    # Verify SSH connectivity
    rc, out = ssh_cmd("echo OK", timeout=15)
    if rc != 0 or "OK" not in out:
        print(f"FAIL: Cannot reach VM. Output: {out}")
        return

    # Verify GPU
    rc, out = ssh_cmd("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    print(f"GPU: {out.strip()}")

    results = []
    for config in CONFIGS:
        result = patch_and_run(config)
        results.append(result)

    # Reset train.py after all runs
    ssh_cmd(f"cd {REMOTE_DIR} && git checkout train.py")

    # Save results
    artifacts_dir = Path("artifacts/v2_gates")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "gate": "C",
        "name": "real_substrate_smoke",
        "runs": results,
    }

    # Determine pass/fail
    successful = [r for r in results if "val_bpb" in r]
    output["n_successful"] = len(successful)
    output["n_total"] = len(results)
    output["passes"] = len(successful) >= 2  # at least 2 of 3 must produce val_bpb

    if successful:
        output["val_bpb_range"] = {
            "min": min(r["val_bpb"] for r in successful),
            "max": max(r["val_bpb"] for r in successful),
        }
        # Check: do different configs produce different val_bpb?
        if len(successful) >= 2:
            bpbs = [r["val_bpb"] for r in successful]
            output["configs_differ"] = max(bpbs) - min(bpbs) > 0.001

    output_path = artifacts_dir / "gate_c_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("GATE C SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        status = f"val_bpb={r['val_bpb']:.6f}" if "val_bpb" in r else f"ERROR: {r.get('error', '?')}"
        print(f"  {r['label']}: {status} ({r.get('wall_time_s', '?')}s)")

    print(f"\n  Successful: {output['n_successful']}/{output['n_total']}")
    print(f"  GATE C: {'PASS' if output['passes'] else 'FAIL'}")
    print(f"\nResults: {output_path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
