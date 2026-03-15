"""Execution functions for different domains.

Each executor takes a Proposal and returns a result dict with:
- metrics: dict of outcome measurements
- compute_cost: float (optional)
- raw_log: str (optional, for audit)

These are passed as execute_fn to Worker.
"""
import logging
import re
import subprocess
import time

from autoresearcher2.v3.proposal import Proposal

logger = logging.getLogger(__name__)


def make_trainpy_executor(
    ssh_host: str = "root@dllm-experiment.home",
    ssh_key: str = "~/.ssh/pve03_key",
    remote_dir: str = "~/github.com/karpathy/autoresearch",
    cuda_device: str = "1",
    timeout: int = 900,
    local: bool = False,
):
    """Create an executor for NanoGPT train.py experiments.

    Handles config_change (patches train.py knobs) and probe (limited steps).
    Set local=True when running on the VM itself to skip SSH.
    """
    def run_cmd(cmd: str) -> tuple[int, str]:
        if local:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )
        else:
            result = subprocess.run(
                ["ssh", "-i", ssh_key, "-o", "ConnectTimeout=10", ssh_host, cmd],
                capture_output=True, text=True, timeout=timeout,
            )
        return result.returncode, result.stdout + result.stderr

    # Expand ~ for local execution
    train_dir = remote_dir.replace("~", "/root") if local else remote_dir

    def execute(proposal: Proposal) -> dict:
        spec = proposal.intervention_spec
        itype = proposal.intervention_type

        if itype not in ("config_change", "probe"):
            logger.warning("trainpy executor: unsupported type %s, dry-run", itype)
            return {"metrics": {"unsupported": True}, "raw_log": f"dry-run for {itype}"}

        # Reset train.py
        run_cmd(f"cd {train_dir} && git checkout train.py")

        # Patch knobs from intervention_spec
        for knob, value in spec.items():
            if knob == "run_steps":
                continue  # Not a train.py knob
            run_cmd(f"sed -i 's/^{knob} = .*/{knob} = {value}/' {train_dir}/train.py")

        # For probes, limit steps
        if itype == "probe" and "run_steps" in spec:
            run_steps = spec["run_steps"]
            run_cmd(f"sed -i 's/^num_steps = .*/num_steps = {run_steps}/' {train_dir}/train.py")

        # Run training
        start = time.time()
        rc, out = run_cmd(
            f"cd {train_dir} && CUDA_VISIBLE_DEVICES={cuda_device} uv run train.py 2>&1"
        )
        wall_time = time.time() - start

        # Reset
        run_cmd(f"cd {train_dir} && git checkout train.py")

        if rc != 0:
            raise RuntimeError(f"train.py failed (exit {rc}): {out[-500:]}")

        # Parse metrics
        metrics = {}
        val_bpb_match = re.search(r"val_bpb:\s+([\d.]+)", out)
        if val_bpb_match:
            metrics["val_bpb"] = float(val_bpb_match.group(1))
            metrics["outcome"] = 2.0 - metrics["val_bpb"]

        for key, pattern in {
            "total_tokens_M": r"total_tokens_M:\s+([\d.]+)",
            "num_steps": r"num_steps:\s+(\d+)",
            "num_params_M": r"num_params_M:\s+([\d.]+)",
        }.items():
            m = re.search(pattern, out)
            if m:
                metrics[key] = float(m.group(1))

        return {
            "metrics": metrics,
            "compute_cost": wall_time / 3600,  # Rough: hours of GPU
            "raw_log": out[-2000:] if len(out) > 2000 else out,
        }

    return execute


def make_dry_run_executor():
    """Executor that logs but doesn't execute. For testing."""
    def execute(proposal: Proposal) -> dict:
        logger.info("DRY RUN: %s %s", proposal.intervention_type, proposal.intervention_spec)
        return {"metrics": {"dry_run": True}, "raw_log": "dry-run"}
    return execute
