"""Execution functions for different domains.

Each executor takes a Proposal and returns a result dict with:
- metrics: dict of outcome measurements
- compute_cost: float (optional)
- raw_log: str (optional, for audit)

These are passed as execute_fn to Worker.
"""
import json
import logging
import re
import subprocess
import time
import urllib.request
import urllib.error

from autoresearcher2.v3.proposal import Proposal

logger = logging.getLogger(__name__)

# gpu-cost-tracker service endpoint
COST_TRACKER_URL = "http://pve03.local:8377"


def _start_cost_job(gpu: int, label: str, client: str = "autoresearcher") -> str | None:
    """Start a cost tracking job. Returns job_id or None on failure."""
    try:
        data = json.dumps({"gpu": gpu, "client": client, "label": label}).encode()
        req = urllib.request.Request(
            f"{COST_TRACKER_URL}/job/start",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            return result.get("job_id")
    except Exception as e:
        logger.warning("Cost tracker start failed: %s", e)
        return None


def _stop_cost_job(job_id: str) -> dict | None:
    """Stop a cost tracking job. Returns cost data or None on failure."""
    try:
        req = urllib.request.Request(
            f"{COST_TRACKER_URL}/job/{job_id}",
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("Cost tracker stop failed: %s", e)
        return None


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

    Each executor gets its own working copy of train.py to avoid race
    conditions when multiple workers run concurrently.
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
    base_dir = remote_dir.replace("~", "/root") if local else remote_dir
    # Per-GPU working directory to avoid concurrent sed-patching of same train.py
    train_dir = f"{base_dir}_gpu{cuda_device}"

    # Create per-worker copy if it doesn't exist
    rc, out = run_cmd(
        f"test -d {train_dir} || cp -r {base_dir} {train_dir}"
    )
    if rc != 0:
        logger.warning("Failed to create per-GPU dir %s: %s", train_dir, out)
        train_dir = base_dir  # Fallback to shared dir

    def execute(proposal: Proposal) -> dict:
        spec = proposal.intervention_spec
        itype = proposal.intervention_type

        if itype not in ("config_change", "probe", "code_change"):
            logger.warning("trainpy executor: unsupported type %s, dry-run", itype)
            return {"metrics": {"unsupported": True}, "raw_log": f"dry-run for {itype}"}

        # Reset train.py from base dir (working copy may not be a git repo)
        run_cmd(f"cp {base_dir}/train.py {train_dir}/train.py")

        if itype == "code_change":
            # LLM provides file contents directly via file_changes key
            file_changes = spec.get("file_changes", {})
            if not file_changes:
                raise ValueError("code_change requires 'file_changes' in intervention_spec")
            for filename, content in file_changes.items():
                # Sanitize filename: only allow simple filenames within train_dir
                safe_name = filename.replace("/", "_").replace("..", "_")
                # Write via python -c to avoid shell quoting issues
                import base64
                b64 = base64.b64encode(content.encode()).decode()
                run_cmd(
                    f"python3 -c \"import base64; open('{train_dir}/{safe_name}','w').write(base64.b64decode('{b64}').decode())\""
                )
                logger.info("code_change: wrote %s (%d bytes)", safe_name, len(content))
        else:
            # Patch knobs from intervention_spec
            for knob, value in spec.items():
                if knob == "run_steps":
                    continue  # Not a train.py knob
                run_cmd(f"sed -i 's/^{knob} = .*/{knob} = {value}/' {train_dir}/train.py")

            # For probes, limit steps
            if itype == "probe" and "run_steps" in spec:
                run_steps = spec["run_steps"]
                run_cmd(f"sed -i 's/^num_steps = .*/num_steps = {run_steps}/' {train_dir}/train.py")

        # Start cost tracking
        gpu_index = int(cuda_device) if cuda_device.isdigit() else 0
        job_id = _start_cost_job(gpu=gpu_index, label=proposal.id)

        # Run training
        start = time.time()
        rc, out = run_cmd(
            f"cd {train_dir} && CUDA_VISIBLE_DEVICES={cuda_device} uv run train.py 2>&1"
        )
        wall_time = time.time() - start

        # Stop cost tracking and collect data
        cost_data = _stop_cost_job(job_id) if job_id else None

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

        result = {
            "metrics": metrics,
            "compute_cost": wall_time / 3600,  # Rough: hours of GPU
            "raw_log": out[-2000:] if len(out) > 2000 else out,
        }

        if cost_data:
            result["energy_kwh"] = cost_data.get("energy_kwh")
            result["cost_eur"] = cost_data.get("cost_eur")
            result["avg_power_w"] = cost_data.get("avg_power_w")

        return result

    return execute


def make_shell_executor(
    command_template: str,
    metric_patterns: dict[str, str] = None,
    timeout: int = 900,
    ssh_host: str = None,
    ssh_key: str = None,
    cuda_device: str = None,
):
    """Generic executor that runs a shell command with intervention_spec as env vars.

    Args:
        command_template: Shell command to run. intervention_spec keys are
            available as env vars (e.g. DEPTH=8, MATRIX_LR=0.04).
        metric_patterns: Dict of {metric_name: regex_pattern} to extract from output.
            Each pattern should have one capture group for the numeric value.
        timeout: Max seconds for the command.
        ssh_host: If set, run via SSH (e.g. "root@host"). None = run locally.
        ssh_key: SSH key path (only used with ssh_host).
    """
    metric_patterns = metric_patterns or {}

    def execute(proposal: Proposal) -> dict:
        spec = proposal.intervention_spec
        env_str = " ".join(f"{k}={v}" for k, v in spec.items())
        cmd = f"{env_str} {command_template}"

        if ssh_host:
            ssh_cmd = ["ssh"]
            if ssh_key:
                ssh_cmd += ["-i", ssh_key]
            ssh_cmd += ["-o", "ConnectTimeout=10", ssh_host, cmd]
            run_args = ssh_cmd
        else:
            run_args = ["bash", "-c", cmd]

        # Start cost tracking if GPU specified
        job_id = None
        if cuda_device and cuda_device.isdigit():
            job_id = _start_cost_job(gpu=int(cuda_device), label=proposal.id)

        start = time.time()
        result = subprocess.run(run_args, capture_output=True, text=True, timeout=timeout)
        wall_time = time.time() - start
        out = result.stdout + result.stderr

        # Stop cost tracking
        cost_data = _stop_cost_job(job_id) if job_id else None

        if result.returncode != 0:
            raise RuntimeError(f"Command failed (exit {result.returncode}): {out[-500:]}")

        metrics = {}
        for name, pattern in metric_patterns.items():
            m = re.search(pattern, out)
            if m:
                metrics[name] = float(m.group(1))

        exec_result = {
            "metrics": metrics,
            "compute_cost": wall_time / 3600,
            "raw_log": out[-2000:] if len(out) > 2000 else out,
        }

        if cost_data:
            exec_result["energy_kwh"] = cost_data.get("energy_kwh")
            exec_result["cost_eur"] = cost_data.get("cost_eur")
            exec_result["avg_power_w"] = cost_data.get("avg_power_w")

        return exec_result

    return execute


def make_dry_run_executor():
    """Executor that logs but doesn't execute. For testing."""
    def execute(proposal: Proposal) -> dict:
        logger.info("DRY RUN: %s %s", proposal.intervention_type, proposal.intervention_spec)
        return {"metrics": {"dry_run": True}, "raw_log": "dry-run"}
    return execute
