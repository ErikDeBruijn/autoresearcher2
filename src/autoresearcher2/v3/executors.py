"""Execution functions for different domains.

Each executor takes a Proposal and returns a result dict with:
- metrics: dict of outcome measurements
- compute_cost: float (optional)
- raw_log: str (optional, for audit)

These are passed as execute_fn to Worker.
"""
import json
import logging
import os
import re
import subprocess
import time
import urllib.request
import urllib.error

from autoresearcher2.v3.proposal import Proposal

logger = logging.getLogger(__name__)

_DEFAULT_SSH_HOST = os.environ.get("AUTORESEARCHER_SSH_HOST", "root@dllm-experiment.local")
_DEFAULT_SSH_KEY = os.environ.get("AUTORESEARCHER_SSH_KEY", "~/.ssh/pve03_key")


def _make_run_cmd(ssh_host: str | None = None, ssh_key: str | None = None, timeout: int = 900):
    """Create a run_cmd function that executes commands locally or via SSH."""
    def run_cmd(cmd: str) -> tuple[int, str]:
        if ssh_host:
            ssh_cmd = ["ssh"]
            if ssh_key:
                ssh_cmd += ["-i", ssh_key]
            ssh_cmd += ["-o", "ConnectTimeout=10", ssh_host, cmd]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        else:
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    return run_cmd


def _parse_metrics(output: str, patterns: dict[str, str]) -> dict[str, float]:
    """Extract named metrics from command output using regex patterns."""
    metrics = {}
    for name, pattern in patterns.items():
        m = re.search(pattern, output)
        if m:
            metrics[name] = float(m.group(1))
    return metrics


def _apply_code_changes(run_cmd, spec: dict, work_dir: str) -> None:
    """Apply code changes from a proposal spec to a working directory.

    Handles two formats:
    - "diff": a unified diff string applied via patch
    - "file_changes": a dict of {filename: content} written as files

    Raises ValueError if neither is present, RuntimeError on apply failure.
    """
    import base64

    diff_content = spec.get("diff")
    file_changes = spec.get("file_changes", {})

    if not diff_content and not file_changes:
        raise ValueError("code_change requires 'diff' or 'file_changes' in intervention_spec")

    if diff_content:
        b64 = base64.b64encode(diff_content.encode()).decode()
        run_cmd(
            f"python3 -c \"import base64; open('/tmp/_ar_patch.diff','w').write(base64.b64decode('{b64}').decode())\""
        )
        rc, out = run_cmd(f"cd {work_dir} && patch -p1 --no-backup-if-mismatch < /tmp/_ar_patch.diff")
        if rc != 0:
            raise RuntimeError(f"patch failed (exit {rc}): {out[-500:]}")
        logger.info("code_change: applied diff (%d bytes)", len(diff_content))
    else:
        for filename, content in file_changes.items():
            safe_name = filename.replace("/", "_").replace("..", "_")
            b64 = base64.b64encode(content.encode()).decode()
            rc, out = run_cmd(
                f"python3 -c \"import base64; open('{work_dir}/{safe_name}','w').write(base64.b64decode('{b64}').decode())\""
            )
            if rc != 0:
                raise RuntimeError(f"Failed to write {safe_name}: {out[-300:]}")
            logger.info("code_change: wrote %s (%d bytes)", safe_name, len(content))

# gpu-cost-tracker service endpoint (override with COST_TRACKER_URL env var)
COST_TRACKER_URL = os.environ.get("COST_TRACKER_URL", "http://pve03.local:8377")


def _start_cost_job(gpu: int, label: str, client: str = "autoresearcher") -> str | None:
    """Start a cost tracking job. Returns job_id or None on failure.

    If a stale job exists on this GPU (409), clears it and retries once.
    """
    data = json.dumps({"gpu": gpu, "client": client, "label": label}).encode()

    for attempt in range(2):
        try:
            req = urllib.request.Request(
                f"{COST_TRACKER_URL}/job/start",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read()).get("job_id")
        except urllib.error.HTTPError as e:
            if e.code == 409 and attempt == 0:
                logger.info("Cost tracker 409 on GPU %d — clearing stale job", gpu)
                _clear_stale_jobs(gpu)
                continue
            logger.warning("Cost tracker start failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Cost tracker start failed: %s", e)
            return None
    return None


def _clear_stale_jobs(gpu: int) -> None:
    """Stop any active cost tracking jobs on the given GPU."""
    try:
        req = urllib.request.Request(f"{COST_TRACKER_URL}/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read())
        for job_id, job in status.get("active_jobs", {}).items():
            if job.get("gpu") == gpu:
                _stop_cost_job(job_id)
                logger.info("Stopped stale job %s on GPU %d", job_id, gpu)
    except Exception as e:
        logger.warning("Failed to clear stale jobs on GPU %d: %s", gpu, e)


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


def with_cost_tracking(execute_fn, cuda_device: str | None = None):
    """Wrap an executor to add GPU cost tracking around each execution.

    Starts a cost tracking job before calling the inner executor and stops it
    after, merging energy_kwh/cost_eur/avg_power_w into the result dict.
    If the inner executor raises, the cost job is still stopped to avoid leaks.

    Args:
        execute_fn: The inner executor function (Proposal -> dict).
        cuda_device: CUDA device ID string (e.g. "0", "1"). None to skip tracking.

    Returns:
        A wrapped executor function with the same signature.
    """
    if cuda_device is None or not cuda_device.isdigit():
        return execute_fn

    gpu_index = int(cuda_device)

    def execute(proposal):
        job_id = _start_cost_job(gpu=gpu_index, label=proposal.id)
        try:
            result = execute_fn(proposal)
        except Exception:
            if job_id:
                _stop_cost_job(job_id)
            raise

        cost_data = _stop_cost_job(job_id) if job_id else None
        if cost_data:
            result["energy_kwh"] = cost_data.get("energy_kwh")
            result["cost_eur"] = cost_data.get("cost_eur")
            result["avg_power_w"] = cost_data.get("avg_power_w")

        return result

    return execute


NANOGPT_METRIC_PATTERNS = {
    "val_bpb": r"val_bpb:\s+([\d.]+)",
    "total_tokens_M": r"total_tokens_M:\s+([\d.]+)",
    "num_steps": r"num_steps:\s+(\d+)",
    "num_params_M": r"num_params_M:\s+([\d.]+)",
}


def make_trainpy_executor(
    ssh_host: str = _DEFAULT_SSH_HOST,
    ssh_key: str = _DEFAULT_SSH_KEY,
    remote_dir: str = "~/github.com/karpathy/autoresearch",
    cuda_device: str = "1",
    timeout: int = 900,
    local: bool = False,
    metric_patterns: dict[str, str] = None,
):
    """Create an executor for train.py-style experiments.

    Handles config_change (patches train.py knobs) and probe (limited steps).
    Set local=True when running on the VM itself to skip SSH.

    Each executor gets its own working copy of train.py to avoid race
    conditions when multiple workers run concurrently.

    Args:
        metric_patterns: Dict of {metric_name: regex_pattern} to extract from output.
            Each pattern should have one capture group for the numeric value.
            Defaults to NANOGPT_METRIC_PATTERNS.
    """
    if metric_patterns is None:
        metric_patterns = NANOGPT_METRIC_PATTERNS

    run_cmd = _make_run_cmd(
        ssh_host=None if local else ssh_host,
        ssh_key=None if local else ssh_key,
        timeout=timeout,
    )

    # Expand ~ for local execution
    base_dir = remote_dir.replace("~", "/root") if local else remote_dir
    # Per-GPU working directory to avoid concurrent sed-patching of same train.py
    train_dir = f"{base_dir}_gpu{cuda_device}"

    # Create lightweight per-worker dir: symlink venv/data/config, copy only train.py
    rc, out = run_cmd(
        f"test -d {train_dir} || ("
        f"mkdir -p {train_dir} && "
        f"ln -sf {base_dir}/.venv {train_dir}/.venv && "
        f"ln -sf {base_dir}/data {train_dir}/data && "
        f"ln -sf {base_dir}/pyproject.toml {train_dir}/pyproject.toml && "
        f"ln -sf {base_dir}/uv.lock {train_dir}/uv.lock && "
        f"ln -sf {base_dir}/.python-version {train_dir}/.python-version && "
        f"cp {base_dir}/train.py {train_dir}/train.py && "
        f"cp {base_dir}/prepare.py {train_dir}/prepare.py"
        f")"
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
            _apply_code_changes(run_cmd, spec, train_dir)
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

        # Run training
        start = time.time()
        rc, out = run_cmd(
            f"cd {train_dir} && CUDA_VISIBLE_DEVICES={cuda_device} uv run train.py 2>&1"
        )
        wall_time = time.time() - start

        if rc != 0:
            raise RuntimeError(f"train.py failed (exit {rc}): {out[-500:]}")

        result = {
            "metrics": _parse_metrics(out, metric_patterns),
            "compute_cost": wall_time / 3600,  # Rough: hours of GPU
            "raw_log": out[-2000:] if len(out) > 2000 else out,
        }

        return result

    return execute


def make_shell_executor(
    command_template: str,
    metric_patterns: dict[str, str] = None,
    timeout: int = 900,
    ssh_host: str = None,
    ssh_key: str = None,
    work_dir: str = None,
    base_script: str = None,
):
    """Generic executor that runs a shell command with intervention_spec as env vars.

    Supports code_change proposals: writes file_changes to work_dir before running.
    Resets base_script before each run to ensure clean state.

    Args:
        command_template: Shell command to run. intervention_spec keys are
            available as env vars (e.g. param1=value1, param2=value2).
        metric_patterns: Dict of {metric_name: regex_pattern} to extract from output.
            Each pattern should have one capture group for the numeric value.
        timeout: Max seconds for the command.
        ssh_host: If set, run via SSH (e.g. "root@host"). None = run locally.
        ssh_key: SSH key path (only used with ssh_host).
        work_dir: Working directory for code_change file writes. Required for code_change.
        base_script: Path to the base training script to reset before each run.
    """
    metric_patterns = metric_patterns or {}

    run_cmd = _make_run_cmd(ssh_host=ssh_host, ssh_key=ssh_key, timeout=timeout)

    def execute(proposal: Proposal) -> dict:
        spec = proposal.intervention_spec
        itype = proposal.intervention_type

        # Always reset base script before any run (prevents stale scripts from prior code_changes)
        if base_script:
            run_cmd(f"cp {base_script} {work_dir}/$(basename {base_script})")

        # Handle code_change: apply diff or write file_changes before running
        if itype == "code_change" and work_dir:
            _apply_code_changes(run_cmd, spec, work_dir)

        # Build env vars from spec (skip file_changes and non-shell-safe values)
        safe_pairs = []
        for k, v in spec.items():
            if k == "file_changes":
                continue
            sv = str(v)
            if re.match(r'^[\w.+\-]+$', k) and re.match(r'^[\w.+\-]+$', sv):
                safe_pairs.append(f"{k}={sv}")
            else:
                logger.debug("Skipping non-shell-safe spec: %s=%s", k, sv[:50])
        env_str = " ".join(safe_pairs)
        cmd = f"{env_str} {command_template}" if env_str else command_template

        start = time.time()
        rc, out = run_cmd(cmd)
        wall_time = time.time() - start

        if rc != 0:
            raise RuntimeError(f"Command failed (exit {rc}): {out[-500:]}")

        metrics = _parse_metrics(out, metric_patterns)

        # Parse artifact paths from stdout (format: "artifact_<name>: /path/to/file")
        artifact_paths = {}
        for m in re.finditer(r"artifact_(\w+):\s+(.+)", out):
            artifact_paths[m.group(1)] = m.group(2).strip()

        exec_result = {
            "metrics": metrics,
            "compute_cost": wall_time / 3600,
            "raw_log": out[-2000:] if len(out) > 2000 else out,
        }

        if artifact_paths:
            exec_result["artifact_paths"] = artifact_paths

        return exec_result

    return execute


def make_dispatch_executor(executors_by_project: dict[str | None, callable]):
    """Executor that dispatches to project-specific executors.

    Args:
        executors_by_project: Dict mapping project_id to executor function.
            Use None key as the default executor.
    """
    def execute(proposal: Proposal) -> dict:
        project_id = getattr(proposal, "project_id", None)
        executor = executors_by_project.get(project_id)
        if executor is None:
            executor = executors_by_project.get(None)
        if executor is None:
            raise ValueError(f"No executor for project {project_id}")
        return executor(proposal)
    return execute


def make_dry_run_executor():
    """Executor that logs but doesn't execute. For testing."""
    def execute(proposal: Proposal) -> dict:
        logger.info("DRY RUN: %s %s", proposal.intervention_type, proposal.intervention_spec)
        return {"metrics": {"dry_run": True}, "raw_log": "dry-run"}
    return execute
