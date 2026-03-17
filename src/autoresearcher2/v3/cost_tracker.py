"""GPU cost tracking via the gpu-cost-tracker HTTP service.

Functions to start/stop cost tracking jobs and a wrapper to add cost
tracking around any executor function.
"""
import json
import logging
import os
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

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
