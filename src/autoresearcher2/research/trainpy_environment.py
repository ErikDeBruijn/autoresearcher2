"""Real train.py environment via SSH.

Patches knobs in train.py on a remote VM, runs training,
parses val_bpb, returns transformed outcome (higher = better).

Training runs via nohup on the VM so SSH disconnects don't kill
experiments. The local side polls for completion with short SSH calls.
"""

import re
import subprocess
import time
import uuid

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.environment import Environment


class TrainPyEnvironment(Environment):
    """Run real train.py experiments on a remote GPU VM."""

    def __init__(
        self,
        schema: InterventionSchema,
        ssh_host: str = "root@dllm-experiment.home",
        ssh_key: str = "~/.ssh/pve03_key",
        remote_dir: str = "~/github.com/karpathy/autoresearch",
        cuda_device: str = "1",
        poll_interval: int = 15,
        max_wait: int = 600,
        dataset: str = "climbmix",
    ):
        self.schema = schema
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key
        self.remote_dir = remote_dir
        self.cuda_device = cuda_device
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.dataset = dataset

    def _ssh(self, cmd: str, timeout: int = 30) -> tuple[int, str]:
        result = subprocess.run(
            ["ssh", "-i", self.ssh_key,
             "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=2",
             self.ssh_host, cmd],
            capture_output=True, text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr

    def run(self, cell_index: int) -> float:
        config = self.schema.cell_to_config(cell_index)
        job_id = uuid.uuid4().hex[:8]
        out_file = f"/tmp/train_{job_id}.out"
        done_file = f"/tmp/train_{job_id}.done"

        # Reset train.py
        self._ssh(f"cd {self.remote_dir} && git checkout train.py")

        # Patch knobs
        for knob, value in config.items():
            self._ssh(
                f"sed -i 's/^{knob} = .*/{knob} = {value}/' "
                f"{self.remote_dir}/train.py"
            )

        # Prepare dataset if not climbmix (default)
        dataset_env = ""
        if self.dataset != "climbmix":
            # Run prepare.py with dataset flag first (idempotent)
            prep_rc, prep_out = self._ssh(
                f"cd {self.remote_dir} && "
                f"uv run prepare.py --dataset {self.dataset} --num-shards 8",
                timeout=300,
            )
            if prep_rc != 0:
                raise RuntimeError(
                    f"prepare.py --dataset {self.dataset} failed: {prep_out[-300:]}"
                )
            dataset_env = f"AUTORESEARCH_DATASET={self.dataset} "

        # Launch training via nohup — SSH disconnects won't kill it
        self._ssh(
            f"nohup bash -c '"
            f"cd {self.remote_dir} && "
            f"{dataset_env}"
            f"CUDA_VISIBLE_DEVICES={self.cuda_device} uv run train.py "
            f"> {out_file} 2>&1; echo $? > {done_file}"
            f"' &>/dev/null &"
        )

        # Poll for completion
        start = time.time()
        while time.time() - start < self.max_wait:
            time.sleep(self.poll_interval)
            try:
                rc, check = self._ssh(f"cat {done_file} 2>/dev/null")
                if rc == 0 and check.strip():
                    break
            except (subprocess.TimeoutExpired, Exception):
                continue  # SSH hiccup, retry next poll
        else:
            # Timeout — kill any remaining train process
            self._ssh(f"pkill -f 'train.py' 2>/dev/null; rm -f {out_file} {done_file}")
            raise RuntimeError(f"train.py timed out after {self.max_wait}s")

        # Read results
        train_rc = int(check.strip())
        _, out = self._ssh(f"cat {out_file}", timeout=30)

        # Cleanup remote files and reset train.py
        self._ssh(f"rm -f {out_file} {done_file}")
        self._ssh(f"cd {self.remote_dir} && git checkout train.py")

        if train_rc != 0:
            raise RuntimeError(f"train.py failed (exit {train_rc}): {out[-300:]}")

        # Parse val_bpb
        match = re.search(r"val_bpb:\s+([\d.]+)", out)
        if not match:
            raise RuntimeError(f"Could not parse val_bpb from output")

        val_bpb = float(match.group(1))

        # Parse training metadata (tokens, steps, throughput)
        self.last_run_metadata = self._parse_metadata(out)

        # Transform: val_bpb ~1.0-1.2, lower is better
        # Map to outcome where higher = better, roughly in [0, 1]
        outcome = 2.0 - val_bpb
        return outcome

    def _parse_metadata(self, output: str) -> dict:
        """Extract training stats from train.py output."""
        meta = {}
        patterns = {
            "total_tokens_M": r"total_tokens_M:\s+([\d.]+)",
            "num_steps": r"num_steps:\s+(\d+)",
            "num_params_M": r"num_params_M:\s+([\d.]+)",
            "steady_state_mfu": r"steady_state_mfu:\s+([\d.]+)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, output)
            if m:
                meta[key] = float(m.group(1))
        return meta
