"""Real train.py environment via SSH.

Patches knobs in train.py on a remote VM, runs training,
parses val_bpb, returns transformed outcome (higher = better).
"""

import re
import subprocess

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
        ssh_timeout: int = 900,
    ):
        self.schema = schema
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key
        self.remote_dir = remote_dir
        self.cuda_device = cuda_device
        self.ssh_timeout = ssh_timeout

    def _ssh(self, cmd: str, timeout: int | None = None) -> tuple[int, str]:
        result = subprocess.run(
            ["ssh", "-i", self.ssh_key,
             "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=30",
             "-o", "ServerAliveCountMax=3",
             self.ssh_host, cmd],
            capture_output=True, text=True,
            timeout=timeout or self.ssh_timeout,
        )
        return result.returncode, result.stdout + result.stderr

    def run(self, cell_index: int) -> float:
        config = self.schema.cell_to_config(cell_index)

        # Reset train.py
        self._ssh(f"cd {self.remote_dir} && git checkout train.py")

        # Patch knobs
        for knob, value in config.items():
            self._ssh(
                f"sed -i 's/^{knob} = .*/{knob} = {value}/' "
                f"{self.remote_dir}/train.py"
            )

        # Run training
        rc, out = self._ssh(
            f"cd {self.remote_dir} && "
            f"CUDA_VISIBLE_DEVICES={self.cuda_device} uv run train.py 2>&1",
        )

        # Reset train.py after run
        self._ssh(f"cd {self.remote_dir} && git checkout train.py")

        if rc != 0:
            raise RuntimeError(f"train.py failed (exit {rc}): {out[-300:]}")

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
