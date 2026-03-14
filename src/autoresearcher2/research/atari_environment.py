"""Atari game environment via SSH.

Runs PPO training on Atari games on a remote VM, parses mean_reward,
returns normalized score (higher = better). Uses the same nohup+polling
pattern as TrainPyEnvironment.
"""

import re
import subprocess
import time
import uuid

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.environment import Environment

# Known score ranges for normalization (random → human-level approximate)
SCORE_RANGES = {
    "Breakout": (1.7, 30.0),
    "SpaceInvaders": (148.0, 1669.0),
    "Pong": (-20.7, 14.6),
}


class AtariEnvironment(Environment):
    """Run Atari PPO experiments on a remote GPU VM."""

    def __init__(
        self,
        schema: InterventionSchema,
        ssh_host: str = "root@dllm-experiment.home",
        ssh_key: str = "~/.ssh/pve03_key",
        remote_script: str = "~/github.com/erikdebruijn/atari-research/train_atari.py",
        cuda_device: str = "0",
        poll_interval: int = 30,
        max_wait: int = 3600,
        total_timesteps: int = 1_000_000,
    ):
        self.schema = schema
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key
        self.remote_script = remote_script
        self.cuda_device = cuda_device
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.total_timesteps = total_timesteps
        self.last_run_metadata = {}

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
        out_file = f"/tmp/atari_{job_id}.out"
        done_file = f"/tmp/atari_{job_id}.done"

        # Build command args from config
        game = config.get("game", "Breakout")
        lr = config.get("learning_rate", "5e-4")
        net_size = config.get("network_size", "medium")

        # Launch training via nohup
        self._ssh(
            f"nohup bash -c '"
            f"CUDA_VISIBLE_DEVICES={self.cuda_device} python3 {self.remote_script} "
            f"--game {game} --lr {lr} --network-size {net_size} "
            f"--total-timesteps {self.total_timesteps} "
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
                continue
        else:
            self._ssh(f"pkill -f 'train_atari.py' 2>/dev/null; "
                      f"rm -f {out_file} {done_file}")
            raise RuntimeError(f"train_atari.py timed out after {self.max_wait}s")

        train_rc = int(check.strip())
        _, out = self._ssh(f"cat {out_file}", timeout=30)

        # Cleanup
        self._ssh(f"rm -f {out_file} {done_file}")

        if train_rc != 0:
            raise RuntimeError(f"train_atari.py failed (exit {train_rc}): {out[-300:]}")

        # Parse mean_reward
        match = re.search(r"mean_reward:\s+([-\d.]+)", out)
        if not match:
            raise RuntimeError("Could not parse mean_reward from output")

        mean_reward = float(match.group(1))

        # Parse metadata
        self.last_run_metadata = self._parse_metadata(out)
        self.last_run_metadata["raw_mean_reward"] = mean_reward

        # Normalize to [0, 1] using known score ranges
        low, high = SCORE_RANGES.get(game, (0.0, 100.0))
        outcome = (mean_reward - low) / (high - low)
        outcome = max(0.0, min(1.0, outcome))

        return outcome

    def _parse_metadata(self, output: str) -> dict:
        meta = {}
        patterns = {
            "total_timesteps": r"total_timesteps:\s+(\d+)",
            "training_time_s": r"training_time_s:\s+([\d.]+)",
            "episodes": r"episodes:\s+(\d+)",
            "fps": r"fps:\s+([\d.]+)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, output)
            if m:
                meta[key] = float(m.group(1))
        return meta
