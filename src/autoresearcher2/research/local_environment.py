"""Local train.py environment — runs training on the local machine.

Patches knobs in a local copy of train.py, runs training,
parses val_bpb, returns transformed outcome (higher = better).

Useful for development/debugging on laptops with MPS or CUDA GPUs.
For production runs, use TrainPyEnvironment (remote SSH).
"""

import os
import re
import subprocess
import tempfile
import shutil
import time

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.environment import Environment


class LocalTrainPyEnvironment(Environment):
    """Run train.py experiments locally (no SSH)."""

    def __init__(
        self,
        schema: InterventionSchema,
        train_dir: str,
        device: str = "mps",
        time_budget: int = 300,
        dataset: str = "climbmix",
    ):
        self.schema = schema
        self.train_dir = os.path.expanduser(train_dir)
        self.device = device
        self.time_budget = time_budget
        self.dataset = dataset
        self.last_run_metadata = {}

    def run(self, cell_index: int) -> float:
        config = self.schema.cell_to_config(cell_index)

        # Create a temporary copy of train.py to patch
        train_py = os.path.join(self.train_dir, "train.py")
        if not os.path.exists(train_py):
            raise RuntimeError(f"train.py not found at {train_py}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=self.train_dir,
            prefix="train_run_", delete=False,
        ) as tmp:
            with open(train_py) as f:
                content = f.read()

            # Patch knobs
            for knob, value in config.items():
                content = re.sub(
                    rf"^{knob} = .*$",
                    f"{knob} = {value}",
                    content,
                    flags=re.MULTILINE,
                )

            # Patch device if needed (replace CUDA references)
            if self.device == "mps":
                content = content.replace(
                    'device = torch.device("cuda")',
                    'device = torch.device("mps")',
                )
                # Disable CUDA-specific calls
                content = content.replace("torch.cuda.synchronize()", "torch.mps.synchronize()")
                content = content.replace("torch.cuda.max_memory_allocated()", "0")
                content = content.replace(
                    'autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)',
                    'autocast_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16)',
                )

            # Patch time budget
            content = re.sub(
                r"^TIME_BUDGET = \d+",
                f"TIME_BUDGET = {self.time_budget}",
                content,
                flags=re.MULTILINE,
            )

            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Set dataset env var
            env = os.environ.copy()
            if self.dataset != "climbmix":
                env["AUTORESEARCH_DATASET"] = self.dataset

            # Run training
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True, text=True,
                timeout=self.time_budget + 120,
                cwd=self.train_dir,
                env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"train.py failed (exit {result.returncode}): "
                    f"{result.stderr[-300:]}"
                )

            output = result.stdout + result.stderr

            # Parse val_bpb
            match = re.search(r"val_bpb:\s+([\d.]+)", output)
            if not match:
                raise RuntimeError("Could not parse val_bpb from output")

            val_bpb = float(match.group(1))
            self.last_run_metadata = self._parse_metadata(output)

            # Transform: higher = better
            return 2.0 - val_bpb

        finally:
            os.unlink(tmp_path)

    def _parse_metadata(self, output: str) -> dict:
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
