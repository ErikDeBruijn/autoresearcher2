"""Retry failed experiments from a completed evidence run.

Reads the JSON data, finds failed experiments, reruns them using the
nohup+polling TrainPyEnvironment, and patches the results back in.
"""

import json
import time
import sys

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment

SCHEMA = InterventionSchema(factors={
    "DEPTH": ["6", "8", "10"],
    "MATRIX_LR": ["0.02", "0.04", "0.08"],
    "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
})


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    data_file = "artifacts/evidence/d240b43f.json"

    with open(data_file) as f:
        data = json.load(f)

    results = data["approaches"]["full"]["results"]
    failed_indices = [i for i, r in enumerate(results) if not r.get("val_bpb")]

    if not failed_indices:
        log("No failed experiments to retry.")
        return

    log(f"Found {len(failed_indices)} failed experiments to retry")

    env = TrainPyEnvironment(
        schema=SCHEMA,
        ssh_host="root@dllm-experiment.local",
        ssh_key="~/.ssh/pve03_key",
        remote_dir="~/github.com/karpathy/autoresearch",
        cuda_device="1",
    )

    retried = 0
    for idx in failed_indices:
        r = results[idx]
        cell = r["cell"]
        config = r["config"]
        original_error = r.get("error", "?")[:80]
        log(f"  Retrying exp {r['experiment']} | cell {cell} | {config}")
        log(f"    original error: {original_error}")

        start = time.time()
        try:
            outcome = env.run(cell)
            val_bpb = 2.0 - outcome
            wall_time = time.time() - start
            metadata = getattr(env, "last_run_metadata", {})
            tokens_M = metadata.get("total_tokens_M")
            tok_per_sec = round(tokens_M * 1e6 / wall_time, 1) if tokens_M and wall_time > 0 else None

            # Patch the result in place
            results[idx]["outcome"] = outcome
            results[idx]["val_bpb"] = val_bpb
            results[idx]["wall_time_s"] = round(wall_time, 1)
            results[idx]["tokens_M"] = tokens_M
            results[idx]["tok_per_sec"] = tok_per_sec
            results[idx]["num_steps"] = metadata.get("num_steps")
            results[idx]["mfu"] = metadata.get("steady_state_mfu")
            results[idx]["error"] = None
            results[idx]["retry_of_original_error"] = r.get("error", "")

            log(f"    SUCCESS: val_bpb={val_bpb:.6f} tokens={tokens_M}M")
            retried += 1

        except Exception as e:
            log(f"    STILL FAILED: {e}")
            results[idx]["retry_error"] = str(e)

        # Save after each retry
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    successful = [r for r in results if r.get("val_bpb")]
    log(f"Done: {retried}/{len(failed_indices)} retries succeeded")
    log(f"Full approach now: {len(successful)}/20 successful")
    if successful:
        best = min(successful, key=lambda r: r["val_bpb"])
        log(f"Best val_bpb: {best['val_bpb']:.6f} — {best['config']}")


if __name__ == "__main__":
    main()
