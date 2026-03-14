#!/usr/bin/env python3
"""Patch to apply to prepare.py on the VM for multi-dataset support.

Run this script on the VM to patch prepare.py with --dataset flag support.
Adds support for: climbmix (default), wikipedia, code.

Usage on VM:
    python prepare_dataset_patch.py  # patches prepare.py in-place
"""

import re
import sys
from pathlib import Path

PREPARE_PY = Path.home() / "github.com/karpathy/autoresearch/prepare.py"

# The dataset configurations to inject
DATASET_CONFIG = '''
# ---------------------------------------------------------------------------
# Dataset configurations (added by autoresearcher2 v2.1)
# ---------------------------------------------------------------------------

DATASETS = {
    "climbmix": {
        "base_url": "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        "max_shard": 6542,
        "val_shard": 6542,
        "description": "ClimbMix-400B shuffled pretraining data",
    },
    "wikipedia": {
        "base_url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en",
        "max_shard": 41,
        "val_shard": 41,
        "description": "English Wikipedia (Nov 2023)",
    },
    "code": {
        "base_url": "https://huggingface.co/datasets/bigcode/the-stack-smol/resolve/main/data/python",
        "max_shard": 15,
        "val_shard": 15,
        "description": "The Stack (Python subset, small)",
    },
}
'''


def patch_prepare():
    if not PREPARE_PY.exists():
        print(f"ERROR: {PREPARE_PY} not found")
        sys.exit(1)

    content = PREPARE_PY.read_text()

    # Check if already patched
    if "AUTORESEARCH_DATASET" in content:
        print("prepare.py is already patched for multi-dataset support")
        return

    # 1. Add --dataset argument to argparse
    content = content.replace(
        'parser.add_argument("--num-shards"',
        'parser.add_argument("--dataset", type=str, default="climbmix",\n'
        '                        choices=["climbmix", "wikipedia", "code"],\n'
        '                        help="Dataset to prepare (default: climbmix)")\n'
        '    parser.add_argument("--num-shards"',
    )

    # 2. Add dataset env var support to CACHE_DIR
    # Replace fixed CACHE_DIR with dataset-aware one
    content = content.replace(
        'CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")',
        '_DATASET = os.environ.get("AUTORESEARCH_DATASET", "climbmix")\n'
        'CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", _DATASET)\n'
        '# For backward compat: climbmix uses the original path\n'
        'if _DATASET == "climbmix":\n'
        '    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")',
    )

    # 3. In __main__, use args.dataset to set env var before download
    content = content.replace(
        'num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards',
        '# Set dataset env var for CACHE_DIR resolution\n'
        '    if args.dataset != "climbmix":\n'
        '        os.environ["AUTORESEARCH_DATASET"] = args.dataset\n'
        '        # Re-derive paths for new dataset (module-level vars, no global needed)\n'
        '        CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", args.dataset)\n'
        '        DATA_DIR = os.path.join(CACHE_DIR, "data")\n'
        '        TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")\n'
        '\n'
        '    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards',
    )

    PREPARE_PY.write_text(content)
    print(f"Patched {PREPARE_PY} with multi-dataset support")
    print("Supported datasets: climbmix (default), wikipedia, code")
    print("Usage: python prepare.py --dataset wikipedia --num-shards 8")
    print("Runtime: AUTORESEARCH_DATASET=wikipedia uv run train.py")


if __name__ == "__main__":
    patch_prepare()
