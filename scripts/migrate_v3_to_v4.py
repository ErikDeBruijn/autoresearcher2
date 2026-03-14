#!/usr/bin/env python3
"""Migrate v3 filesystem workspace to v4 SQLite database.

Usage:
    python scripts/migrate_v3_to_v4.py /path/to/workspace /path/to/output.db
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.migrate import migrate_workspace_to_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Migrate v3 workspace to v4 SQLite")
    parser.add_argument("workspace", type=Path, help="v3 workspace directory")
    parser.add_argument("database", type=Path, help="v4 SQLite database path")
    args = parser.parse_args()

    if not args.workspace.exists():
        print(f"Error: workspace {args.workspace} does not exist")
        sys.exit(1)

    ws = Workspace(args.workspace)
    store = Store(args.database)

    summary = migrate_workspace_to_store(ws, store)
    print(f"Migration complete: {summary}")

    store.close()


if __name__ == "__main__":
    main()
