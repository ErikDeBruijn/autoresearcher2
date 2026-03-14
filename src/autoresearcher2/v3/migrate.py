"""Migrate v3 filesystem workspace to v4 SQLite store.

Usage:
    from autoresearcher2.v3.migrate import migrate_workspace_to_store
    migrate_workspace_to_store(workspace, store)
"""
import json
import logging

from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.store import Store

logger = logging.getLogger(__name__)


def migrate_workspace_to_store(workspace: Workspace, store: Store) -> dict:
    """Migrate all data from filesystem workspace to SQLite store.

    Returns summary of migrated items.
    """
    store.init()
    summary = {"observations": 0, "world_model_versions": 0, "proposals": 0}

    # Layer 1: Observations
    for obs in workspace.list_observations():
        try:
            store.save_observation(obs)
            summary["observations"] += 1
        except Exception as e:
            logger.warning("Skipping observation %s: %s", obs.id, e)

    # Layer 2: World Model (current + history)
    # Migrate history first
    for path in sorted(workspace.history_dir.glob("world_model_v*.json")):
        try:
            data = json.loads(path.read_text())
            from autoresearcher2.v3.world_model import WorldModel
            wm = WorldModel.from_dict(data)
            store.save_world_model(wm)
            summary["world_model_versions"] += 1
        except Exception as e:
            logger.warning("Skipping history %s: %s", path.name, e)

    # Then current
    wm = workspace.load_world_model()
    store.save_world_model(wm)
    summary["world_model_versions"] += 1

    # Layer 3: Proposals (all stages)
    for stage in ("backlog", "todo", "running", "done"):
        for proposal in workspace.list_proposals(stage):
            try:
                store.save_proposal(proposal)
                summary["proposals"] += 1
            except Exception as e:
                logger.warning("Skipping proposal %s: %s", proposal.id, e)

    logger.info("Migration complete: %s", summary)
    return summary
