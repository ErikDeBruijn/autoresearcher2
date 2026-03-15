"""Workspace: filesystem-based research workspace manager.

Manages the directory structure for v3's kanban-style workflow:
  proposals/backlog/  — generator output, not yet reviewed
  proposals/todo/     — critic-approved, ready for execution
  proposals/running/  — currently being executed
  proposals/done/     — completed, linked to observation
  results/            — observations (reality contact)
  world_model.json    — current epistemic state
  world_model_history/ — previous versions for audit
"""

import json
from pathlib import Path

from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


class Workspace:
    """Filesystem-based research workspace."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.backlog_dir = self.root / "proposals" / "backlog"
        self.todo_dir = self.root / "proposals" / "todo"
        self.running_dir = self.root / "proposals" / "running"
        self.done_dir = self.root / "proposals" / "done"
        self.results_dir = self.root / "results"
        self.world_model_path = self.root / "world_model.json"
        self.history_dir = self.root / "world_model_history"

    def init(self):
        """Create workspace directory structure."""
        for d in [self.backlog_dir, self.todo_dir, self.running_dir,
                  self.done_dir, self.results_dir, self.history_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # Create initial world model if it doesn't exist
        if not self.world_model_path.exists():
            wm = WorldModel()
            wm.save(self.world_model_path)

    def load_world_model(self) -> WorldModel:
        """Load current world model from disk."""
        return WorldModel.load(self.world_model_path)

    def save_world_model(self, wm: WorldModel):
        """Save world model with history."""
        wm.save(self.world_model_path, history_dir=self.history_dir)

    def save_proposal(self, proposal: Proposal):
        """Save proposal to the appropriate stage directory."""
        stage_dir = self._stage_dir(proposal.status)
        proposal.save(stage_dir / f"{proposal.id}.json")

    def move_proposal(self, proposal: Proposal, new_status: str):
        """Move proposal from current stage to new stage."""
        old_dir = self._stage_dir(proposal.status)
        old_path = old_dir / f"{proposal.id}.json"
        if old_path.exists():
            old_path.unlink()
        proposal.promote(new_status)
        self.save_proposal(proposal)

    def save_observation(self, observation: Observation):
        """Save observation to results directory."""
        observation.save(self.results_dir / f"{observation.id}.json")

    def list_proposals(self, stage: str) -> list[Proposal]:
        """List all proposals in a given stage."""
        stage_dir = self._stage_dir(stage)
        proposals = []
        for path in sorted(stage_dir.glob("*.json")):
            proposals.append(Proposal.load(path))
        return proposals

    def load_observation(self, obs_id: str) -> Observation | None:
        """Load a single observation by ID."""
        path = self.results_dir / f"{obs_id}.json"
        if path.exists():
            return Observation.load(path)
        return None

    def list_observations(self) -> list[Observation]:
        """List all observations."""
        observations = []
        for path in sorted(self.results_dir.glob("*.json")):
            observations.append(Observation.load(path))
        return observations

    def count_proposals(self, stage: str) -> int:
        """Count proposals in a given stage."""
        return len(list(self._stage_dir(stage).glob("*.json")))

    def claim_next_todo(self, worker_id: str) -> Proposal | None:
        """Atomically claim the highest-ranked todo item for a worker.

        Moves the file from todo/ to running/ as an atomic operation.
        Returns None if no items available.
        """
        proposals = self.list_proposals("todo")
        if not proposals:
            return None

        # Sort by critic rank (lowest = highest priority)
        proposals.sort(key=lambda p: (p.critic or {}).get("rank", 999))
        proposal = proposals[0]

        # Atomic claim: rename from todo to running
        old_path = self.todo_dir / f"{proposal.id}.json"
        new_path = self.running_dir / f"{proposal.id}.json"

        try:
            old_path.rename(new_path)
        except FileNotFoundError:
            # Another worker claimed it first
            return None

        proposal.promote("running")
        proposal.save(new_path)
        return proposal

    def complete_proposal(self, proposal: Proposal, observation: Observation):
        """Mark a proposal as done, linking to its observation."""
        # Save observation
        self.save_observation(observation)

        # Move proposal to done
        old_path = self.running_dir / f"{proposal.id}.json"
        if old_path.exists():
            old_path.unlink()
        proposal.complete(observation.id)
        self.save_proposal(proposal)

    def _stage_dir(self, stage: str) -> Path:
        dirs = {
            "backlog": self.backlog_dir,
            "todo": self.todo_dir,
            "running": self.running_dir,
            "done": self.done_dir,
        }
        if stage not in dirs:
            raise ValueError(f"Unknown stage: {stage}")
        return dirs[stage]
