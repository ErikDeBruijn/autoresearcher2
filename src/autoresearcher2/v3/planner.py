"""Planner: generator + critic + world model update loop.

Pull-based pipeline:
1. Orient: process new observations → update world model
2. Critique: if todo is low, promote from backlog
3. Generate: if backlog is low, produce new proposals
4. Critique again: ensure todo stays stocked after generation

Workers pull from todo. When todo empties, critic promotes from backlog.
When backlog empties, generator creates new proposals. This ensures
workers are never idle while the system can generate work.

Works with both Workspace (filesystem, v3) and Store (SQLite, v4).
"""

import logging
import time

from autoresearcher2.v3.orientation import orient
from autoresearcher2.v3.generator import generate_proposals
from autoresearcher2.v3.critic import critique_proposals

logger = logging.getLogger(__name__)


class Planner:
    """Manages the Orient and Decide phases of the OODA loop."""

    def __init__(
        self,
        workspace,
        llm_call_fn,
        min_queue_size: int = 5,
        min_todo: int = 2,
        n_proposals: int = 5,
        n_select: int = 2,
        domain=None,
        project_id: str = None,
    ):
        self.workspace = workspace
        self.llm_call_fn = llm_call_fn
        self.min_queue_size = min_queue_size
        self.min_todo = min_todo
        self.n_proposals = n_proposals
        self.n_select = n_select
        self.domain = domain
        self.project_id = project_id
        self._processed_observations = set()

    def tick(self) -> dict:
        """Run one planning cycle. Pull-based: ensure todo and backlog stay stocked."""
        summary = {"oriented": 0, "generated": 0, "promoted": 0}
        pid = self.project_id

        # Phase 1: Orient — process new observations
        done_proposals = self._list_proposals("done")
        for p in done_proposals:
            if p.observation_id and p.observation_id not in self._processed_observations:
                obs = self.workspace.load_observation(p.observation_id)
                if obs is not None:
                    wm = self._load_world_model()
                    delta = orient(wm, obs, self.llm_call_fn)
                    if delta:
                        self._save_world_model(wm, trigger_obs_id=p.observation_id, delta=delta)
                        summary["oriented"] += 1
                    if hasattr(self.workspace, 'mark_reviewed'):
                        self.workspace.mark_reviewed(p.id)
                    self._processed_observations.add(p.observation_id)

        # Phase 2: Critique — if todo is low, promote from backlog
        todo_count = self._count_proposals("todo")
        if todo_count < self.min_todo:
            backlog = self._list_proposals("backlog")
            if backlog:
                wm = self._load_world_model()
                accepted = critique_proposals(
                    wm, backlog, n_select=self.n_select, llm_call_fn=self.llm_call_fn
                )
                for p in accepted:
                    self.workspace.move_proposal(p, "todo")
                    summary["promoted"] += 1

        # Phase 3: Generate — if backlog is low, produce new proposals
        backlog_count = self._count_proposals("backlog")
        if backlog_count < self.min_queue_size:
            wm = self._load_world_model()
            proposals = generate_proposals(
                wm, n_proposals=self.n_proposals, llm_call_fn=self.llm_call_fn,
                domain=self.domain,
            )
            for p in proposals:
                self._save_proposal(p)
            summary["generated"] = len(proposals)

        # Phase 4: Critique again — if generation just restocked backlog, promote immediately
        if summary["generated"] > 0 and summary["promoted"] == 0:
            todo_count = self._count_proposals("todo")
            if todo_count < self.min_todo:
                backlog = self._list_proposals("backlog")
                if backlog:
                    wm = self._load_world_model()
                    accepted = critique_proposals(
                        wm, backlog, n_select=self.n_select, llm_call_fn=self.llm_call_fn
                    )
                    for p in accepted:
                        self.workspace.move_proposal(p, "todo")
                        summary["promoted"] += 1

        return summary

    def _list_proposals(self, stage: str):
        """List proposals, scoped to project if set."""
        try:
            return self.workspace.list_proposals(stage, project_id=self.project_id)
        except TypeError:
            return self.workspace.list_proposals(stage)

    def _count_proposals(self, stage: str):
        """Count proposals, scoped to project if set."""
        try:
            return self.workspace.count_proposals(stage, project_id=self.project_id)
        except TypeError:
            return self.workspace.count_proposals(stage)

    def _load_world_model(self):
        """Load world model, scoped to project if set."""
        try:
            return self.workspace.load_world_model(project_id=self.project_id)
        except TypeError:
            return self.workspace.load_world_model()

    def _save_proposal(self, proposal):
        """Save proposal with project_id if set."""
        try:
            self.workspace.save_proposal(proposal, project_id=self.project_id)
        except TypeError:
            self.workspace.save_proposal(proposal)

    def _save_world_model(self, wm, trigger_obs_id=None, delta=None):
        """Save world model, passing delta info if the backend supports it (Store)."""
        try:
            self.workspace.save_world_model(
                wm, trigger_obs_id=trigger_obs_id, delta=delta,
                project_id=self.project_id,
            )
        except TypeError:
            # Workspace (filesystem) doesn't accept extra kwargs
            self.workspace.save_world_model(wm)

    def run(self, poll_interval: float = 60.0, max_ticks: int = None):
        """Run the planner loop.

        Args:
            poll_interval: Seconds between ticks
            max_ticks: Stop after this many ticks (None = run forever)
        """
        tick_count = 0
        while max_ticks is None or tick_count < max_ticks:
            try:
                summary = self.tick()
                if any(v > 0 for v in summary.values()):
                    logger.info("Planner tick %d: %s", tick_count, summary)
            except Exception:
                logger.error("Planner tick failed", exc_info=True)

            tick_count += 1
            if max_ticks is None or tick_count < max_ticks:
                time.sleep(poll_interval)
