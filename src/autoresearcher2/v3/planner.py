"""Planner: generator + critic + world model update loop.

Pull-based pipeline:
1. Orient: process new observations -> update world model
2. Critique: if todo is low, promote from backlog
3. Generate: if backlog is low, produce new proposals
4. Critique again: ensure todo stays stocked after generation

Workers pull from todo. When todo empties, critic promotes from backlog.
When backlog empties, generator creates new proposals. This ensures
workers are never idle while the system can generate work.

Works with Store (SQLite backend).
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
        store,
        llm_call_fn,
        min_queue_size: int = 5,
        min_todo: int = 2,
        n_proposals: int = 5,
        n_select: int = 2,
        domain=None,
        project_id: str = None,
    ):
        self.store = store
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
        done_proposals = self.store.list_proposals("done", project_id=pid)
        for p in done_proposals:
            if p.observation_id and p.observation_id not in self._processed_observations:
                obs = self.store.load_observation(p.observation_id)
                if obs is not None:
                    self.store.set_pipeline_activity("orienting", pid, p.id)
                    wm = self.store.load_world_model(project_id=pid)
                    delta = orient(wm, obs, self.llm_call_fn, domain=self.domain)
                    if delta:
                        self.store.save_world_model(
                            wm, trigger_obs_id=p.observation_id, delta=delta,
                            project_id=pid,
                        )
                        summary["oriented"] += 1
                    self.store.mark_reviewed(p.id)
                    self._processed_observations.add(p.observation_id)
                    self.store.clear_pipeline_activity()

        # Phase 2: Critique — if todo is low, promote from backlog
        todo_count = self.store.count_proposals("todo", project_id=pid)
        if todo_count < self.min_todo:
            backlog = self.store.list_proposals("backlog", project_id=pid)
            if backlog:
                self.store.set_pipeline_activity("critiquing", pid)
                wm = self.store.load_world_model(project_id=pid)
                accepted = critique_proposals(
                    wm, backlog, n_select=self.n_select, llm_call_fn=self.llm_call_fn,
                    domain=self.domain,
                )
                for p in accepted:
                    self.store.move_proposal(p, "todo")
                    summary["promoted"] += 1
                self.store.clear_pipeline_activity()

        # Phase 3: Generate — if backlog is low, produce new proposals
        backlog_count = self.store.count_proposals("backlog", project_id=pid)
        if backlog_count < self.min_queue_size:
            self.store.set_pipeline_activity("generating", pid)
            wm = self.store.load_world_model(project_id=pid)
            proposals = generate_proposals(
                wm, n_proposals=self.n_proposals, llm_call_fn=self.llm_call_fn,
                domain=self.domain,
            )
            for p in proposals:
                self.store.save_proposal(p, project_id=pid)
            summary["generated"] = len(proposals)
            self.store.clear_pipeline_activity()

        # Phase 4: Critique again — if generation just restocked backlog, promote immediately
        if summary["generated"] > 0 and summary["promoted"] == 0:
            todo_count = self.store.count_proposals("todo", project_id=pid)
            if todo_count < self.min_todo:
                backlog = self.store.list_proposals("backlog", project_id=pid)
                if backlog:
                    self.store.set_pipeline_activity("critiquing", pid)
                    wm = self.store.load_world_model(project_id=pid)
                    accepted = critique_proposals(
                        wm, backlog, n_select=self.n_select, llm_call_fn=self.llm_call_fn
                    )
                    for p in accepted:
                        self.store.move_proposal(p, "todo")
                        summary["promoted"] += 1
                    self.store.clear_pipeline_activity()

        return summary

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
