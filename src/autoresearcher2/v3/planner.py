"""Planner: generator + critic + world model update loop.

Polling loop that:
1. If new results in done/: run orientation step (update world model)
2. If backlog + todo < threshold: run generator
3. If backlog has items: run critic, promote top-N to todo
"""

import logging
import time

from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.orientation import orient
from autoresearcher2.v3.generator import generate_proposals
from autoresearcher2.v3.critic import critique_proposals

logger = logging.getLogger(__name__)


class Planner:
    """Manages the Orient and Decide phases of the OODA loop."""

    def __init__(
        self,
        workspace: Workspace,
        llm_call_fn,
        min_queue_size: int = 5,
        n_proposals: int = 5,
        n_select: int = 2,
    ):
        self.workspace = workspace
        self.llm_call_fn = llm_call_fn
        self.min_queue_size = min_queue_size
        self.n_proposals = n_proposals
        self.n_select = n_select
        self._processed_observations = set()

    def tick(self) -> dict:
        """Run one planning cycle. Returns summary of what happened."""
        summary = {"oriented": 0, "generated": 0, "promoted": 0}

        # Phase 1: Orient — process new observations
        done_proposals = self.workspace.list_proposals("done")
        for p in done_proposals:
            if p.observation_id and p.observation_id not in self._processed_observations:
                obs_path = self.workspace.results_dir / f"{p.observation_id}.json"
                if obs_path.exists():
                    from autoresearcher2.v3.observation import Observation
                    obs = Observation.load(obs_path)
                    wm = self.workspace.load_world_model()
                    delta = orient(wm, obs, self.llm_call_fn)
                    if delta:
                        self.workspace.save_world_model(wm)
                        summary["oriented"] += 1
                    self._processed_observations.add(p.observation_id)

        # Phase 2: Generate — if queue is low, produce proposals
        backlog_count = self.workspace.count_proposals("backlog")
        todo_count = self.workspace.count_proposals("todo")
        if backlog_count + todo_count < self.min_queue_size:
            wm = self.workspace.load_world_model()
            proposals = generate_proposals(
                wm, n_proposals=self.n_proposals, llm_call_fn=self.llm_call_fn
            )
            for p in proposals:
                self.workspace.save_proposal(p)
            summary["generated"] = len(proposals)

        # Phase 3: Critique — if backlog has items, rank and promote
        backlog = self.workspace.list_proposals("backlog")
        if backlog:
            wm = self.workspace.load_world_model()
            accepted = critique_proposals(
                wm, backlog, n_select=self.n_select, llm_call_fn=self.llm_call_fn
            )
            for p in accepted:
                self.workspace.move_proposal(p, "todo")
                summary["promoted"] += 1

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
