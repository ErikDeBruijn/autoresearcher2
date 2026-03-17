"""Worker: claims todo items and executes interventions.

Polling loop that:
1. Claim highest-ranked todo item
2. Execute the intervention
3. Record observation
4. Move proposal to done
"""

import logging
import time

from autoresearcher2.v3.store import Store
from autoresearcher2.v3.observation import Observation
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.research.environment import Environment

logger = logging.getLogger(__name__)


class Worker:
    """Executes experiments from the todo queue."""

    def __init__(
        self,
        store: Store,
        execute_fn=None,
        worker_id: str = "worker_0",
        project_ids: list[str] | None = None,
        post_complete_fn=None,
    ):
        self.store = store
        self.execute_fn = execute_fn
        self.worker_id = worker_id
        self.project_ids = project_ids
        self.post_complete_fn = post_complete_fn

    def tick(self) -> dict | None:
        """Try to claim and execute one todo item. Returns observation dict or None."""
        proposal = self.store.claim_next_todo(self.worker_id, project_ids=self.project_ids)
        if proposal is None:
            return None

        logger.info("[%s] Claimed %s: %s", self.worker_id, proposal.id, proposal.intent)

        start = time.time()
        try:
            result = self.execute_fn(proposal)
            wall_time = time.time() - start

            obs = Observation(
                intervention_type=proposal.intervention_type,
                intervention_spec=proposal.intervention_spec,
                outcome_metrics=result.get("metrics", {}),
                outcome_success=True,
                wall_time_s=wall_time,
                compute_cost=result.get("compute_cost"),
                worker_id=self.worker_id,
                raw_log=result.get("raw_log"),
                energy_kwh=result.get("energy_kwh"),
                cost_eur=result.get("cost_eur"),
                avg_power_w=result.get("avg_power_w"),
            )
            if result.get("artifact_paths"):
                obs.artifact_paths = result["artifact_paths"]
        except Exception as e:
            wall_time = time.time() - start
            logger.error("[%s] Execution failed: %s", self.worker_id, e)
            obs = Observation(
                intervention_type=proposal.intervention_type,
                intervention_spec=proposal.intervention_spec,
                outcome_success=False,
                error=str(e),
                wall_time_s=wall_time,
                worker_id=self.worker_id,
            )

        self.store.complete_proposal(proposal, obs)
        logger.info("[%s] Completed %s: success=%s, %.1fs",
                    self.worker_id, proposal.id, obs.outcome_success, wall_time)

        if self.post_complete_fn and obs.outcome_success:
            try:
                self.post_complete_fn(proposal, obs)
            except Exception:
                logger.warning("[%s] post_complete_fn failed", self.worker_id, exc_info=True)

        return obs.to_dict()

    def run(self, poll_interval: float = 30.0, max_ticks: int = None):
        """Run the worker loop.

        Args:
            poll_interval: Seconds between checks when idle
            max_ticks: Stop after this many ticks (None = run forever)
        """
        tick_count = 0
        while max_ticks is None or tick_count < max_ticks:
            result = self.tick()
            tick_count += 1

            if result is None:
                # Nothing to do, wait
                if max_ticks is None or tick_count < max_ticks:
                    time.sleep(poll_interval)
            # If we executed something, immediately check for more (no sleep)
