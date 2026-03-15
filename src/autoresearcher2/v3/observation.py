"""Observation: immutable reality contact record.

An observation is evidence, not understanding. It records what was done,
what came out, and under what conditions. Append-only — never mutated.
"""

import json
import time
import uuid
from pathlib import Path


class Observation:
    """Immutable record of a reality contact moment."""

    def __init__(
        self,
        intervention_type,
        intervention_spec,
        outcome_success,
        outcome_metrics=None,
        error=None,
        wall_time_s=None,
        compute_cost=None,
        worker_id=None,
        raw_log=None,
        id=None,
    ):
        self.id = id or f"obs_{uuid.uuid4().hex[:8]}"
        self.created_at = time.time()
        self.intervention_type = intervention_type
        self.intervention_spec = intervention_spec
        self.outcome_metrics = outcome_metrics
        self.outcome_success = outcome_success
        self.error = error
        self.wall_time_s = wall_time_s
        self.compute_cost = compute_cost
        self.worker_id = worker_id
        self.raw_log = raw_log
        self.project_id = None

    def to_dict(self):
        """Serialize to dict."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "intervention_type": self.intervention_type,
            "intervention_spec": self.intervention_spec,
            "outcome_metrics": self.outcome_metrics,
            "outcome_success": self.outcome_success,
            "error": self.error,
            "wall_time_s": self.wall_time_s,
            "compute_cost": self.compute_cost,
            "worker_id": self.worker_id,
            "raw_log": self.raw_log,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        obs = cls(
            id=data["id"],
            intervention_type=data["intervention_type"],
            intervention_spec=data["intervention_spec"],
            outcome_metrics=data.get("outcome_metrics"),
            outcome_success=data["outcome_success"],
            error=data.get("error"),
            wall_time_s=data.get("wall_time_s"),
            compute_cost=data.get("compute_cost"),
            worker_id=data.get("worker_id"),
            raw_log=data.get("raw_log"),
        )
        obs.created_at = data.get("created_at", time.time())
        return obs

    def save(self, path):
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path):
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
