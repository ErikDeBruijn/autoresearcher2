"""Proposal: rationale-first experiment proposal.

Cognitive order: epistemic intent → rationale → expected learning →
intervention → executable spec. Not action-first.
"""

import json
import time
import uuid
from pathlib import Path


class Proposal:
    """A research proposal, rationale-first."""

    def __init__(
        self,
        intent,
        rationale,
        expected_learning,
        intervention_type,
        intervention_spec,
        estimated_cost=None,
        id=None,
        status="backlog",
    ):
        self.id = id or f"prop_{uuid.uuid4().hex[:8]}"
        self.created_at = time.time()
        self.status = status

        # Rationale-first (cognition)
        self.intent = intent
        self.rationale = rationale
        self.expected_learning = expected_learning

        # Intervention (execution)
        self.intervention_type = intervention_type
        self.intervention_spec = intervention_spec
        self.estimated_cost = estimated_cost or {}

        # Critic metadata (set later)
        self.critic = None

        # Lifecycle
        self.promoted_at = None
        self.started_at = None
        self.finished_at = None
        self.observation_id = None

    def set_critic_decision(self, decision, rank, rationale):
        """Set the critic's ordinal ranking and decision."""
        self.critic = {
            "decision": decision,
            "rank": rank,
            "rationale": rationale,
        }

    def promote(self, new_status):
        """Move proposal to a new stage."""
        self.status = new_status
        if new_status == "todo":
            self.promoted_at = time.time()
        elif new_status == "running":
            self.started_at = time.time()

    def complete(self, observation_id):
        """Mark proposal as done, linking to the observation."""
        self.status = "done"
        self.finished_at = time.time()
        self.observation_id = observation_id

    def to_dict(self):
        """Serialize preserving rationale-first field order."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "status": self.status,
            # Rationale-first
            "intent": self.intent,
            "rationale": self.rationale,
            "expected_learning": self.expected_learning,
            # Intervention
            "intervention_type": self.intervention_type,
            "intervention_spec": self.intervention_spec,
            "estimated_cost": self.estimated_cost,
            # Critic
            "critic": self.critic,
            # Lifecycle
            "promoted_at": self.promoted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "observation_id": self.observation_id,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        p = cls(
            id=data["id"],
            status=data.get("status", "backlog"),
            intent=data["intent"],
            rationale=data["rationale"],
            expected_learning=data.get("expected_learning", ""),
            intervention_type=data["intervention_type"],
            intervention_spec=data["intervention_spec"],
            estimated_cost=data.get("estimated_cost", {}),
        )
        p.created_at = data.get("created_at", time.time())
        p.critic = data.get("critic")
        p.promoted_at = data.get("promoted_at")
        p.started_at = data.get("started_at")
        p.finished_at = data.get("finished_at")
        p.observation_id = data.get("observation_id")
        return p

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
