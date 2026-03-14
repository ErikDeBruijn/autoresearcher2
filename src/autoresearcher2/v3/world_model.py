"""World model: structured epistemic state of the research system.

The world model is the accumulated, testable intuition of the system.
It is not a summary, document, or prompt — it is a structured epistemic
state that gets updated via evidence-anchored deltas.

JSON serialization is the persistence form, not the essence.
"""

import json
import shutil
import time
from pathlib import Path


class WorldModel:
    """Structured epistemic state: beliefs, expectations, tensions, costs."""

    def __init__(self):
        self.version = 0
        self.updated_at = time.time()
        self.beliefs = []
        self.expectations = []
        self.tensions = []
        self.cost_beliefs = {}
        self.probe_fidelity = []
        self.salience = {
            "high_learntropy": [],
            "unresolved_tensions": [],
            "stale_beliefs": [],
        }
        self._next_belief_id = 1
        self._next_tension_id = 1

    def add_belief(self, claim, confidence, evidence_for, evidence_against=None):
        """Add a new belief to the world model. Returns the belief ID."""
        bid = f"B{self._next_belief_id}"
        self._next_belief_id += 1
        self.beliefs.append({
            "id": bid,
            "claim": claim,
            "confidence": confidence,
            "evidence_for": list(evidence_for),
            "evidence_against": list(evidence_against or []),
            "first_held": time.time(),
            "last_tested": time.time(),
        })
        return bid

    def add_expectation(self, condition, prediction, confidence, basis):
        """Add an expectation: if condition then prediction."""
        self.expectations.append({
            "if": condition,
            "then": prediction,
            "confidence": confidence,
            "basis": list(basis),
        })

    def add_tension(self, belief_ids, nature, salience):
        """Add a tension between beliefs. Returns the tension ID."""
        tid = f"T{self._next_tension_id}"
        self._next_tension_id += 1
        self.tensions.append({
            "id": tid,
            "beliefs": list(belief_ids),
            "nature": nature,
            "salience": salience,
        })
        return tid

    def apply_delta(self, delta):
        """Apply a structured update-delta to the world model.

        The delta is produced by the LLM orientation step. It contains
        specific changes to beliefs, tensions, costs, etc. — not a rewrite.
        """
        # Revise existing beliefs
        for rev in delta.get("beliefs_revised", []):
            for b in self.beliefs:
                if b["id"] == rev["id"]:
                    if "new_confidence" in rev:
                        b["confidence"] = rev["new_confidence"]
                    if "new_evidence_for" in rev:
                        b["evidence_for"].extend(rev["new_evidence_for"])
                    if "new_evidence_against" in rev:
                        b["evidence_against"].extend(rev["new_evidence_against"])
                    b["last_tested"] = time.time()

        # Add new beliefs
        for added in delta.get("beliefs_added", []):
            self.add_belief(
                claim=added["claim"],
                confidence=added["confidence"],
                evidence_for=added.get("evidence_for", []),
                evidence_against=added.get("evidence_against", []),
            )

        # Retire beliefs
        retired_ids = {r["id"] for r in delta.get("beliefs_retired", [])}
        self.beliefs = [b for b in self.beliefs if b["id"] not in retired_ids]

        # Add tensions
        for t in delta.get("tensions_added", []):
            self.add_tension(
                belief_ids=t.get("beliefs", []),
                nature=t.get("nature", ""),
                salience=t.get("salience", "medium"),
            )

        # Resolve tensions
        resolved_ids = {r["id"] for r in delta.get("tensions_resolved", [])}
        self.tensions = [t for t in self.tensions if t["id"] not in resolved_ids]

        # Update cost beliefs
        self.cost_beliefs.update(delta.get("cost_beliefs_updated", {}))

        # Bump version
        self.version += 1
        self.updated_at = time.time()

    def to_dict(self):
        """Serialize to dict for JSON persistence."""
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "beliefs": self.beliefs,
            "expectations": self.expectations,
            "tensions": self.tensions,
            "cost_beliefs": self.cost_beliefs,
            "probe_fidelity": self.probe_fidelity,
            "salience": self.salience,
            "_next_belief_id": self._next_belief_id,
            "_next_tension_id": self._next_tension_id,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        wm = cls()
        wm.version = data["version"]
        wm.updated_at = data["updated_at"]
        wm.beliefs = data["beliefs"]
        wm.expectations = data.get("expectations", [])
        wm.tensions = data.get("tensions", [])
        wm.cost_beliefs = data.get("cost_beliefs", {})
        wm.probe_fidelity = data.get("probe_fidelity", [])
        wm.salience = data.get("salience", {
            "high_learntropy": [],
            "unresolved_tensions": [],
            "stale_beliefs": [],
        })
        wm._next_belief_id = data.get("_next_belief_id", len(wm.beliefs) + 1)
        wm._next_tension_id = data.get("_next_tension_id", len(wm.tensions) + 1)
        return wm

    def save(self, path, history_dir=None):
        """Save to JSON file. Optionally archive previous version."""
        path = Path(path)
        if history_dir and path.exists():
            history_dir = Path(history_dir)
            history_dir.mkdir(parents=True, exist_ok=True)
            old = json.loads(path.read_text())
            old_version = old.get("version", 0)
            shutil.copy2(path, history_dir / f"world_model_v{old_version}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path):
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
