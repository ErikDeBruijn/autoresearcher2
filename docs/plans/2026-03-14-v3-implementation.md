# v3.0 Generator-Critic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement generator-critic MVP: generate more proposals than executed, evaluate before acting, world model as structured epistemic state.

**Architecture:** Three LLM roles (orientation, generator, critic) operating on a shared world_model.json. Proposals are rationale-first JSON. LLM called via `claude -p` over SSH (existing pattern). Filesystem-first persistence.

**Tech Stack:** Python, pytest, JSON files, `claude -p` via SSH

---

### Task 1: World Model data structure

**Files:**
- Create: `src/autoresearcher2/v3/world_model.py`
- Test: `tests/v3/test_world_model.py`

**Step 1: Write the failing test**

```python
# tests/v3/test_world_model.py
import pytest
from autoresearcher2.v3.world_model import WorldModel

def test_empty_world_model():
    wm = WorldModel()
    assert wm.version == 0
    assert wm.beliefs == []
    assert wm.expectations == []
    assert wm.tensions == []
    assert wm.cost_beliefs == {}

def test_add_belief():
    wm = WorldModel()
    wm.add_belief(
        claim="learning_rate is the dominant factor",
        confidence=0.5,
        evidence_for=["obs_001"],
    )
    assert len(wm.beliefs) == 1
    assert wm.beliefs[0]["claim"] == "learning_rate is the dominant factor"
    assert wm.beliefs[0]["confidence"] == 0.5
    assert wm.beliefs[0]["id"].startswith("B")

def test_serialize_roundtrip():
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.7, evidence_for=[])
    data = wm.to_dict()
    wm2 = WorldModel.from_dict(data)
    assert wm2.beliefs == wm.beliefs

def test_save_and_load(tmp_path):
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.7, evidence_for=[])
    path = tmp_path / "world_model.json"
    wm.save(path)
    wm2 = WorldModel.load(path)
    assert wm2.beliefs[0]["claim"] == "test"

def test_apply_delta():
    wm = WorldModel()
    wm.add_belief(claim="lr matters", confidence=0.5, evidence_for=["obs_001"])
    bid = wm.beliefs[0]["id"]
    delta = {
        "beliefs_added": [{"claim": "depth has diminishing returns", "confidence": 0.3, "evidence_for": ["obs_002"]}],
        "beliefs_revised": [{"id": bid, "new_confidence": 0.8, "new_evidence_for": ["obs_003"], "reason": "confirmed by obs_003"}],
        "beliefs_retired": [],
        "tensions_added": [],
        "tensions_resolved": [],
        "cost_beliefs_updated": {"config_change": {"wall_time_s": 300}},
    }
    old_version = wm.version
    wm.apply_delta(delta)
    assert wm.version == old_version + 1
    assert len(wm.beliefs) == 2
    assert wm.beliefs[0]["confidence"] == 0.8
    assert "obs_003" in wm.beliefs[0]["evidence_for"]

def test_save_with_history(tmp_path):
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.5, evidence_for=[])
    wm.save(tmp_path / "world_model.json")
    wm.apply_delta({"beliefs_revised": [{"id": wm.beliefs[0]["id"], "new_confidence": 0.9, "reason": "updated"}]})
    wm.save(tmp_path / "world_model.json", history_dir=tmp_path / "history")
    assert (tmp_path / "history" / "world_model_v0.json").exists()
```

**Step 2:** Run tests to verify they fail.

**Step 3: Implement WorldModel**

```python
# src/autoresearcher2/v3/world_model.py
"""World model: structured epistemic state of the research system."""
import json
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field

class WorldModel:
    def __init__(self):
        self.version = 0
        self.updated_at = time.time()
        self.beliefs = []
        self.expectations = []
        self.tensions = []
        self.cost_beliefs = {}
        self.probe_fidelity = []
        self.salience = {"high_learntropy": [], "unresolved_tensions": [], "stale_beliefs": []}
        self._next_belief_id = 1
        self._next_tension_id = 1

    def add_belief(self, claim, confidence, evidence_for, evidence_against=None):
        bid = f"B{self._next_belief_id}"
        self._next_belief_id += 1
        self.beliefs.append({
            "id": bid,
            "claim": claim,
            "confidence": confidence,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against or [],
            "first_held": time.time(),
            "last_tested": time.time(),
        })
        return bid

    def apply_delta(self, delta):
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
            )
        # Retire beliefs
        retired_ids = {r["id"] for r in delta.get("beliefs_retired", [])}
        self.beliefs = [b for b in self.beliefs if b["id"] not in retired_ids]
        # Tensions
        for t in delta.get("tensions_added", []):
            tid = f"T{self._next_tension_id}"
            self._next_tension_id += 1
            t["id"] = tid
            self.tensions.append(t)
        resolved_ids = {r["id"] for r in delta.get("tensions_resolved", [])}
        self.tensions = [t for t in self.tensions if t["id"] not in resolved_ids]
        # Cost beliefs
        self.cost_beliefs.update(delta.get("cost_beliefs_updated", {}))
        # Bump version
        self.version += 1
        self.updated_at = time.time()

    def to_dict(self):
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
        wm = cls()
        wm.version = data["version"]
        wm.updated_at = data["updated_at"]
        wm.beliefs = data["beliefs"]
        wm.expectations = data.get("expectations", [])
        wm.tensions = data.get("tensions", [])
        wm.cost_beliefs = data.get("cost_beliefs", {})
        wm.probe_fidelity = data.get("probe_fidelity", [])
        wm.salience = data.get("salience", {"high_learntropy": [], "unresolved_tensions": [], "stale_beliefs": []})
        wm._next_belief_id = data.get("_next_belief_id", len(wm.beliefs) + 1)
        wm._next_tension_id = data.get("_next_tension_id", len(wm.tensions) + 1)
        return wm

    def save(self, path, history_dir=None):
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
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
```

**Step 4:** Run tests, verify they pass.

**Step 5:** Commit: `git commit -m "v3.0: add WorldModel data structure with delta-based updates"`

---

### Task 2: Proposal data structure

**Files:**
- Create: `src/autoresearcher2/v3/proposal.py`
- Test: `tests/v3/test_proposal.py`

**Step 1: Write the failing test**

```python
# tests/v3/test_proposal.py
import pytest
from autoresearcher2.v3.proposal import Proposal

def test_create_proposal():
    p = Proposal(
        intent="Test whether lr generalizes across depths",
        rationale="lr=0.04 works at depth=8, untested at depth=10",
        expected_learning="Clarifies lr×depth interaction",
        intervention_type="config_change",
        intervention_spec={"DEPTH": "10", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.2"},
        estimated_cost={"cost_to_test": "~5 min GPU", "cheaper_probe": None},
    )
    assert p.id.startswith("prop_")
    assert p.status == "backlog"
    assert p.intent == "Test whether lr generalizes across depths"

def test_proposal_serialize_roundtrip():
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    data = p.to_dict()
    p2 = Proposal.from_dict(data)
    assert p2.id == p.id
    assert p2.intent == p.intent

def test_proposal_save_load(tmp_path):
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    path = tmp_path / f"{p.id}.json"
    p.save(path)
    p2 = Proposal.load(path)
    assert p2.id == p.id

def test_proposal_set_critic_decision():
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.set_critic_decision(decision="accept", rank=1, rationale="High value, low cost")
    assert p.critic["decision"] == "accept"
    assert p.critic["rank"] == 1
```

**Step 2:** Run tests to verify they fail.

**Step 3: Implement Proposal**

A simple dataclass with JSON serialization. Keep it minimal.

**Step 4:** Run tests, verify they pass.

**Step 5:** Commit: `git commit -m "v3.0: add Proposal data structure"`

---

### Task 3: Observation (reality contact) data structure

**Files:**
- Create: `src/autoresearcher2/v3/observation.py`
- Test: `tests/v3/test_observation.py`

**Step 1:** Write tests for creating, serializing, and loading observations.

**Step 2:** Implement Observation class with fields: id, intervention_type, intervention_spec, outcome_metrics, outcome_success, wall_time_s, compute_cost, raw_log.

**Step 3:** Commit: `git commit -m "v3.0: add Observation data structure (reality contact)"`

---

### Task 4: LLM call wrapper

**Files:**
- Create: `src/autoresearcher2/v3/llm_call.py`
- Test: `tests/v3/test_llm_call.py`

The existing `llm/proposal.py` calls `claude -p` via SSH. We need a reusable wrapper.

**Step 1:** Write test that mocks SSH and verifies prompt is sent, JSON response is parsed.

**Step 2:** Implement `call_llm(prompt: str, output_schema: dict = None, ssh_host: str, ssh_key: str) -> dict` that:
- Builds SSH command with `claude -p --output-format json`
- Sends prompt via stdin
- Parses JSON response
- Returns parsed dict or raises on failure

**Step 3:** Commit: `git commit -m "v3.0: add reusable LLM call wrapper"`

---

### Task 5: Orientation step (world model update)

**Files:**
- Create: `src/autoresearcher2/v3/orientation.py`
- Test: `tests/v3/test_orientation.py`

**Step 1:** Write test with a mock LLM that returns a known delta. Verify the world model is updated correctly.

**Step 2:** Implement `orient(world_model: WorldModel, observation: Observation, llm_call) -> dict`:
- Build evidence-first prompt: new observation → current world model → instructions → required output schema
- Call LLM
- Parse structured delta
- Apply delta to world model
- Return delta for audit

**Step 3:** Commit: `git commit -m "v3.0: add orientation step (LLM-led world model update)"`

---

### Task 6: Generator

**Files:**
- Create: `src/autoresearcher2/v3/generator.py`
- Test: `tests/v3/test_generator.py`

**Step 1:** Write test with mock LLM. Given a world model with beliefs and tensions, verify generator returns rationale-first proposals.

**Step 2:** Implement `generate_proposals(world_model: WorldModel, n_proposals: int, llm_call) -> list[Proposal]`:
- Build prompt: world model state → recent observations → instructions → required output schema
- Prompt emphasizes: rationale-first, diverse types, cost-aware
- Parse response into Proposal objects
- Return list

**Step 3:** Commit: `git commit -m "v3.0: add generator (rationale-first proposal generation)"`

---

### Task 7: Critic

**Files:**
- Create: `src/autoresearcher2/v3/critic.py`
- Test: `tests/v3/test_critic.py`

**Step 1:** Write test with mock LLM. Given proposals of varying quality, verify critic ranks them ordinally and selects top-N.

**Step 2:** Implement `critique_proposals(world_model: WorldModel, proposals: list[Proposal], n_select: int, llm_call) -> list[Proposal]`:
- Build prompt: world model → proposals → instructions (rank ordinally, accept/reject/deprioritize)
- Parse response: ranking + decisions + rationales
- Update each proposal's critic field
- Return accepted proposals sorted by rank

**Step 3:** Commit: `git commit -m "v3.0: add critic (ordinal ranking of proposals)"`

---

### Task 8: Integration test — one full cycle

**Files:**
- Test: `tests/v3/test_integration.py`

**Step 1:** Write end-to-end test with mock LLM:
1. Start with empty world model
2. Create a fake observation (reality contact)
3. Run orientation → world model should have beliefs
4. Run generator → should produce proposals addressing those beliefs
5. Run critic → should rank and select top proposals
6. Verify rationale-first ordering, cost awareness, structured output

**Step 2:** Verify all v3.0 test criteria from the design doc pass.

**Step 3:** Commit: `git commit -m "v3.0: integration test — full orientation→generator→critic cycle"`

---

### Task 9: v3.0 manual smoke test with real LLM

**Step 1:** Write a script `scripts/run_v3_smoke.py` that:
- Creates a world model with beliefs from v1.5 evidence run data
- Feeds a real observation
- Calls orientation, generator, critic with real `claude -p`
- Prints results

**Step 2:** Run it, verify outputs make sense.

**Step 3:** Commit: `git commit -m "v3.0: smoke test script with real LLM"`

---

### Task 10: v3.0 complete — tag and branch

**Step 1:** Run full test suite: `uv run pytest tests/v3/ -v`

**Step 2:** Commit any fixes.

**Step 3:** Tag: `git tag v3.0`

**Step 4:** Start v3.1 work (filesystem persistence, backlog/todo/done directories).
