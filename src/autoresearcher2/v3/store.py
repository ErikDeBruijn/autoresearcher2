"""SQLite storage backend for v4.0.

Replaces filesystem-based workspace with SQLite tables:
- observations: append-only reality contact (layer 1)
- world_model: versioned epistemic state (layer 2)
- queue: kanban workflow with stage mutations (layer 3)

Same API surface as Workspace so planner/worker code stays unchanged.
"""
import json
import sqlite3
import time
from pathlib import Path

from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id                 TEXT PRIMARY KEY,
    created_at         REAL NOT NULL,
    intervention_type  TEXT NOT NULL,
    intervention_spec  TEXT NOT NULL,
    outcome_metrics    TEXT,
    outcome_success    INTEGER,
    error              TEXT,
    wall_time_s        REAL,
    compute_cost       REAL,
    worker_id          TEXT,
    raw_log            TEXT
);

CREATE TABLE IF NOT EXISTS world_model (
    version        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     REAL NOT NULL,
    trigger_obs_id TEXT,
    delta          TEXT NOT NULL,
    reasoning      TEXT,
    state          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS queue (
    id                TEXT PRIMARY KEY,
    created_at        REAL NOT NULL,
    stage             TEXT NOT NULL,
    intent            TEXT NOT NULL,
    rationale         TEXT NOT NULL,
    expected_learning TEXT,
    estimated_cost    TEXT,
    intervention_type TEXT,
    intervention_spec TEXT,
    rank              INTEGER,
    critic_rationale  TEXT,
    critic_decision   TEXT,
    promoted_at       REAL,
    started_at        REAL,
    finished_at       REAL,
    worker_id         TEXT,
    observation_id    TEXT
);

CREATE INDEX IF NOT EXISTS idx_stage ON queue(stage);
CREATE INDEX IF NOT EXISTS idx_priority ON queue(stage, rank);
"""


class Store:
    """SQLite-backed research workspace."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init(self):
        """Create tables and seed initial world model."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn.executescript(SCHEMA)
        # Seed empty world model if none exists
        if self.conn.execute("SELECT COUNT(*) FROM world_model").fetchone()[0] == 0:
            wm = WorldModel()
            self.conn.execute(
                "INSERT INTO world_model (created_at, trigger_obs_id, delta, reasoning, state) VALUES (?, ?, ?, ?, ?)",
                (time.time(), None, "{}", "initial empty state", json.dumps(wm.to_dict())),
            )
            self.conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Layer 1: Observations (append-only) ---

    def save_observation(self, obs: Observation):
        self.conn.execute(
            """INSERT INTO observations
               (id, created_at, intervention_type, intervention_spec,
                outcome_metrics, outcome_success, error, wall_time_s,
                compute_cost, worker_id, raw_log)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs.id, obs.created_at, obs.intervention_type,
                json.dumps(obs.intervention_spec),
                json.dumps(obs.outcome_metrics) if obs.outcome_metrics else None,
                1 if obs.outcome_success else 0,
                obs.error, obs.wall_time_s, obs.compute_cost,
                obs.worker_id, obs.raw_log,
            ),
        )
        self.conn.commit()

    def load_observation(self, obs_id: str) -> Observation | None:
        row = self.conn.execute(
            "SELECT * FROM observations WHERE id = ?", (obs_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_observation(row)

    def list_observations(self) -> list[Observation]:
        rows = self.conn.execute(
            "SELECT * FROM observations ORDER BY created_at"
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def _row_to_observation(self, row) -> Observation:
        obs = Observation(
            intervention_type=row["intervention_type"],
            intervention_spec=json.loads(row["intervention_spec"]),
            outcome_metrics=json.loads(row["outcome_metrics"]) if row["outcome_metrics"] else None,
            outcome_success=bool(row["outcome_success"]),
            error=row["error"],
            wall_time_s=row["wall_time_s"],
            compute_cost=row["compute_cost"],
            worker_id=row["worker_id"],
            raw_log=row["raw_log"],
        )
        obs.id = row["id"]
        obs.created_at = row["created_at"]
        return obs

    # --- Layer 2: World Model (versioned) ---

    def load_world_model(self) -> WorldModel:
        row = self.conn.execute(
            "SELECT * FROM world_model ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return WorldModel()
        wm = WorldModel.from_dict(json.loads(row["state"]))
        wm.version = row["version"]
        return wm

    def save_world_model(self, wm: WorldModel, trigger_obs_id: str = None,
                         delta: dict = None, reasoning: str = None):
        """Save new world model version with delta traceability."""
        self.conn.execute(
            """INSERT INTO world_model (created_at, trigger_obs_id, delta, reasoning, state)
               VALUES (?, ?, ?, ?, ?)""",
            (
                time.time(), trigger_obs_id,
                json.dumps(delta or {}),
                reasoning,
                json.dumps(wm.to_dict()),
            ),
        )
        self.conn.commit()
        # Update wm version to match DB
        row = self.conn.execute("SELECT MAX(version) FROM world_model").fetchone()
        wm.version = row[0]

    def get_world_model_history(self) -> list[dict]:
        """Return all world model versions with deltas."""
        rows = self.conn.execute(
            "SELECT version, created_at, trigger_obs_id, delta, reasoning FROM world_model ORDER BY version"
        ).fetchall()
        return [
            {
                "version": r["version"],
                "created_at": r["created_at"],
                "trigger_obs_id": r["trigger_obs_id"],
                "delta": json.loads(r["delta"]),
                "reasoning": r["reasoning"],
            }
            for r in rows
        ]

    # --- Layer 3: Queue (stage mutations) ---

    def save_proposal(self, proposal: Proposal):
        critic = proposal.critic or {}
        self.conn.execute(
            """INSERT OR REPLACE INTO queue
               (id, created_at, stage, intent, rationale, expected_learning,
                estimated_cost, intervention_type, intervention_spec,
                rank, critic_rationale, critic_decision,
                promoted_at, started_at, finished_at, worker_id, observation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                proposal.id, proposal.created_at, proposal.status,
                proposal.intent, proposal.rationale, proposal.expected_learning,
                json.dumps(proposal.estimated_cost) if proposal.estimated_cost else None,
                proposal.intervention_type,
                json.dumps(proposal.intervention_spec),
                critic.get("rank"), critic.get("rationale"), critic.get("decision"),
                None, None, None,
                getattr(proposal, "worker_id", None),
                proposal.observation_id,
            ),
        )
        self.conn.commit()

    def list_proposals(self, stage: str) -> list[Proposal]:
        rows = self.conn.execute(
            "SELECT * FROM queue WHERE stage = ? ORDER BY rank ASC NULLS LAST, created_at ASC",
            (stage,),
        ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def count_proposals(self, stage: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM queue WHERE stage = ?", (stage,)
        ).fetchone()
        return row[0]

    def move_proposal(self, proposal: Proposal, new_stage: str):
        proposal.promote(new_stage)
        now = time.time()
        updates = {"stage": new_stage}
        if new_stage == "todo":
            updates["promoted_at"] = now
        elif new_stage == "running":
            updates["started_at"] = now
        elif new_stage == "done":
            updates["finished_at"] = now

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [proposal.id]
        self.conn.execute(f"UPDATE queue SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def claim_next_todo(self, worker_id: str) -> Proposal | None:
        """Atomically claim highest-ranked todo item."""
        # Use a transaction for atomicity
        cursor = self.conn.execute(
            "SELECT * FROM queue WHERE stage = 'todo' ORDER BY rank ASC NULLS LAST, created_at ASC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None

        now = time.time()
        updated = self.conn.execute(
            "UPDATE queue SET stage = 'running', worker_id = ?, started_at = ? WHERE id = ? AND stage = 'todo'",
            (worker_id, now, row["id"]),
        )
        self.conn.commit()

        if updated.rowcount == 0:
            return None  # Another worker claimed it

        proposal = self._row_to_proposal(row)
        proposal.promote("running")
        return proposal

    def complete_proposal(self, proposal: Proposal, observation: Observation):
        """Mark proposal done, save observation, link them."""
        self.save_observation(observation)
        proposal.complete(observation.id)
        now = time.time()
        self.conn.execute(
            "UPDATE queue SET stage = 'done', finished_at = ?, observation_id = ? WHERE id = ?",
            (now, observation.id, proposal.id),
        )
        self.conn.commit()

    def mark_reviewed(self, proposal_id: str):
        """Move a done proposal to reviewed after orientation has processed it."""
        self.conn.execute(
            "UPDATE queue SET stage = 'reviewed' WHERE id = ? AND stage = 'done'",
            (proposal_id,),
        )
        self.conn.commit()

    def _row_to_proposal(self, row) -> Proposal:
        p = Proposal(
            intent=row["intent"],
            rationale=row["rationale"],
            expected_learning=row["expected_learning"],
            intervention_type=row["intervention_type"],
            intervention_spec=json.loads(row["intervention_spec"]) if row["intervention_spec"] else {},
            estimated_cost=json.loads(row["estimated_cost"]) if row["estimated_cost"] else None,
        )
        p.id = row["id"]
        p.created_at = row["created_at"]
        p.status = row["stage"]
        p.observation_id = row["observation_id"]
        if row["critic_decision"]:
            p.critic = {
                "decision": row["critic_decision"],
                "rank": row["rank"],
                "rationale": row["critic_rationale"],
            }
        return p
