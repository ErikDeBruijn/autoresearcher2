"""SQLite storage backend for v4.0.

Three-layer schema:
- observations: append-only reality contact (layer 1)
- world_model: versioned epistemic state (layer 2)
- queue: kanban workflow with stage mutations (layer 3)
"""
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    domain_config TEXT,
    executor_script TEXT,
    docker_image  TEXT,
    created_at    REAL NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    priority      TEXT NOT NULL DEFAULT 'auto'
);

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
    raw_log            TEXT,
    project_id         TEXT REFERENCES projects(id),
    energy_kwh         REAL,
    cost_eur           REAL,
    avg_power_w        REAL,
    artifact_paths     TEXT
);

CREATE TABLE IF NOT EXISTS world_model (
    version        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     REAL NOT NULL,
    trigger_obs_id TEXT,
    delta          TEXT NOT NULL,
    reasoning      TEXT,
    state          TEXT NOT NULL,
    project_id     TEXT REFERENCES projects(id)
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
    observation_id    TEXT,
    project_id        TEXT REFERENCES projects(id)
);

CREATE INDEX IF NOT EXISTS idx_stage ON queue(stage);
CREATE INDEX IF NOT EXISTS idx_priority ON queue(stage, rank);
CREATE INDEX IF NOT EXISTS idx_queue_project ON queue(project_id);
CREATE INDEX IF NOT EXISTS idx_obs_project ON observations(project_id);
CREATE INDEX IF NOT EXISTS idx_wm_project ON world_model(project_id);
"""

MIGRATIONS = [
    # v4.4: Add projects table and project_id columns
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
        domain_config TEXT, executor_script TEXT, docker_image TEXT,
        created_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
    );
    """,
    "ALTER TABLE queue ADD COLUMN project_id TEXT REFERENCES projects(id);",
    "ALTER TABLE observations ADD COLUMN project_id TEXT REFERENCES projects(id);",
    "ALTER TABLE world_model ADD COLUMN project_id TEXT REFERENCES projects(id);",
    "CREATE INDEX IF NOT EXISTS idx_queue_project ON queue(project_id);",
    "CREATE INDEX IF NOT EXISTS idx_obs_project ON observations(project_id);",
    "CREATE INDEX IF NOT EXISTS idx_wm_project ON world_model(project_id);",
    # v4.5: Cost tracking fields on observations
    "ALTER TABLE observations ADD COLUMN energy_kwh REAL;",
    "ALTER TABLE observations ADD COLUMN cost_eur REAL;",
    "ALTER TABLE observations ADD COLUMN avg_power_w REAL;",
    # v4.7: Pipeline activity tracking
    """CREATE TABLE IF NOT EXISTS pipeline_activity (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        phase TEXT,
        project_id TEXT,
        proposal_id TEXT,
        started_at REAL,
        updated_at REAL
    );""",
    "INSERT OR IGNORE INTO pipeline_activity (id) VALUES (1);",
    # v4.9: Artifact storage for recordings/previews
    "ALTER TABLE observations ADD COLUMN artifact_paths TEXT;",
    # v4.11: Project priority system
    "ALTER TABLE projects ADD COLUMN priority TEXT NOT NULL DEFAULT 'auto';",
]


class Store:
    """SQLite-backed research store."""

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
        self._backup_if_has_data()
        self._migrate()
        self.conn.executescript(SCHEMA)
        # Seed empty world model if none exists
        if self.conn.execute("SELECT COUNT(*) FROM world_model").fetchone()[0] == 0:
            wm = WorldModel()
            self.conn.execute(
                "INSERT INTO world_model (created_at, trigger_obs_id, delta, reasoning, state) VALUES (?, ?, ?, ?, ?)",
                (time.time(), None, "{}", "initial empty state", json.dumps(wm.to_dict())),
            )
            self.conn.commit()

    def _backup_if_has_data(self):
        """Auto-backup the DB before schema changes if it contains research data."""
        if not self.db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            has_obs = conn.execute(
                "SELECT COUNT(*) FROM observations"
            ).fetchone()[0] > 0
            has_queue = conn.execute(
                "SELECT COUNT(*) FROM queue"
            ).fetchone()[0] > 0
            conn.close()
        except sqlite3.OperationalError:
            return  # DB doesn't have these tables yet
        if has_obs or has_queue:
            import shutil
            from datetime import datetime
            backup_dir = self.db_path.parent / "db_backups"
            backup_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{self.db_path.stem}_{ts}.db"
            shutil.copy2(self.db_path, backup_path)
            # Also copy WAL if it exists
            wal = Path(str(self.db_path) + "-wal")
            if wal.exists():
                shutil.copy2(wal, backup_dir / f"{self.db_path.stem}_{ts}.db-wal")
            shm = Path(str(self.db_path) + "-shm")
            if shm.exists():
                shutil.copy2(shm, backup_dir / f"{self.db_path.stem}_{ts}.db-shm")
            logger.info("Auto-backed up DB to %s (%s obs, %s queue items)",
                       backup_path.name,
                       "has" if has_obs else "no",
                       "has" if has_queue else "no")

    def _migrate(self):
        """Apply migrations for existing databases (idempotent)."""
        for sql in MIGRATIONS:
            try:
                self.conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column/table/index already exists
        self.conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # --- Pipeline Activity ---

    def set_pipeline_activity(self, phase: str = None, project_id: str = None, proposal_id: str = None):
        """Update pipeline activity (what the planner is currently doing)."""
        import time
        self.conn.execute(
            "UPDATE pipeline_activity SET phase=?, project_id=?, proposal_id=?, started_at=COALESCE(started_at, ?), updated_at=? WHERE id=1",
            (phase, project_id, proposal_id, time.time(), time.time()),
        )
        self.conn.commit()

    def clear_pipeline_activity(self):
        """Clear pipeline activity (planner is idle)."""
        import time
        self.conn.execute(
            "UPDATE pipeline_activity SET phase=NULL, project_id=NULL, proposal_id=NULL, started_at=NULL, updated_at=? WHERE id=1",
            (time.time(),),
        )
        self.conn.commit()

    def get_pipeline_activity(self) -> dict:
        """Get current pipeline activity."""
        row = self.conn.execute("SELECT * FROM pipeline_activity WHERE id=1").fetchone()
        if row is None:
            return {}
        return {
            "phase": row["phase"],
            "project_id": row["project_id"],
            "proposal_id": row["proposal_id"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
        }

    # --- Projects ---

    def create_project(self, name: str, description: str = None,
                       domain_config: dict = None, executor_script: str = None,
                       docker_image: str = None) -> str:
        pid = f"proj_{uuid.uuid4().hex[:8]}"
        self.conn.execute(
            """INSERT INTO projects (id, name, description, domain_config,
               executor_script, docker_image, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pid, name, description,
             json.dumps(domain_config) if domain_config else None,
             executor_script, docker_image, time.time()),
        )
        self.conn.commit()
        return pid

    def get_project(self, project_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_project(row)

    def list_projects(self, active_only: bool = False) -> list[dict]:
        if active_only:
            rows = self.conn.execute(
                "SELECT * FROM projects WHERE active = 1 ORDER BY created_at"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM projects ORDER BY created_at"
            ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def update_project(self, project_id: str, **kwargs):
        allowed = {"name", "description", "domain_config", "executor_script",
                    "docker_image", "active", "priority"}
        updates = {}
        for k, v in kwargs.items():
            if k not in allowed:
                raise ValueError(f"Unknown project field: {k}")
            if k == "domain_config" and isinstance(v, dict):
                v = json.dumps(v)
            updates[k] = v
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [project_id]
        self.conn.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def _row_to_project(self, row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "domain_config": json.loads(row["domain_config"]) if row["domain_config"] else None,
            "executor_script": row["executor_script"],
            "docker_image": row["docker_image"],
            "created_at": row["created_at"],
            "active": bool(row["active"]),
            "priority": row["priority"] if "priority" in row.keys() else "auto",
        }

    # --- Layer 1: Observations (append-only) ---

    def save_observation(self, obs: Observation, project_id: str = None):
        self.conn.execute(
            """INSERT INTO observations
               (id, created_at, intervention_type, intervention_spec,
                outcome_metrics, outcome_success, error, wall_time_s,
                compute_cost, worker_id, raw_log, project_id,
                energy_kwh, cost_eur, avg_power_w, artifact_paths)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs.id, obs.created_at, obs.intervention_type,
                json.dumps(obs.intervention_spec),
                json.dumps(obs.outcome_metrics) if obs.outcome_metrics else None,
                1 if obs.outcome_success else 0,
                obs.error, obs.wall_time_s, obs.compute_cost,
                obs.worker_id, obs.raw_log,
                project_id,
                obs.energy_kwh, obs.cost_eur, obs.avg_power_w,
                json.dumps(obs.artifact_paths) if obs.artifact_paths else None,
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

    def list_observations(self, project_id: str = None) -> list[Observation]:
        if project_id:
            rows = self.conn.execute(
                "SELECT * FROM observations WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        else:
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
        obs.project_id = row["project_id"] if "project_id" in row.keys() else None
        if "energy_kwh" in row.keys():
            obs.energy_kwh = row["energy_kwh"]
            obs.cost_eur = row["cost_eur"]
            obs.avg_power_w = row["avg_power_w"]
        if "artifact_paths" in row.keys() and row["artifact_paths"]:
            obs.artifact_paths = json.loads(row["artifact_paths"])
        return obs

    def update_observation_artifacts(self, obs_id: str, artifact_paths: dict):
        """Update artifact_paths on an existing observation."""
        self.conn.execute(
            "UPDATE observations SET artifact_paths = ? WHERE id = ?",
            (json.dumps(artifact_paths), obs_id),
        )
        self.conn.commit()

    # --- Layer 2: World Model (versioned) ---

    def load_world_model(self, project_id: str = None) -> WorldModel:
        if project_id:
            row = self.conn.execute(
                "SELECT * FROM world_model WHERE project_id = ? ORDER BY version DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM world_model WHERE project_id IS NULL ORDER BY version DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return WorldModel()
        wm = WorldModel.from_dict(json.loads(row["state"]))
        wm.version = row["version"]
        return wm

    def save_world_model(self, wm: WorldModel, trigger_obs_id: str = None,
                         delta: dict = None, reasoning: str = None,
                         project_id: str = None):
        """Save new world model version with delta traceability."""
        self.conn.execute(
            """INSERT INTO world_model (created_at, trigger_obs_id, delta, reasoning, state, project_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                time.time(), trigger_obs_id,
                json.dumps(delta or {}),
                reasoning,
                json.dumps(wm.to_dict()),
                project_id,
            ),
        )
        self.conn.commit()
        # Update wm version to match DB
        row = self.conn.execute("SELECT MAX(version) FROM world_model").fetchone()
        wm.version = row[0]

    def get_world_model_history(self, project_id: str = None) -> list[dict]:
        """Return all world model versions with deltas."""
        if project_id:
            rows = self.conn.execute(
                "SELECT version, created_at, trigger_obs_id, delta, reasoning FROM world_model WHERE project_id = ? ORDER BY version",
                (project_id,),
            ).fetchall()
        else:
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

    def compute_expected_gain(self, project_id: str) -> float:
        """Compute expected learning gain for a project (0.0-1.0).

        Based on recent learntropy trend, belief uncertainty, and unresolved tensions.
        Used by Auto priority to allocate planner resources dynamically.
        """
        wm = self.load_world_model(project_id=project_id)
        history = self.get_world_model_history(project_id=project_id)

        # 1. Recent learntropy trend (last 5 updates)
        recent_lt = [h["delta"].get("learntropy", 0) for h in history[-5:] if h["delta"]]
        avg_learntropy = sum(recent_lt) / len(recent_lt) if recent_lt else 0.5

        # 2. Low-confidence belief ratio
        beliefs = wm.beliefs
        low_conf = [b for b in beliefs if float(b.get("confidence", 1.0)) < 0.5]
        uncertainty_ratio = len(low_conf) / max(len(beliefs), 1)

        # 3. Unresolved tension ratio
        tension_ratio = len(wm.tensions) / max(len(beliefs), 1)

        return min(1.0, 0.4 * avg_learntropy + 0.35 * uncertainty_ratio + 0.25 * tension_ratio)

    # --- Layer 3: Queue (stage mutations) ---

    def save_proposal(self, proposal: Proposal, project_id: str = None):
        critic = proposal.critic or {}
        pid = project_id or getattr(proposal, "project_id", None)
        self.conn.execute(
            """INSERT OR REPLACE INTO queue
               (id, created_at, stage, intent, rationale, expected_learning,
                estimated_cost, intervention_type, intervention_spec,
                rank, critic_rationale, critic_decision,
                promoted_at, started_at, finished_at, worker_id, observation_id,
                project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                pid,
            ),
        )
        self.conn.commit()

    def list_proposals(self, stage: str, project_id: str = None) -> list[Proposal]:
        if project_id:
            rows = self.conn.execute(
                "SELECT * FROM queue WHERE stage = ? AND project_id = ? ORDER BY rank ASC NULLS LAST, created_at ASC",
                (stage, project_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM queue WHERE stage = ? ORDER BY rank ASC NULLS LAST, created_at ASC",
                (stage,),
            ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def count_proposals(self, stage: str, project_id: str = None) -> int:
        if project_id:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM queue WHERE stage = ? AND project_id = ?",
                (stage, project_id),
            ).fetchone()
        else:
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

    def claim_next_todo(self, worker_id: str, project_ids: list[str] | None = None) -> Proposal | None:
        """Atomically claim highest-ranked todo item from active projects.

        Retries up to 3 times in case of concurrent claims by other workers.
        Skips proposals from paused (active=0) projects.

        Args:
            worker_id: Worker identifier for tracking.
            project_ids: If set, only claim from these project IDs.
                         Use to restrict workers to specific project domains.
        """
        for _attempt in range(3):
            if project_ids is not None:
                if len(project_ids) == 0:
                    return None  # No projects to claim from
                placeholders = ",".join("?" for _ in project_ids)
                cursor = self.conn.execute(
                    f"""SELECT q.* FROM queue q
                       LEFT JOIN projects p ON q.project_id = p.id
                       WHERE q.stage = 'todo'
                         AND (q.project_id IS NULL OR p.active = 1)
                         AND q.project_id IN ({placeholders})
                       ORDER BY q.rank ASC NULLS LAST, q.created_at ASC LIMIT 1""",
                    project_ids,
                )
            else:
                cursor = self.conn.execute(
                    """SELECT q.* FROM queue q
                       LEFT JOIN projects p ON q.project_id = p.id
                       WHERE q.stage = 'todo'
                         AND (q.project_id IS NULL OR p.active = 1)
                       ORDER BY q.rank ASC NULLS LAST, q.created_at ASC LIMIT 1"""
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

            if updated.rowcount > 0:
                break  # Successfully claimed
            # Another worker claimed it, retry with next item
        else:
            return None

        proposal = self._row_to_proposal(row)
        proposal.promote("running")
        return proposal

    def complete_proposal(self, proposal: Proposal, observation: Observation):
        """Mark proposal done, save observation, link them."""
        self.save_observation(observation, project_id=getattr(proposal, "project_id", None))
        proposal.complete(observation.id)
        now = time.time()
        self.conn.execute(
            "UPDATE queue SET stage = 'done', finished_at = ?, observation_id = ? WHERE id = ?",
            (now, observation.id, proposal.id),
        )
        self.conn.commit()

    def cancel_proposal(self, proposal_id: str) -> bool:
        """Cancel a running proposal: move it back to backlog.

        The worker may still finish executing, but the result won't be
        linked to this proposal. Returns True if a proposal was cancelled.
        """
        cursor = self.conn.execute(
            "UPDATE queue SET stage = 'backlog', worker_id = NULL, "
            "started_at = NULL, rank = NULL "
            "WHERE id = ? AND stage = 'running'",
            (proposal_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def reclaim_stale_running(self, timeout_s: float = 600) -> int:
        """Move proposals stuck in 'running' back to 'todo'.

        This handles the case where a service restart leaves proposals
        in 'running' state with no worker actually executing them.
        """
        cutoff = time.time() - timeout_s
        cursor = self.conn.execute(
            "UPDATE queue SET stage = 'todo', worker_id = NULL, started_at = NULL "
            "WHERE stage = 'running' AND started_at < ?",
            (cutoff,),
        )
        self.conn.commit()
        return cursor.rowcount

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
        p.project_id = row["project_id"] if "project_id" in row.keys() else None
        if row["critic_decision"]:
            p.critic = {
                "decision": row["critic_decision"],
                "rank": row["rank"],
                "rationale": row["critic_rationale"],
            }
        return p
