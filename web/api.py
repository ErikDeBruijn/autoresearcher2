"""FastAPI backend for autoresearcher2 web UI.

Serves the SQLite Store via REST API + WebSocket for live updates.
"""
import asyncio
import json
import time
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store
from autoresearcher2.v3.proposal import Proposal

# --- Config ---
DB_PATH = Path(__file__).parent.parent / "research_v4.db"

# --- WebSocket manager ---
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in self.active[:]:
            try:
                await ws.send_json(message)
            except Exception:
                self.active.remove(ws)

manager = ConnectionManager()

# --- App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB exists with demo data if needed
    store = get_store()
    store.init()
    store.close()
    yield

app = FastAPI(title="AutoResearcher2 — Research Dashboard", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store() -> Store:
    return Store(DB_PATH)


# --- Pydantic models ---
class ProposalCreate(BaseModel):
    intent: str
    rationale: str
    expected_learning: str
    intervention_type: str = "config_change"
    intervention_spec: dict = {}
    estimated_cost: dict | None = None


class ChatMessage(BaseModel):
    message: str


# --- REST endpoints ---

@app.get("/api/queue")
def get_queue():
    """Get all proposals grouped by stage."""
    store = get_store()
    try:
        stages = {}
        for stage in ("backlog", "todo", "running", "done", "reviewed"):
            proposals = store.list_proposals(stage)
            stages[stage] = [p.to_dict() for p in proposals]
        return stages
    finally:
        store.close()


@app.get("/api/queue/counts")
def get_queue_counts():
    """Get proposal counts per stage."""
    store = get_store()
    try:
        return {
            stage: store.count_proposals(stage)
            for stage in ("backlog", "todo", "running", "done", "reviewed")
        }
    finally:
        store.close()


@app.get("/api/observations")
def get_observations():
    """Get all observations."""
    store = get_store()
    try:
        obs = store.list_observations()
        return [o.to_dict() for o in obs]
    finally:
        store.close()


@app.get("/api/world-model")
def get_world_model():
    """Get current world model state."""
    store = get_store()
    try:
        wm = store.load_world_model()
        return wm.to_dict()
    finally:
        store.close()


@app.get("/api/world-model/history")
def get_world_model_history():
    """Get world model version history."""
    store = get_store()
    try:
        return store.get_world_model_history()
    finally:
        store.close()


@app.get("/api/stats")
def get_stats():
    """Get research statistics."""
    store = get_store()
    try:
        observations = store.list_observations()
        wm = store.load_world_model()
        history = store.get_world_model_history()

        # Worker stats
        worker_times: dict[str, list[float]] = {}
        success_count = 0
        failure_count = 0
        total_wall_time = 0.0

        for obs in observations:
            if obs.outcome_success:
                success_count += 1
            else:
                failure_count += 1
            if obs.wall_time_s:
                total_wall_time += obs.wall_time_s
                wid = obs.worker_id or "unknown"
                worker_times.setdefault(wid, []).append(obs.wall_time_s)

        # Per-worker stats
        workers = {}
        for wid, times in worker_times.items():
            workers[wid] = {
                "experiments": len(times),
                "total_time_s": sum(times),
                "avg_time_s": sum(times) / len(times) if times else 0,
                "min_time_s": min(times) if times else 0,
                "max_time_s": max(times) if times else 0,
            }

        # Learntropy from world model history
        learntropy_trace = []
        for h in history:
            lt = h["delta"].get("learntropy")
            if lt is not None:
                learntropy_trace.append({
                    "version": h["version"],
                    "learntropy": lt,
                    "trigger": h["trigger_obs_id"],
                })

        # Queue counts
        counts = {
            stage: store.count_proposals(stage)
            for stage in ("backlog", "todo", "running", "done", "reviewed")
        }

        # Intervention type breakdown
        type_counts: dict[str, int] = {}
        for obs in observations:
            itype = obs.intervention_type or "unknown"
            type_counts[itype] = type_counts.get(itype, 0) + 1

        return {
            "total_observations": len(observations),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / max(len(observations), 1),
            "total_wall_time_s": total_wall_time,
            "world_model_version": wm.version,
            "belief_count": len(wm.beliefs),
            "tension_count": len(wm.tensions),
            "queue_counts": counts,
            "workers": workers,
            "learntropy_trace": learntropy_trace,
            "intervention_types": type_counts,
        }
    finally:
        store.close()


@app.post("/api/proposals")
async def create_proposal(data: ProposalCreate):
    """Submit a new proposal to the backlog."""
    store = get_store()
    try:
        p = Proposal(
            intent=data.intent,
            rationale=data.rationale,
            expected_learning=data.expected_learning,
            intervention_type=data.intervention_type,
            intervention_spec=data.intervention_spec,
            estimated_cost=data.estimated_cost,
        )
        store.save_proposal(p)
        await manager.broadcast({"type": "proposal_created", "proposal": p.to_dict()})
        return p.to_dict()
    finally:
        store.close()


@app.post("/api/proposals/{proposal_id}/promote")
async def promote_proposal(proposal_id: str, target_stage: str = "todo"):
    """Move a proposal to a different stage."""
    store = get_store()
    try:
        # Find the proposal
        for stage in ("backlog", "todo", "running", "done", "reviewed"):
            for p in store.list_proposals(stage):
                if p.id == proposal_id:
                    store.move_proposal(p, target_stage)
                    await manager.broadcast({
                        "type": "proposal_moved",
                        "proposal_id": proposal_id,
                        "from_stage": stage,
                        "to_stage": target_stage,
                    })
                    return {"status": "ok", "proposal_id": proposal_id, "new_stage": target_stage}
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    finally:
        store.close()


# --- WebSocket for live updates ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(ws)


# --- Serve frontend static files ---
frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
