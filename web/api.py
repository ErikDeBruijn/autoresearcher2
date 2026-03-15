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
    """Get all proposals grouped by stage, with worker_id for running."""
    store = get_store()
    try:
        stages: dict[str, list] = {}
        for stage in ("backlog", "todo", "running", "done", "reviewed"):
            proposals = store.list_proposals(stage)
            items = []
            for p in proposals:
                d = p.to_dict()
                # Add worker_id from DB for running proposals
                if stage == "running":
                    row = store.conn.execute(
                        "SELECT worker_id FROM queue WHERE id = ?", (p.id,)
                    ).fetchone()
                    d["worker_id"] = row["worker_id"] if row else None
                items.append(d)
            stages[stage] = items
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


# --- Chat endpoint ---

class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user", "content": "..."}, ...]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat with LLM about the research, grounded in current Store data."""
    import subprocess

    # Build context from Store
    store = get_store()
    try:
        wm = store.load_world_model()
        observations = store.list_observations()
        counts = {
            stage: store.count_proposals(stage)
            for stage in ("backlog", "todo", "running", "done", "reviewed")
        }
    finally:
        store.close()

    def fmt_num(val):
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)

    beliefs_text = "\n".join(
        f"  - [{fmt_num(b['confidence'])}] {b['claim']}"
        for b in wm.beliefs
    )
    tensions_text = "\n".join(
        f"  - [{fmt_num(t['salience'])}] {t['nature']}"
        for t in wm.tensions
    ) or "  None"
    recent_obs = observations[-5:]
    obs_lines = []
    for o in recent_obs:
        line = f"  - {o.intervention_type}: {json.dumps(o.intervention_spec)} → "
        line += "success" if o.outcome_success else "FAIL"
        if o.outcome_metrics:
            line += f" val_bpb={o.outcome_metrics.get('val_bpb', '?')}"
        obs_lines.append(line)
    obs_text = "\n".join(obs_lines) or "  None yet"
    queue_text = ", ".join(f"{k}: {v}" for k, v in counts.items())

    system = f"""You are a research assistant for AutoResearcher2, an autonomous NanoGPT training experiment system.

Current state (World Model v{wm.version}):

Beliefs:
{beliefs_text}

Tensions:
{tensions_text}

Recent observations (last 5):
{obs_text}

Queue: {queue_text}
Total experiments: {len(observations)}, Success rate: {sum(1 for o in observations if o.outcome_success)}/{len(observations)}

Be concise and direct. Lower val_bpb is better."""

    # Build conversation for LLM
    conversation = [{"role": "system", "content": system}]
    for msg in req.messages:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    # Call LLM via claude CLI
    prompt_json = json.dumps(conversation)
    try:
        result = subprocess.run(
            ["claude", "-p", "--output-format", "text"],
            input=prompt_json,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return {"role": "assistant", "content": f"LLM error: {result.stderr[:200]}"}
        return {"role": "assistant", "content": result.stdout.strip()}
    except FileNotFoundError:
        return {"role": "assistant", "content": "Claude CLI not available on this machine. Chat requires the research agent to run on the VM."}
    except Exception as e:
        return {"role": "assistant", "content": f"Error: {str(e)}"}


# --- Worker management ---

@app.get("/api/workers/status")
def get_worker_status():
    """Check if the research loop is running (via screen sessions)."""
    import subprocess
    try:
        result = subprocess.run(
            ["screen", "-ls"], capture_output=True, text=True, timeout=5,
        )
        screens = result.stdout
        running = "v4research" in screens
        return {
            "running": running,
            "screens": screens.strip(),
        }
    except Exception as e:
        return {"running": False, "error": str(e)}


class WorkerStartRequest(BaseModel):
    cuda_device: str = "1"
    max_cycles: int = 50


@app.post("/api/workers/start")
def start_worker(req: WorkerStartRequest):
    """Start the v4 research loop in a screen session."""
    import subprocess

    # Check if already running
    check = subprocess.run(["screen", "-ls"], capture_output=True, text=True, timeout=5)
    if "v4research" in check.stdout:
        return {"status": "already_running"}

    project_dir = Path(__file__).resolve().parent.parent
    cmd = (
        f"cd {project_dir} && PYTHONPATH=src python3 scripts/run_v4_real.py "
        f"--init --local-llm --cuda-device {req.cuda_device} "
        f"--max-cycles {req.max_cycles} 2>&1 | tee v4research.log"
    )
    subprocess.run(
        ["screen", "-dmS", "v4research", "bash", "-c", cmd],
        timeout=5,
    )
    return {"status": "started", "cuda_device": req.cuda_device, "max_cycles": req.max_cycles}


@app.post("/api/workers/stop")
def stop_worker():
    """Stop the research loop screen session."""
    import subprocess
    try:
        subprocess.run(["screen", "-S", "v4research", "-X", "quit"], timeout=5)
        return {"status": "stopped"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


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
