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
async def poll_store_changes():
    """Background task: broadcast queue counts to WebSocket clients every few seconds."""
    last_counts = None
    while True:
        await asyncio.sleep(5)
        if not manager.active:
            continue
        try:
            store = get_store()
            counts = {
                stage: store.count_proposals(stage)
                for stage in ("backlog", "todo", "running", "done", "reviewed")
            }
            store.close()
            if counts != last_counts:
                last_counts = counts
                await manager.broadcast({"type": "queue_update", "counts": counts})
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB exists with demo data if needed
    store = get_store()
    store.init()
    store.close()
    app.state.start_time = time.time()
    # Start background poller for live updates
    task = asyncio.create_task(poll_store_changes())
    yield
    task.cancel()

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

@app.get("/api/health")
def health():
    """Health check: service uptime, DB accessible, last experiment time."""
    import os
    store = get_store()
    try:
        obs = store.list_observations()
        last_obs_time = max((o.created_at for o in obs), default=None)
        return {
            "status": "ok",
            "uptime_s": time.time() - app.state.start_time if hasattr(app.state, "start_time") else None,
            "db_observations": len(obs),
            "last_experiment_at": last_obs_time,
            "seconds_since_last": round(time.time() - last_obs_time, 1) if last_obs_time else None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        store.close()


@app.get("/api/projects")
def get_projects():
    """Get all projects."""
    store = get_store()
    try:
        return store.list_projects()
    finally:
        store.close()


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    """Get a single project."""
    store = get_store()
    try:
        project = store.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        return project
    finally:
        store.close()


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    domain_config: dict | None = None


@app.post("/api/projects")
async def create_project(data: ProjectCreate):
    """Create a new research project."""
    store = get_store()
    try:
        pid = store.create_project(
            name=data.name,
            description=data.description,
            domain_config=data.domain_config,
        )
        project = store.get_project(pid)
        await manager.broadcast({"type": "project_created", "project": project})
        return project
    finally:
        store.close()


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    active: bool | None = None
    domain_config: dict | None = None


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, data: ProjectUpdate):
    """Update a project."""
    store = get_store()
    try:
        updates = {k: v for k, v in data.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        store.update_project(project_id, **updates)
        project = store.get_project(project_id)
        await manager.broadcast({"type": "project_updated", "project": project})
        return project
    finally:
        store.close()


@app.get("/api/queue")
def get_queue():
    """Get all proposals grouped by stage, with worker_id for running and observation data for done/reviewed."""
    store = get_store()
    try:
        # Pre-load world model history for delta lookups
        wm_history = store.get_world_model_history()
        delta_by_obs = {h["trigger_obs_id"]: h for h in wm_history if h["trigger_obs_id"]}

        stages: dict[str, list] = {}
        for stage in ("backlog", "todo", "running", "done", "reviewed"):
            proposals = store.list_proposals(stage)
            items = []
            for p in proposals:
                d = p.to_dict()
                d["project_id"] = p.project_id
                # Add worker_id from DB for running proposals
                if stage == "running":
                    row = store.conn.execute(
                        "SELECT worker_id FROM queue WHERE id = ?", (p.id,)
                    ).fetchone()
                    d["worker_id"] = row["worker_id"] if row else None
                # Add observation and world model delta for done/reviewed
                if stage in ("done", "reviewed") and p.observation_id:
                    obs = store.load_observation(p.observation_id)
                    if obs:
                        d["observation"] = {
                            "outcome_success": obs.outcome_success,
                            "outcome_metrics": obs.outcome_metrics,
                            "wall_time_s": obs.wall_time_s,
                            "error": obs.error,
                            "energy_kwh": obs.energy_kwh,
                            "cost_eur": obs.cost_eur,
                            "avg_power_w": obs.avg_power_w,
                        }
                    # Find the world model update triggered by this observation
                    wm_update = delta_by_obs.get(p.observation_id)
                    if wm_update:
                        d["world_model_update"] = {
                            "version": wm_update["version"],
                            "reasoning": wm_update["reasoning"],
                            "delta": wm_update["delta"],
                        }
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
def get_world_model(project_id: str = None):
    """Get current world model state. If no project_id, returns the first active project's WM."""
    store = get_store()
    try:
        if project_id:
            wm = store.load_world_model(project_id=project_id)
        else:
            # Find the first active project with a world model
            projects = store.list_projects(active_only=True)
            wm = None
            for p in projects:
                candidate = store.load_world_model(project_id=p["id"])
                if candidate.version > 0:
                    if wm is None or candidate.version > wm.version:
                        wm = candidate
            if wm is None:
                # Fallback to default (no project)
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
        history = store.get_world_model_history()

        # Find the best world model across all active projects
        wm = store.load_world_model()  # default (no project)
        projects = store.list_projects(active_only=True)
        for p in projects:
            candidate = store.load_world_model(project_id=p["id"])
            if candidate.version > wm.version:
                wm = candidate

        # Worker stats
        worker_times: dict[str, list[float]] = {}
        success_count = 0
        failure_count = 0
        total_wall_time = 0.0

        total_energy_kwh = 0.0
        total_cost_eur = 0.0

        for obs in observations:
            if obs.outcome_success:
                success_count += 1
            else:
                failure_count += 1
            if obs.wall_time_s:
                total_wall_time += obs.wall_time_s
                wid = obs.worker_id or "unknown"
                worker_times.setdefault(wid, []).append(obs.wall_time_s)
            if obs.energy_kwh:
                total_energy_kwh += obs.energy_kwh
            if obs.cost_eur:
                total_cost_eur += obs.cost_eur

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
            "total_energy_kwh": round(total_energy_kwh, 4),
            "total_cost_eur": round(total_cost_eur, 4),
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


@app.delete("/api/proposals/{proposal_id}")
async def delete_proposal(proposal_id: str):
    """Delete a proposal from any stage."""
    store = get_store()
    try:
        result = store.conn.execute("DELETE FROM queue WHERE id = ?", (proposal_id,))
        store.conn.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
        await manager.broadcast({"type": "proposal_deleted", "proposal_id": proposal_id})
        return {"status": "ok", "proposal_id": proposal_id}
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

COST_TRACKER_URL = "http://pve03.local:8377"


@app.get("/api/workers/status")
def get_worker_status():
    """Check research loop status and GPU power from gpu-cost-tracker service."""
    import subprocess
    import urllib.request
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "autoresearcher"],
            capture_output=True, text=True, timeout=5,
        )
        active = result.stdout.strip() == "active"

        # Get GPU and energy data from gpu-cost-tracker service
        gpu_info = None
        energy_status = None
        try:
            req = urllib.request.Request(f"{COST_TRACKER_URL}/status", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                gpu_powers = data.get("gpu_powers_w", {})
                gpu_utils = data.get("gpu_utilizations_pct", {})
                gpu_info = []
                for gpu_id in sorted(gpu_powers.keys(), key=int):
                    gpu_info.append({
                        "utilization_pct": gpu_utils.get(str(gpu_id), gpu_utils.get(int(gpu_id), 0)),
                        "power_w": gpu_powers[gpu_id],
                    })
                energy_status = {
                    "shelly_total_w": data.get("shelly_total_w"),
                    "system_base_w": data.get("system_base_w"),
                    "price_eur_per_kwh": data.get("price_eur_per_kwh"),
                    "active_jobs": data.get("active_jobs", {}),
                }
        except Exception:
            pass

        return {
            "running": active,
            "service": "autoresearcher.service",
            "gpus": gpu_info,
            "energy": energy_status,
        }
    except FileNotFoundError:
        return {"running": False, "error": "systemctl not available"}
    except Exception as e:
        return {"running": False, "error": str(e)}


@app.post("/api/workers/start")
def start_worker():
    """Start the autoresearcher systemd service."""
    import subprocess
    try:
        subprocess.run(["systemctl", "start", "autoresearcher"], timeout=10, check=True)
        return {"status": "started"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": str(e)}
    except FileNotFoundError:
        return {"status": "error", "error": "systemctl not available"}


@app.post("/api/workers/stop")
def stop_worker():
    """Stop the autoresearcher systemd service."""
    import subprocess
    try:
        subprocess.run(["systemctl", "stop", "autoresearcher"], timeout=10, check=True)
        return {"status": "stopped"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": str(e)}
    except FileNotFoundError:
        return {"status": "error", "error": "systemctl not available"}


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


# --- Serve frontend static files with cache control ---
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Set cache headers: no-cache for HTML, long cache for hashed assets."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/_next/static/"):
            # Hashed assets — cache forever
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path == "/" or path.endswith(".html"):
            # HTML — always revalidate
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists():
    app.add_middleware(CacheControlMiddleware)
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
