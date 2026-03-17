"""FastAPI backend for autoresearcher2 web UI.

Serves the SQLite Store via REST API + WebSocket for live updates.
"""
import asyncio
import json
import re as _re
import subprocess
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
QUEUE_STAGES = ("backlog", "todo", "running", "done", "reviewed")

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
            with get_store() as store:
                counts = {
                    stage: store.count_proposals(stage)
                    for stage in QUEUE_STAGES
                }
            if counts != last_counts:
                last_counts = counts
                await manager.broadcast({"type": "queue_update", "counts": counts})
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB exists with demo data if needed
    with get_store() as store:
        store.init()
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
    with get_store() as store:
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


@app.get("/api/projects")
def get_projects():
    """Get all projects with energy stats."""
    with get_store() as store:
        projects = store.list_projects()
        observations = store.list_observations()

        # Compute per-project energy stats
        avg_power_samples = []
        for obs in observations:
            if obs.avg_power_w:
                avg_power_samples.append(obs.avg_power_w)
        fallback_power_w = (sum(avg_power_samples) / len(avg_power_samples)) if avg_power_samples else 400.0

        for proj in projects:
            pid = proj["id"]
            proj_obs = [o for o in observations if o.project_id == pid]
            energy = 0.0
            cost = 0.0
            wall_time = 0.0
            for o in proj_obs:
                if o.wall_time_s:
                    wall_time += o.wall_time_s
                if o.energy_kwh:
                    energy += o.energy_kwh
                elif o.wall_time_s:
                    energy += (fallback_power_w * o.wall_time_s) / 3_600_000
                if o.cost_eur:
                    cost += o.cost_eur
                elif o.wall_time_s:
                    cost += (fallback_power_w * o.wall_time_s) / 3_600_000 * 0.23
            proj["energy_kwh"] = round(energy, 4)
            proj["cost_eur"] = round(cost, 4)
            proj["wall_time_s"] = round(wall_time, 1)
            proj["experiment_count"] = len(proj_obs)
            # Compute expected gain for Auto priority display
            try:
                proj["expected_gain"] = round(store.compute_expected_gain(proj["id"]), 3)
            except Exception:
                proj["expected_gain"] = None

        return projects


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    """Get a single project."""
    with get_store() as store:
        project = store.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        return project


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    domain_config: dict | None = None


@app.post("/api/projects")
async def create_project(data: ProjectCreate):
    """Create a new research project."""
    with get_store() as store:
        pid = store.create_project(
            name=data.name,
            description=data.description,
            domain_config=data.domain_config,
        )
        project = store.get_project(pid)
        await manager.broadcast({"type": "project_created", "project": project})
        return project


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    active: bool | None = None
    domain_config: dict | None = None
    priority: str | None = None


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, data: ProjectUpdate):
    """Update a project."""
    with get_store() as store:
        updates = {k: v for k, v in data.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        store.update_project(project_id, **updates)
        project = store.get_project(project_id)
        await manager.broadcast({"type": "project_updated", "project": project})
        return project


@app.get("/api/projects/{project_id}/expected_gain")
def get_expected_gain(project_id: str):
    """Compute expected learning gain for a project."""
    with get_store() as store:
        gain = store.compute_expected_gain(project_id)
        return {"project_id": project_id, "expected_gain": round(gain, 3)}


@app.get("/api/queue")
def get_queue():
    """Get all proposals grouped by stage, with worker_id for running and observation data for done/reviewed."""
    with get_store() as store:
        # Pre-load world model history for delta lookups
        wm_history = store.get_world_model_history()
        delta_by_obs = {h["trigger_obs_id"]: h for h in wm_history if h["trigger_obs_id"]}

        stages: dict[str, list] = {}
        for stage in QUEUE_STAGES:
            proposals = store.list_proposals(stage)
            items = []
            for p in proposals:
                d = p.to_dict()
                d["project_id"] = p.project_id
                # Add worker_id and started_at from DB for running proposals
                if stage == "running":
                    row = store.conn.execute(
                        "SELECT worker_id, started_at FROM queue WHERE id = ?", (p.id,)
                    ).fetchone()
                    d["worker_id"] = row["worker_id"] if row else None
                    d["started_at"] = row["started_at"] if row else None
                # Add stage timestamps for all proposals
                row = store.conn.execute(
                    "SELECT promoted_at, started_at, finished_at FROM queue WHERE id = ?", (p.id,)
                ).fetchone()
                if row:
                    d["promoted_at"] = row["promoted_at"]
                    d["started_at"] = d.get("started_at") or row["started_at"]
                    d["finished_at"] = row["finished_at"]
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
                            "artifact_paths": obs.artifact_paths or {},
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


@app.get("/api/queue/counts")
def get_queue_counts():
    """Get proposal counts per stage."""
    with get_store() as store:
        return {
            stage: store.count_proposals(stage)
            for stage in QUEUE_STAGES
        }


@app.get("/api/observations")
def get_observations():
    """Get all observations."""
    with get_store() as store:
        obs = store.list_observations()
        return [o.to_dict() for o in obs]


@app.get("/api/world-model")
def get_world_model(project_id: str = None):
    """Get current world model state. If no project_id, returns the first active project's WM."""
    with get_store() as store:
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


@app.get("/api/world-model/history")
def get_world_model_history():
    """Get world model version history."""
    with get_store() as store:
        return store.get_world_model_history()


@app.get("/api/stats")
def get_stats():
    """Get research statistics."""
    with get_store() as store:
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
        tracked_energy_kwh = 0.0
        untracked_wall_time = 0.0
        avg_power_samples = []

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
                tracked_energy_kwh += obs.energy_kwh
                total_energy_kwh += obs.energy_kwh
                if obs.avg_power_w:
                    avg_power_samples.append(obs.avg_power_w)
            elif obs.wall_time_s:
                untracked_wall_time += obs.wall_time_s
            if obs.cost_eur:
                total_cost_eur += obs.cost_eur

        # Estimate energy for untracked runs using average power from tracked runs
        if untracked_wall_time > 0 and avg_power_samples:
            avg_power_w = sum(avg_power_samples) / len(avg_power_samples)
            estimated_kwh = (avg_power_w * untracked_wall_time) / 3_600_000
            total_energy_kwh += estimated_kwh
            total_cost_eur += estimated_kwh * 0.23  # fallback price

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

        # Active workers: those currently running proposals
        active_worker_ids = set()
        running_rows = store.conn.execute(
            "SELECT DISTINCT worker_id FROM queue WHERE stage = 'running' AND worker_id IS NOT NULL"
        ).fetchall()
        for row in running_rows:
            active_worker_ids.add(row["worker_id"])

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
            for stage in QUEUE_STAGES
        }

        # Intervention type breakdown
        type_counts: dict[str, int] = {}
        for obs in observations:
            itype = obs.intervention_type or "unknown"
            type_counts[itype] = type_counts.get(itype, 0) + 1

        # Pipeline activity (what the planner is currently doing)
        pipeline_activity = store.get_pipeline_activity()

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
            "active_worker_ids": list(active_worker_ids),
            "learntropy_trace": learntropy_trace,
            "intervention_types": type_counts,
            "total_energy_kwh": round(total_energy_kwh, 4),
            "total_cost_eur": round(total_cost_eur, 4),
            "pipeline_activity": pipeline_activity,
        }


@app.post("/api/proposals")
async def create_proposal(data: ProposalCreate):
    """Submit a new proposal to the backlog."""
    with get_store() as store:
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


@app.post("/api/proposals/{proposal_id}/cancel")
async def cancel_proposal(proposal_id: str):
    """Cancel a running proposal: move it back to backlog."""
    with get_store() as store:
        if store.cancel_proposal(proposal_id):
            await manager.broadcast({
                "type": "proposal_moved",
                "proposal_id": proposal_id,
                "from_stage": "running",
                "to_stage": "backlog",
            })
            return {"status": "ok", "proposal_id": proposal_id}
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found in running stage")


@app.post("/api/proposals/{proposal_id}/promote")
async def promote_proposal(proposal_id: str, target_stage: str = "todo"):
    """Move a proposal to a different stage."""
    with get_store() as store:
        # Find the proposal
        for stage in QUEUE_STAGES:
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


@app.delete("/api/proposals/{proposal_id}")
async def delete_proposal(proposal_id: str):
    """Delete a proposal from any stage."""
    with get_store() as store:
        result = store.conn.execute("DELETE FROM queue WHERE id = ?", (proposal_id,))
        store.conn.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
        await manager.broadcast({"type": "proposal_deleted", "proposal_id": proposal_id})
        return {"status": "ok", "proposal_id": proposal_id}


# --- Chat endpoint ---

# Default domain config for new projects (users provide full config via chat or API)
_DEFAULT_DOMAIN_CONFIG = {
    "name": "research experiment",
    "description": "We run experiments to optimize a target metric.",
    "target_metric": "target_metric",
    "optimize": "minimize",
    "intervention_types": "config_change, probe, code_change",
    "parameters": "any key-value pairs relevant to the domain",
}

def _execute_chat_commands(response_text: str) -> str:
    """Detect and execute structured commands in LLM chat response."""
    # Look for CREATE_PROJECT command
    pattern = r'```command\s*\n\s*CREATE_PROJECT\s+(\{.*?\})\s*\n\s*```'
    match = _re.search(pattern, response_text, _re.DOTALL)
    if not match:
        # Also try without code fence (LLM might not wrap it)
        pattern2 = r'CREATE_PROJECT\s+(\{.*?\})'
        match = _re.search(pattern2, response_text)
    if not match:
        return response_text

    try:
        cmd = json.loads(match.group(1))
        name = cmd.get("name", "Unnamed Project")
        description = cmd.get("description", "")
        domain_type = cmd.get("domain_type", "generic")
        parameters = cmd.get("parameters", "")

        # Accept a full domain_config dict, or fall back to template lookup
        domain_cfg = cmd.get("domain_config")
        if domain_cfg and isinstance(domain_cfg, dict):
            # User/LLM provided a full custom domain config
            pass
        else:
            domain_cfg = dict(_DEFAULT_DOMAIN_CONFIG)
        if parameters:
            domain_cfg["parameters"] = parameters

        with get_store() as store:
            pid = store.create_project(
                name=name,
                description=description,
                domain_config=domain_cfg,
            )

        # Replace command block with success message
        success_msg = f"\n\n**Project created:** {name} (id: `{pid}`, domain: {domain_type})"
        response_text = response_text[:match.start()] + success_msg + response_text[match.end():]
    except json.JSONDecodeError as e:
        response_text += f"\n\n(Failed to parse project command: {e})"
    except Exception as e:
        response_text += f"\n\n(Failed to create project: {e})"

    return response_text


def _build_chat_system_prompt(store: Store) -> str:
    """Build the chat agent system prompt dynamically from store data.

    Reads projects, world model, and observations to construct a prompt
    that is domain-agnostic -- no hardcoded metric names or domain types.
    """
    wm = store.load_world_model()
    observations = store.list_observations()
    counts = {
        stage: store.count_proposals(stage)
        for stage in QUEUE_STAGES
    }
    projects = store.list_projects()

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

    # Build project descriptions with their metrics and optimization targets
    projects_text_lines = []
    metric_lines = []
    project_descriptions = []
    for p in projects:
        dc = p.get("domain_config") or {}
        metric = dc.get("target_metric", "target_metric")
        optimize = dc.get("optimize", "minimize")
        desc = dc.get("description", p.get("description", ""))
        projects_text_lines.append(
            f"  - {p['name']} (id={p['id']}, active={p.get('active', True)}, metric={metric}, {optimize})"
        )
        metric_lines.append(f"  - {p['name']}: {optimize} {metric}")
        if desc:
            project_descriptions.append(desc)
    projects_text = "\n".join(projects_text_lines) or "  No projects yet"
    metric_guidance = ("Optimization targets:\n" + "\n".join(metric_lines)) if metric_lines else "No target metrics configured yet."

    # Format observations using actual metrics from their projects
    recent_obs = observations[-5:]
    obs_lines = []
    # Build a project lookup for metric display
    proj_lookup = {p["id"]: p for p in projects}
    for o in recent_obs:
        line = f"  - {o.intervention_type}: {json.dumps(o.intervention_spec)} -> "
        line += "success" if o.outcome_success else "FAIL"
        if o.outcome_metrics:
            # Show the project's target metric first, then others
            proj = proj_lookup.get(o.project_id, {})
            dc = (proj.get("domain_config") or {})
            primary_metric = dc.get("target_metric")
            shown = []
            if primary_metric and primary_metric in o.outcome_metrics:
                shown.append(f"{primary_metric}={o.outcome_metrics[primary_metric]}")
            for k, v in o.outcome_metrics.items():
                if k != primary_metric:
                    shown.append(f"{k}={v}")
            if shown:
                line += " " + ", ".join(shown)
        obs_lines.append(line)
    obs_text = "\n".join(obs_lines) or "  None yet"
    queue_text = ", ".join(f"{k}: {v}" for k, v in counts.items())

    # Build domain context from projects
    domain_context = ""
    if project_descriptions:
        domain_context = "Research context: " + " ".join(project_descriptions)

    system = f"""You are a research assistant for AutoResearcher2, an autonomous experiment system.
{domain_context}

Current state (World Model v{wm.version}):

Beliefs:
{beliefs_text}

Tensions:
{tensions_text}

Recent observations (last 5):
{obs_text}

Queue: {queue_text}
Total experiments: {len(observations)}, Success rate: {sum(1 for o in observations if o.outcome_success)}/{len(observations)}

Existing projects:
{projects_text}

{metric_guidance}

Be concise and direct.

## Actions (use Bash tool)

You have CLI tools available. Use the Bash tool to run them:

- `research-create-project --name "Name" --description "Desc" --domain <domain_type_or_custom> --parameters "param1,param2"`
- `research-list-projects`
- `research-status`
- `research-submit-proposal --project proj_id --intent "Test X" --type config_change --spec '{{"param": "value"}}'`

When creating a project, ask the user what they want to optimize and which parameters to vary, then construct a domain_config dict from their answers."""
    return system


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user", "content": "..."}, ...]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat with LLM about the research, grounded in current Store data."""


    # Build context from Store
    with get_store() as store:
        system = _build_chat_system_prompt(store)

    # Build conversation for LLM
    conversation = [{"role": "system", "content": system}]
    for msg in req.messages:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    # Call LLM via claude CLI with research management tools
    prompt_json = json.dumps(conversation)
    repo_dir = str(Path(__file__).resolve().parent.parent)
    skill_bin = f"{repo_dir}/.claude/skills/research-management/bin"
    env = {**__import__("os").environ, "PATH": f"{skill_bin}:{__import__('os').environ.get('PATH', '')}"}
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--output-format", "text",
                "--dangerously-skip-permissions",
                "--allowedTools", "Bash(research-*)",
            ],
            input=prompt_json,
            capture_output=True, text=True, timeout=120,
            env=env,
        )
        if result.returncode != 0:
            return {"role": "assistant", "content": f"LLM error: {result.stderr[:200]}"}
        response_text = result.stdout.strip()
    except FileNotFoundError:
        return {"role": "assistant", "content": "Claude CLI not available on this machine. Chat requires the research agent to run on the VM."}
    except Exception as e:
        return {"role": "assistant", "content": f"Error: {str(e)}"}

    # Also check for legacy command format (fallback)
    response_text = _execute_chat_commands(response_text)

    return {"role": "assistant", "content": response_text}


# --- Worker management ---

from autoresearcher2.v3.executors import COST_TRACKER_URL


@app.get("/api/workers/status")
def get_worker_status():
    """Check research loop status and GPU power from gpu-cost-tracker service."""

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
                gpu_temps = data.get("gpu_temperatures_c", {})
                gpu_vram = data.get("gpu_vram_gb", {})
                gpu_info = []
                for gpu_id in sorted(gpu_powers.keys(), key=int):
                    info = {
                        "utilization_pct": gpu_utils.get(str(gpu_id), gpu_utils.get(int(gpu_id), 0)),
                        "power_w": gpu_powers[gpu_id],
                        "temperature_c": gpu_temps.get(str(gpu_id), gpu_temps.get(int(gpu_id), None)),
                    }
                    vram = gpu_vram.get(str(gpu_id), gpu_vram.get(int(gpu_id), None))
                    if vram:
                        info["vram_used_gb"] = vram.get("used_gb")
                        info["vram_total_gb"] = vram.get("total_gb")
                    gpu_info.append(info)
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


# --- Report generation ---
from starlette.responses import FileResponse


REPORT_DIR = Path(__file__).parent.parent / "artifacts" / "reports"
REPORT_SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_report.py"


def _run_report_script():
    """Run generate_report.py and raise HTTPException on failure."""
    result = subprocess.run(
        ["uv", "run", "python", str(REPORT_SCRIPT), "--db", str(DB_PATH), "--no-llm"],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {result.stderr[-500:]}")
    return result


@app.post("/api/report")
async def generate_report():
    """Run generate_report.py --db ... --no-llm and return the PDF path."""
    try:
        _run_report_script()
        pdf = _find_latest_pdf()
        if not pdf:
            raise HTTPException(status_code=500, detail="Report generated but no PDF found")
        return {"status": "ok", "path": str(pdf)}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Report generation timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _find_latest_pdf() -> Path | None:
    """Return the most recently modified PDF in REPORT_DIR (searches subdirectories)."""
    if not REPORT_DIR.exists():
        return None
    pdfs = sorted(REPORT_DIR.glob("**/*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[0] if pdfs else None


@app.get("/api/report/latest")
def get_latest_report():
    """Return metadata about the latest generated report."""
    pdf = _find_latest_pdf()
    if not pdf:
        raise HTTPException(status_code=404, detail="No report found")
    return {"path": str(pdf), "filename": pdf.name, "size_bytes": pdf.stat().st_size}


@app.get("/api/report/download")
def download_report(regenerate: bool = True):
    """Generate (if needed) and serve the PDF report.

    With regenerate=True (default), always regenerates for fresh data.
    """
    if regenerate:
        _run_report_script()
    pdf = _find_latest_pdf()
    if not pdf:
        raise HTTPException(status_code=404, detail="No report found")
    from datetime import datetime as _dt
    fname = f"autoresearcher-report-{_dt.now().strftime('%Y-%m-%d')}.pdf"
    return FileResponse(pdf, media_type="application/pdf", filename=fname)


# --- Artifact serving ---


@app.get("/api/artifacts/{obs_id}/{artifact_name}")
def get_artifact(obs_id: str, artifact_name: str):
    """Serve an artifact file (video, image, etc.) for an observation."""
    with get_store() as store:
        obs = store.load_observation(obs_id)
        if not obs or not obs.artifact_paths:
            raise HTTPException(status_code=404, detail="Artifact not found")
        path = obs.artifact_paths.get(artifact_name)
        if not path:
            raise HTTPException(status_code=404, detail=f"Artifact '{artifact_name}' not found")
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Artifact file missing from disk")
        # Determine media type
        suffix = file_path.suffix.lower()
        media_types = {".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg"}
        media_type = media_types.get(suffix, "application/octet-stream")
        return FileResponse(file_path, media_type=media_type)


# --- Serve frontend static files with cache control ---
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
