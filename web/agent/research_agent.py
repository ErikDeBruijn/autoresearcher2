"""AG2 research agent backend for CopilotKit chat sidebar.

Provides a conversational agent with access to the autoresearcher2 Store.
The agent can answer questions about beliefs, observations, queue status,
and submit new proposals.

Runs on port 8008, compatible with CopilotKit via AG-UI protocol.
"""
import json
import sys
from pathlib import Path

from autogen import ConversableAgent, LLMConfig
from autogen.oai.client import OpenAILLMConfigEntry
from autogen.ag_ui import AGUIStream
from ag_ui.core import RunAgentInput
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import uvicorn

# Add src to path for Store access
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from autoresearcher2.v3.store import Store

DB_PATH = Path(__file__).resolve().parent.parent.parent / "research_v4.db"

MODELS = [
    {
        "id": "qwen3.5-397b",
        "name": "Qwen 3.5 397B (local)",
        "api_base": "http://ollama.local:8080/v1",
        "max_tokens": 8192,
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "api_base": None,
    },
]

DEFAULT_MODEL = "qwen3.5-397b"


def build_system_message() -> str:
    """Build system message with current research state."""
    try:
        store = Store(DB_PATH)
        projects = store.list_projects(active_only=True)

        # Build per-project context
        project_sections = []
        for proj in projects:
            pid = proj["id"]
            wm = store.load_world_model(project_id=pid)
            dc = proj.get("domain_config") or {}
            target_metric = dc.get("target_metric")

            beliefs_text = "\n".join(
                f"    - [{b['confidence']:.2f}] {b['claim']}"
                for b in wm.beliefs
            ) or "    None"
            tensions_text = "\n".join(
                f"    - [{t['salience']:.2f}] {t['nature']}"
                for t in wm.tensions
            ) or "    None"

            project_sections.append(
                f"### Project: {proj['name']} (v{wm.version})\n"
                f"  Description: {proj.get('description', 'N/A')}\n"
                f"  Target metric: {target_metric or 'not set'}\n"
                f"  Optimize: {dc.get('optimize', 'N/A')}\n\n"
                f"  **Beliefs**\n{beliefs_text}\n\n"
                f"  **Tensions**\n{tensions_text}\n\n"
                f"  **Cost Beliefs**\n  {json.dumps(wm.cost_beliefs, indent=2)}"
            )

        # If no active projects, load default world model
        if not project_sections:
            wm = store.load_world_model()
            beliefs_text = "\n".join(
                f"  - [{b['confidence']:.2f}] {b['claim']}"
                for b in wm.beliefs
            ) or "  None"
            tensions_text = "\n".join(
                f"  - [{t['salience']:.2f}] {t['nature']}"
                for t in wm.tensions
            ) or "  None"
            project_sections.append(
                f"### Default World Model (v{wm.version})\n\n"
                f"**Beliefs**\n{beliefs_text}\n\n"
                f"**Tensions**\n{tensions_text}\n\n"
                f"**Cost Beliefs**\n{json.dumps(wm.cost_beliefs, indent=2)}"
            )

        # Observations across all projects
        obs = store.list_observations()
        recent_obs = obs[-5:] if obs else []

        def _format_metrics(o):
            if not o.outcome_metrics:
                return ""
            parts = [f"{k}={v}" for k, v in o.outcome_metrics.items()]
            return " " + ", ".join(parts)

        obs_text = "\n".join(
            f"  - {o.intervention_type}: {json.dumps(o.intervention_spec)} → "
            f"{'success' if o.outcome_success else 'FAIL'}"
            f"{_format_metrics(o)}"
            for o in recent_obs
        ) or "  None yet"

        counts = {
            stage: store.count_proposals(stage)
            for stage in ("backlog", "todo", "running", "done", "reviewed")
        }
        queue_text = ", ".join(f"{k}: {v}" for k, v in counts.items())

        store.close()

        projects_context = "\n\n".join(project_sections)

        # Build project summary for the intro
        if projects:
            project_list = ", ".join(
                f"{p['name']} ({(p.get('domain_config') or {}).get('target_metric', 'no metric')})"
                for p in projects
            )
            intro_desc = f"an autonomous research system currently working on: {project_list}"
        else:
            intro_desc = "an autonomous research system"

        context = f"""## Current Research State

{projects_context}

### Recent Observations (last 5)
{obs_text}

### Queue
{queue_text}"""

    except Exception:
        context = "(Research database not available)"
        intro_desc = "an autonomous research system"

    return f"""You are a research assistant for AutoResearcher2, {intro_desc}.

You help the researcher understand the current state of research, explain findings, suggest next experiments, and answer questions about the methodology.

{context}

## Your capabilities
- Explain what the system has learned and what remains uncertain
- Suggest promising experiments based on current beliefs and tensions
- Explain the OODA loop: Observe → Orient → Decide → Act
- Help interpret results in context of each project's target metric and optimization direction
- Discuss tradeoffs between exploration (probing unknowns) and exploitation (refining knowns)

Be concise and direct. Use the research data above to ground your answers."""


def get_llm_config(model_id: str) -> LLMConfig:
    model_def = next((m for m in MODELS if m["id"] == model_id), None)
    if model_def is None:
        model_def = next(m for m in MODELS if m["id"] == DEFAULT_MODEL)
    entry_kwargs = {"model": model_def["id"], "stream": True}
    if model_def.get("api_base"):
        entry_kwargs["base_url"] = model_def["api_base"]
        entry_kwargs["api_key"] = "not-needed"
    if model_def.get("max_tokens"):
        entry_kwargs["max_tokens"] = model_def["max_tokens"]
    return LLMConfig(OpenAILLMConfigEntry(**entry_kwargs))


def create_agent(model_id: str | None = None) -> ConversableAgent:
    config = get_llm_config(model_id or DEFAULT_MODEL)
    return ConversableAgent(
        name="researcher",
        system_message=build_system_message(),
        llm_config=config,
    )


app = FastAPI(title="AutoResearcher2 Chat Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
@app.post("/chat/")
async def chat(request: Request):
    body = await request.body()
    incoming = RunAgentInput.model_validate_json(body)
    model_id = None
    if incoming.forwarded_props and isinstance(incoming.forwarded_props, dict):
        model_id = incoming.forwarded_props.get("model")
    agent = create_agent(model_id)
    stream = AGUIStream(agent)
    return StreamingResponse(
        stream.dispatch(incoming, accept=request.headers.get("accept"))
    )


@app.get("/models")
def list_models():
    return [{"id": m["id"], "name": m["name"]} for m in MODELS]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
