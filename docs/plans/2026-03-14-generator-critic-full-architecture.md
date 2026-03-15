# Generator-Critic Full Architecture — v4.0 Design

## Relationship to MVP

This document describes the full architecture that the MVP (see `2026-03-14-generator-critic-design.md`) is a stepping stone toward. Build the MVP first. If it validates the core idea (proposal generation + evaluation + selective execution), then evolve toward this architecture incrementally.

## Guiding principle

The world model is the accumulated, testable intuition of the system; experiments are not the knowledge itself, but reality contacts that correct, sharpen, or undermine that intuition.

Intelligent research is not only sample-efficient. It is also deliberation-efficient.

## Four conceptual layers

### Layer 1 — Reality Contact

Immutable registration of contact moments with reality. An observation is evidence, not understanding.

Contains:
- What was done (intervention: type + structured spec)
- What came out (outcome: metrics + success/failure)
- Under what conditions (runtime metadata, compute cost, timestamp)

**Append-only.** Never mutated after creation.

### Layer 2 — World Model (Epistemic State)

A structured epistemic state of the system. Not a summary, not a document, not a prompt, not a log. JSON/YAML are the serialization form, not the essence.

Contains:
- **Beliefs** — claims about the domain, each with confidence, evidence for/against, age
- **Expectations** — if we do X we expect Y, with confidence and basis
- **Tensions / contradictions** — conflicting beliefs or unexplained observations, with salience
- **Cost beliefs** — estimated cost_to_think, cost_to_test per intervention type, cost_of_being_wrong per belief
- **Probe-fidelity beliefs** — which cheap probes exist, how reliable they are vs full experiments, cost ratio
- **Preference model** — what we're optimizing, constraints, trade-offs
- **Salience** — which observations reshaped the model most (learntropy), unresolved tensions, stale beliefs

Each version is preserved. Version N+1 arises exclusively through a structured LLM-led orientation step.

### Layer 3 — Operational Workflow

Queue with kanban stages: backlog → todo → running → done → reviewed. This is workflow, not cognition. Items contain:

- Epistemic intent and rationale (rationale-first: why this proposal, which belief/tension it addresses)
- Intervention spec (executable action, concretized only after rationale)
- Ordinal ranking (critic), estimated costs
- Lifecycle metadata (timestamps per stage, worker_id, link to observation after completion)

### Layer 4 — Textual Renderings

Projections of layers 1-3. Prompts, explanations, summaries, reports. Never source of truth. Generated from structured state, not the other way around.

## Workflow (OODA × Active Inference)

### Observe — Reality Contact

A worker completes an intervention. The result is registered as an observation in layer 1. This triggers the orientation step.

### Orient — World Model Update (LLM-led)

The orientation step is **primarily LLM-led**, but evidence-anchored, schema-constrained, and auditable.

The LLM receives a structured prompt in fixed order:

```
1. New observations      — the observation(s) triggering this update
2. Current world model   — full layer 2 state (version N)
3. Cost beliefs          — current cost_to_think, cost_to_test, cost_of_being_wrong
4. Update instructions   — "Reason about what these facts mean for beliefs,
                            expectations, tensions, salience, costs, and probe
                            fidelity. Produce a structured delta."
5. Required output schema — exact JSON schema for the update-delta
```

Order is mandatory: facts first, current state second, reasoning third, structured delta fourth.

The LLM produces an **update-delta**, not a new narrative:

```json
{
  "beliefs_added": [],
  "beliefs_revised": [{"id": "B1", "new_confidence": 0.75, "reasoning": "..."}],
  "beliefs_retired": [{"id": "B5", "reason": "..."}],
  "expectations_added": [],
  "expectations_revised": [],
  "tensions_added": [],
  "tensions_resolved": [{"id": "T2", "resolution": "...", "reasoning": "..."}],
  "salience_updated": {"high_learntropy": ["obs_014"], "stale_beliefs": ["B3"]},
  "cost_beliefs_updated": {},
  "probe_fidelity_updated": {}
}
```

Delta applied to version N → version N+1. Both versions preserved, delta traceable. The LLM's reasoning is layer 4 (rendering); the delta itself is layer 2 (state).

**Learntropy** is calculated retrospectively here: how much did this observation actually reshape the world model? Stored in salience as signal for future decisions.

### Decide — Generator + Critic

Both reason **against the world model** (layer 2), not against loose logs or raw observations.

**Generator** (LLM role: curious researcher, rationale-first):

Cognitive order per proposal:
1. **Epistemic intent** — which belief, tension, or open question is addressed?
2. **Rationale** — why is this valuable now? What is the expected epistemic and pragmatic yield?
3. **Cost assessment** — is a cheap probe sufficient, or is a full experiment needed? Is further deliberation more rational than testing?
4. **Proposed intervention** — concrete action (config change, code change, schema extension, probe, replication, or other type)
5. **Executable spec** — structured action payload for the worker

Interventions are not limited to a fixed parameter grid. The system can propose introducing new parameters, rewriting code, choosing a different experimental design, or running a minimal probe.

**Critic** (LLM role: skeptical reviewer, ordinal):

Ranks backlog items relative to each other. No absolute scores. Considers:
- Which proposal addresses the most salient tension or stale belief?
- What is the ratio of expected information value to cost_to_test?
- Is there a cheaper probe that tests the same thing?
- Prunes items made irrelevant by recent observations

**Deliberation cost check**: if the todo queue is nearly empty and cheap probes are available, skip the full generator→critic cycle. Don't always deliberate when testing is cheaper.

### Act — Workers

Workers claim todo items and execute the intervention spec. They produce observations (layer 1), not knowledge. The todo queue should always contain items so workers never idle.

## Persistence (full architecture)

| Layer | Storage | Mutation model |
|-------|---------|---------------|
| 1 — Reality Contact | SQLite `observations` | **Append-only.** Never mutated. |
| 2 — World Model | SQLite `world_model` | **Versioned state.** Each version preserved, delta traceable. |
| 3 — Workflow | SQLite `queue` | **Stage mutations.** Items move through stages, timestamps per transition. |
| 4 — Renderings | Filesystem / stdout | **Ephemeral.** Generated, not persisted as source. |

```sql
-- Layer 1: append-only
CREATE TABLE observations (
    id                 TEXT PRIMARY KEY,
    created_at         REAL NOT NULL,
    intervention_type  TEXT NOT NULL,
    intervention_spec  TEXT NOT NULL,     -- JSON
    outcome_metrics    TEXT,              -- JSON
    outcome_success    INTEGER,
    error              TEXT,
    wall_time_s        REAL,
    compute_cost       REAL,
    worker_id          TEXT,
    raw_log            TEXT               -- audit trail, not for reasoning
);

-- Layer 2: versioned state
CREATE TABLE world_model (
    version        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     REAL NOT NULL,
    trigger_obs_id TEXT,                  -- FK observations.id
    delta          TEXT NOT NULL,          -- JSON: structured update-delta
    reasoning      TEXT,                   -- layer 4: LLM reasoning (audit)
    state          TEXT NOT NULL           -- JSON: full epistemic state after delta
);

-- Layer 3: stage mutations
CREATE TABLE queue (
    id                TEXT PRIMARY KEY,
    created_at        REAL NOT NULL,
    stage             TEXT NOT NULL,       -- backlog|todo|running|done|reviewed

    -- Rationale-first (cognition)
    epistemic_intent  TEXT NOT NULL,
    rationale         TEXT NOT NULL,
    cost_assessment   TEXT,                -- JSON: est_cost_to_test, probe_alternative

    -- Intervention (execution)
    intervention_type TEXT,
    intervention_spec TEXT,                -- JSON: executable payload

    -- Critic
    rank              INTEGER,
    critic_rationale  TEXT,

    -- Lifecycle
    promoted_at       REAL,
    started_at        REAL,
    finished_at       REAL,
    worker_id         TEXT,
    observation_id    TEXT                 -- FK observations.id after completion
);

CREATE INDEX idx_stage ON queue(stage);
CREATE INDEX idx_priority ON queue(stage, rank);
```

## State transitions (full architecture)

### After a new observation

```
Worker completes intervention
  → INSERT observations (layer 1, append-only)
  → Trigger orientation step:
      LLM receives: [new observation, world_model.state v(N), cost_beliefs]
      LLM produces: structured delta + reasoning
      → INSERT world_model v(N+1) (layer 2, versioned)
  → UPDATE queue item: stage='done', observation_id=... (layer 3)
```

### World model version N → N+1

1. Observation arrives
2. LLM-led orientation: facts → current state → reasoning → delta
3. Delta contains: beliefs added/revised/retired, expectations updated, tensions added/resolved, salience recalculated, cost beliefs updated
4. Delta applied to state N → state N+1
5. Learntropy calculated: magnitude of delta as fraction of total model
6. Both versions preserved, delta traceable

### Queue items through stages

```
Generator produces proposal
  → INSERT queue (stage='backlog', epistemic_intent, rationale, intervention_spec)

Critic ranks backlog
  → UPDATE queue SET rank=..., critic_rationale=...
  → UPDATE queue SET stage='todo', promoted_at=... WHERE rank <= N

Worker claims
  → UPDATE queue SET stage='running', worker_id=..., started_at=...
    WHERE stage='todo' ORDER BY rank LIMIT 1

Worker completes
  → INSERT observations (layer 1)
  → UPDATE queue SET stage='done', finished_at=..., observation_id=...
  → Trigger orientation (layer 2 update)

Review (after orientation)
  → UPDATE queue SET stage='reviewed'
```

## Process model

**run_planner.py** — generator + critic + world model updates
- Polling loop (~60s)
- If new results in done: run orientation step (update world model)
- If backlog + todo < threshold: run generator
- If backlog has unranked items: run critic, promote top-N to todo

**run_worker.py** — execution
- Polling loop (~30s)
- If todo has items: claim highest-ranked, execute, write result
- One instance per GPU, multiple instances can coexist

Optional: `run_research.py` wrapper for local convenience, not architecturally leading.

## Migration from MVP

| MVP (filesystem) | Full (SQLite) |
|-------------------|---------------|
| `world_model.json` | `world_model` table (versioned) |
| `proposals/backlog/*.json` | `queue` table, stage='backlog' |
| `proposals/todo/*.json` | `queue` table, stage='todo' |
| `proposals/done/*.json` | `queue` table, stage='done' |
| `results/*.json` | `observations` table |
| `world_model_history/` | `world_model` table (all versions) |

The migration is mechanical: same data model, different storage backend. The cognitive architecture (generator, critic, orientation step, rationale-first proposals) stays identical.

## Design principles

1. **World model = epistemic state**, not document/summary/log
2. **Observations = reality contact**, not knowledge
3. **Queue = workflow**, not cognition
4. **Text = rendering**, not source of truth
5. **Rationale-first**: epistemic intent → rationale → intervention → executable
6. **Cost-aware**: cost_to_think vs cost_to_test vs cost_of_being_wrong
7. **Ordinal ranking**: critic compares proposals relatively, no absolute scores
8. **LLM-led orientation**: structured, evidence-anchored, schema-constrained, auditable
9. **Active inference (prospective)**: action selection balances epistemic and pragmatic value
10. **Learntropy (retrospective)**: appraisal measures how much an observation reshaped the world model
11. **Open interventions**: not limited to parameter grids — config, code, schema, probe, replication, other
12. **Deliberation-efficient**: sometimes a cheap test is more rational than more thinking
