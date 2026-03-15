# Generator-Critic Architecture — MVP Design (v3.0–v3.2)

## Version roadmap

- **v3.0** — Generator + Critic as two LLM calls (core behavior)
- **v3.1** — Backlog/todo/done on filesystem (persistence)
- **v3.2** — Runner loop + planner (automation)
- **v4.0** — Full architecture (see `2026-03-14-generator-critic-full-architecture.md`)

## Core idea

Generate more proposals than you execute. Evaluate before acting.

This is the first real upgrade beyond sequential experiment selection: an explicit bottleneck between idea and execution, where a critic filters and ranks proposals before any compute is spent.

## Guiding principle

The world model is the accumulated, testable intuition of the system; experiments are not the knowledge itself, but reality contacts that correct, sharpen, or undermine that intuition.

## Conceptual layers

```
1. Reality contact    — trial results, metrics, runtime metadata
2. World model        — beliefs, confidence, expectations, tensions, salience, cost beliefs
3. Operational workflow — backlog, todo, running, done, reviewed
4. Textual renderings — explanations, prompts, summaries (never source of truth)
```

OODA as ordering principle:
- Observe → reality contact (new results arrive)
- Orient → world model update (LLM-led, evidence-anchored)
- Decide → generator + critic (rationale-first proposals, ordinal ranking)
- Act → runner executes top picks

Active inference is prospective (action selection: expected epistemic + pragmatic value, cost-aware).
Learntropy is retrospective (appraisal: how much did this observation reshape the world model).

## MVP scope

### Phase 2 — Proposal generation + review

**Generator** (LLM role: curious researcher / hypothesis generator)
- Input: world_model.json + recent results + current best + recent surprises
- Output: 5-10 proposals, each rationale-first:
  1. epistemic intent — what belief/tension/question does this address?
  2. rationale — why is this valuable now?
  3. expected learning — what would we learn regardless of outcome?
  4. proposed intervention — config change, code change, probe, replication, etc.
  5. estimated cost — cost_to_test, and whether a cheaper probe exists

**Critic** (LLM role: skeptical reviewer / editor)
- Input: world_model.json + generator's proposals
- Output: ordinal ranking + accept/reject/deprioritize per proposal
  - Which proposal challenges our weakest assumptions for the lowest cost?
  - Are any proposals redundant with done experiments?
  - Is there a cheaper probe that tests the same belief?
  - Select top 1-3 for execution

**Key constraint**: the critic ranks ordinally (this > that), not with absolute scores.

**Cost awareness**: both generator and critic reason about cost_to_think vs cost_to_test vs cost_of_being_wrong. A cheap probe that tests a shaky belief beats an expensive experiment that confirms a strong one.

### Phase 3 — Backlog + todo + execution handoff

Filesystem-first, not database-first:

```
proposals/
  backlog/       — generator output, not yet reviewed
  todo/          — critic-approved, ready for execution
  done/          — completed, linked to result
results/         — observation files (reality contact)
world_model.json — current epistemic state
```

Each proposal is one JSON file:

```json
{
  "id": "exp_20260314_001",
  "status": "backlog",
  "intent": "Test whether low weight decay generalizes in the promising depth/lr region",
  "rationale": "Best outcomes cluster at DEPTH=8, MATRIX_LR=0.04. WEIGHT_DECAY response curve is underexplored.",
  "expected_learning": "Clarifies whether low weight decay is broadly helpful or a corner effect",
  "intervention_type": "config_change",
  "intervention_spec": {
    "DEPTH": "8",
    "MATRIX_LR": "0.04",
    "WEIGHT_DECAY": "0.1"
  },
  "estimated_cost": {
    "cost_to_test": "1 train.py run (~5 min)",
    "cost_to_think": "low",
    "cheaper_probe": null
  },
  "critic": {
    "decision": "accept",
    "rank": 1,
    "rationale": "Cheap probe in highest-value region, addresses underexplored factor"
  }
}
```

### Phase 4 — Runner loop

- Runner picks next file from `todo/` (by rank)
- Executes the intervention
- Writes result to `results/`
- Moves proposal to `done/`
- Triggers world model update (orientation step)

No complex scheduling. One runner per GPU. Multiple runners can coexist by claiming files (rename = atomic on most filesystems).

## World model structure

`world_model.json` is a structured epistemic state, not a summary:

```json
{
  "version": 1,
  "updated_at": "2026-03-14T12:00:00Z",
  "beliefs": [
    {
      "id": "B1",
      "claim": "learning_rate is the dominant factor for outcome",
      "confidence": 0.82,
      "evidence_for": ["obs_001", "obs_007"],
      "evidence_against": [],
      "first_held": "2026-03-14T10:00:00Z",
      "last_tested": "2026-03-14T11:30:00Z"
    }
  ],
  "expectations": [
    {
      "if": {"DEPTH": "8", "MATRIX_LR": "0.04"},
      "then": "outcome > 1.03",
      "confidence": 0.7,
      "basis": ["B1", "obs_003"]
    }
  ],
  "tensions": [
    {
      "id": "T1",
      "beliefs": ["B1", "B3"],
      "nature": "B1 claims lr is dominant but B3 shows network_size matters at large scale",
      "salience": "high"
    }
  ],
  "cost_beliefs": {
    "cost_to_test": {"config_change": "~5 min GPU", "code_change": "~15 min GPU"},
    "cost_to_think": "~$0.05 per LLM call",
    "cost_of_being_wrong": {"B1": "high — all current experiment selection depends on it"}
  },
  "probe_fidelity": [
    {
      "probe": "short training run (100 steps)",
      "fidelity": 0.6,
      "cost_ratio": 0.1
    }
  ],
  "salience": {
    "high_learntropy": ["obs_009"],
    "unresolved_tensions": ["T1"],
    "stale_beliefs": ["B3"]
  }
}
```

## World model update (orientation step)

After each observation, the LLM performs a structured orientation:

1. **Input** (evidence-first prompt, fixed order):
   - New observation(s)
   - Current world_model.json
   - Update instructions
   - Required output schema (delta format)

2. **LLM reasons**: facts first → current state second → implications third → structured delta fourth

3. **Output**: a delta, not a rewrite:
   ```json
   {
     "beliefs_added": [],
     "beliefs_revised": [{"id": "B1", "new_confidence": 0.75, "reason": "..."}],
     "beliefs_retired": [],
     "expectations_revised": [],
     "tensions_added": [],
     "tensions_resolved": [],
     "salience_updated": {},
     "cost_beliefs_updated": {}
   }
   ```

4. Delta applied to world_model.json → new version. Previous version kept in `world_model_history/` for audit.

## One MVP cycle

```
1. Runner completes experiment → result written to results/
2. Orientation: LLM updates world_model.json (observe + orient)
3. Generator: LLM reads world_model.json, produces 5 proposals → backlog/
4. Critic: LLM reads world_model.json + backlog/, ranks, selects top 2 → todo/
5. Runner picks from todo/, executes → back to step 1
```

## Process model

Two scripts (phase 4):

**run_planner.py** — generator + critic + world model updates
- Polling loop (~60s)
- If new results in done/: run orientation step (update world model)
- If backlog + todo < threshold: run generator
- If backlog has unranked items: run critic, promote top-N to todo

**run_worker.py** — execution
- Polling loop (~30s)
- If todo/ has items: claim highest-ranked, execute, write result
- One instance per GPU

Optional later: `run_research.py` wrapper that starts both for convenience.

## What this is NOT (yet)

- Not SQLite (filesystem-first; SQLite is an implementation detail for later)
- Not a full active inference system (but rooted in the concepts)
- Not computing literal EFE (but reasoning about epistemic vs pragmatic value)
- Not multi-step planning (one cycle at a time)
- Not limited to parameter grids (intervention_type is open)

## Cost learning

Cost beliefs are **learned from observations**, not hardcoded. The system doesn't know how expensive an experiment is until it has run one.

- Each observation records actual `wall_time_s` and `compute_cost` (layer 1)
- The orientation step compares expected vs actual cost after each run
- `cost_beliefs` in the world model are revised like any other belief
- Early runs have unreliable cost estimates → critic should rank conservatively
- As runs accumulate, cost beliefs sharpen → critic can make finer trade-offs

This is essential: theorizing about which experiment to run (cost_to_think: ~$0.05 LLM call) can be orders of magnitude cheaper than running a multi-hour experiment (cost_to_test: hours of GPU). The system must learn this asymmetry from experience.

Feedback that informs cost learning:
- wall_time per intervention type
- whether the result actually produced usable information (not just "did it finish")
- ratio of information gained to compute spent

## Test plan

### v3.0 — Generator + Critic (core behavior)

Tested with synthetic or real results, manual execution.

1. **Generator produces diverse proposals**: given a world model with known tensions, generator should produce proposals that address different beliefs/tensions, not just parameter variations
2. **Rationale-first ordering**: each proposal must have epistemic_intent and rationale before intervention_spec
3. **Critic ranking is defensible**: given proposals of varying informativeness, critic should rank high-value-low-cost probes above low-value-high-cost experiments
4. **Cost awareness**: critic prefers cheap probes for shaky beliefs over expensive experiments confirming strong beliefs
5. **Not limited to parameter grid**: at least one proposal in a batch should be a non-config-change type (probe, code change, schema extension)
6. **World model update is structured**: after feeding a surprising result, the delta should revise relevant beliefs and add tensions, not just append

### v3.1 — Filesystem persistence

7. **Proposals survive restart**: write proposals, kill process, restart, proposals still in backlog/
8. **World model versioning**: after 3 updates, world_model_history/ contains versions 0-2
9. **No prompt-soup**: all structured fields (intent, rationale, intervention_spec, cost_assessment) are queryable JSON, not buried in free text

### v3.2 — Runner loop + planner

10. **Todo queue stays filled**: with planner running, todo/ should never be empty for more than one polling cycle while backlog has items
11. **End-to-end cycle**: observation → orientation → generator → critic → runner → observation completes without manual intervention
12. **Cost beliefs improve**: after 5+ runs, cost_beliefs in world model should reflect actual observed costs, not initial guesses
13. **Deliberation efficiency**: system should sometimes skip full generator→critic cycle for cheap probes when todo is nearly empty

### Real-world validation

14. **Run on GPT domain**: full cycle with train.py on dllm-experiment, compare proposal quality to v2.0's fixed 3-per-round
15. **Run on Atari domain**: same system, different domain — does the world model capture domain-specific cost structure?
16. **Cross-domain**: does cost learning from GPT runs inform cost beliefs when switching to Atari?

## Migration path

- v3.0 alone is testable: generator + critic as two LLM calls, manual execution
- v3.1 adds persistence: proposals on disk
- v3.2 adds automation: runner loop
- v4.0: SQLite for locking/querying, multiple workers, versioned world model (see full architecture doc)
