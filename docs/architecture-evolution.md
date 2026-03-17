# Architecture Evolution: v1 Design vs v4 Reality

This document records the significant architectural shifts between the original design
(documented in the README and design docs from v1/v1.5) and what was actually built
by v4.11. Neither direction is "wrong" — the system evolved pragmatically as we learned
what worked. But the delta is large enough to document explicitly.

## Summary

The original design was a mathematically grounded active inference system with a Bayesian
linear model, EFE-based policy evaluation, and structured factor schemas. The actual system
became an LLM-native generator-critic pipeline with a natural-language world model, code-change
proposals, and multi-project scheduling. The theoretical roots remain relevant as design
inspiration, but the implementation took a fundamentally different path.

## What Changed

### 1. World Model: Bayesian Linear Model → LLM-Managed Beliefs

**Designed (v1):**
```python
latent_states = {
    factor_effects:   N(μ_f, Σ_f),
    interactions:     N(μ_i, Σ_i),
    regime:           Cat(π_regime),
    outcome_noise:    Gamma(α_on, β_on),
    proxy_fidelity:   Beta(α_pf, β_pf),
}
```
A conjugate Bayesian model with explicit posterior updates. Factor effects were Gaussian
distributions, regimes were categorical, precision was a first-class inferred variable.

**Built (v4):**
```json
{
  "beliefs": [
    {"id": "B1", "claim": "DEPTH=8 is optimal for val_bpb", "confidence": 0.85},
    {"id": "B2", "claim": "higher MATRIX_LR trades stability for speed", "confidence": 0.6}
  ],
  "tensions": [
    {"id": "T1", "description": "B1 contradicts recent observation at DEPTH=10"}
  ],
  "cost_beliefs": {"config_change": {"wall_time_s": 300}}
}
```
A structured JSON document where beliefs are natural-language claims with confidence scores.
The LLM updates this via structured deltas after each observation. No explicit probability
distributions — confidence is a scalar the LLM sets based on evidence.

**Why this happened:** The Bayesian model worked well on synthetic environments with known
factorial structure, but real research problems don't have clean factor grids. code_change
proposals (where the LLM rewrites arbitrary training code) can't be mapped to a fixed
intervention schema. The LLM-managed world model is more flexible and domain-agnostic.

**What was lost:** Mathematical guarantees about convergence, calibration, and uncertainty
quantification. The Bayesian model provably reduces epistemic uncertainty; the LLM world
model relies on the LLM's judgment about confidence.

**What was gained:** Domain generality. The same world model structure works for NanoGPT
training, Atari RL, and any future domain without redesigning the factor schema.

### 2. Experiment Selection: EFE Controller → Generator-Critic Pipeline

**Designed (v1):**
```python
for π in candidate_policies:
    G(π) = Σ_t [risk_t(π) + ambiguity_t(π)]
best_policy = argmin(G)
```
Expected free energy decomposed into risk (pragmatic) and ambiguity (epistemic).
Thompson sampling with two-step lookahead simulating belief updates.

**Built (v4):**
- **Generator**: LLM proposes N experiments with rationale-first justification
- **Critic**: Separate LLM call ranks proposals ordinally, accepts/rejects
- Selected proposals enter a queue executed by domain-specific workers

**Why this happened:** EFE computation requires a tractable generative model. Once the
world model became LLM-managed natural-language beliefs (shift #1), there's no distribution
to compute KL divergence over. The generator-critic pattern is the LLM-native equivalent:
the generator's rationale serves the same role as EFE (why is this experiment worth running?),
and the critic provides the selection pressure.

**What was lost:** Principled explore/exploit balance. EFE automatically trades off epistemic
and pragmatic value. The generator-critic relies on the LLM's judgment about this tradeoff.

**What was gained:** The ability to handle open-ended proposals (code changes, not just
schema cells) and cross-domain reasoning.

### 3. Intervention Model: Fixed Schema → Code Changes

**Designed (v1):**
```python
schema = InterventionSchema(factors={
    "DEPTH": ["6", "8", "10"],
    "MATRIX_LR": ["0.02", "0.04", "0.08"],
    "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
})
# 27 cells, each a specific combination of factor levels
```
A fixed factorial grid. The system reasons about factor effects and interactions across cells.

**Built (v4):**
Proposals can be `config_change` (modify known parameters) or `code_change` (LLM rewrites
the training script). For Atari Breakout, the LLM writes entirely new training code with
different network architectures, reward shaping, frame processing, etc.

**Why this happened:** The schema is powerful for hyperparameter optimization but can't
express structural changes (e.g., "try a different positional encoding" or "add frame
stacking to the Atari agent"). code_change proposals let the LLM explore a much larger
space. The schema still exists as a concept within NanoGPT config_change proposals but
is no longer the only intervention type.

### 4. Storage: JSON Files → SQLite

**Designed (v1):** Filesystem-first persistence. `world_model.json`, proposal files in
`backlog/todo/done/` directories, observation JSON files.

**Built (v4):** SQLite database (`research_v4.db`) with tables for projects, observations,
proposals, world models, and world model history. Versioned world models with full delta
history.

**Why this happened:** Multi-project support requires relational queries (which observations
belong to which project? what's the world model for project X?). The web UI needs efficient
queries for the Kanban board and stats. File-per-entity doesn't scale to hundreds of
observations across multiple projects.

### 5. Scope: Single-Domain → Multi-Project

**Designed (v1):** One research campaign at a time (GPT training optimization).

**Built (v4):** Multiple concurrent projects with different domains:
- NanoGPT (val_bpb, minimize)
- Atari Breakout (mean_reward, maximize)
- Priority scheduling (Auto/Exclusive/High/Normal/Low/Paused)
- Dispatch executor routing proposals to domain-specific executors

### 6. Interface: CLI-Only → Web UI

**Designed (v1):** Scripts and CLI tools. `run_v1_synthetic_validation.py`, etc.

**Built (v4):**
- Next.js frontend with Kanban board, stats bar, project filter
- FastAPI backend serving API + static frontend
- PDF report generation
- Priority dropdown per project with learntropy-based auto scheduling

## What Was Dropped (Not Yet Built)

These were planned but not implemented. Some may still be valuable; others may no longer
make sense given the architectural shifts.

| Feature | Original plan | Status | Still relevant? |
|---|---|---|---|
| **EFE policy evaluation** | Exact in toy, approximate in applied | Replaced by generator-critic | The *intent* (balance explore/exploit) is still served, just differently |
| **Memory activation/decay** | Ebbinghaus decay, Hebbian linking, spaced repetition | Not implemented | Maybe — could help prioritize which observations to show the LLM |
| **Transfer subsystem** | Cross-campaign prior blending with proxy fidelity guards | Not implemented | Yes — multi-project makes this more relevant, not less |
| **Proxy fidelity** | Latent Beta variable learned by comparing cheap/expensive runs | Not implemented | Yes — relevant for scaling from 5-min to longer experiments |
| **GP-UCB / ASHA baselines** | Stronger baselines for comparison | Not implemented | Less relevant — the system isn't doing fixed-schema optimization anymore |
| **Calibration plots** | Brier scores, reliability diagrams | Not implemented | Would be valuable for evaluating LLM confidence accuracy |
| **Multi-step policy planning** | 2-3 step lookahead with belief simulation | Not implemented | Could still work if generator proposes experiment sequences |

## What Was Added (Not Originally Planned)

| Feature | What it does | Why it matters |
|---|---|---|
| **code_change proposals** | LLM rewrites training scripts, not just hyperparameters | Vastly larger intervention space |
| **Multi-project scheduling** | Priority-based slot allocation across concurrent projects | Enables cross-domain research |
| **Web UI** | Kanban board, stats, PDF reports, priority controls | Makes the system observable and steerable |
| **Energy cost tracking** | GPU power monitoring → EUR cost per experiment | Real accountability for compute spend |
| **Dispatch executor** | Routes proposals to domain-specific executors (NanoGPT, Atari) | Clean separation of execution concerns |
| **Learntropy-based priority** | expected_gain computed from belief changes, uncertainty, tensions | Auto-prioritizes projects with most learning potential |

## The Deeper Shift

The original design treated the LLM as a **proposal generator feeding a mathematical
controller**. The LLM suggests; the Bayesian model decides.

The actual system treats the LLM as the **primary reasoning engine**. The LLM maintains
the world model, generates proposals with rationale, and critiques them. The mathematical
components (learntropy computation, expected_gain) are lightweight scoring functions that
inform scheduling, not the core decision loop.

This isn't necessarily worse — it's a different bet. The original bet was that explicit
Bayesian structure would outperform pure LLM reasoning. The current bet is that LLM
reasoning with structured memory (world model, observations, beliefs) outperforms LLM
reasoning with flat logs. The theoretical roots (active inference, learntropy) still
inform the design — beliefs, tensions, surprise, confidence — but through natural language
rather than probability distributions.

The v1 synthetic validation results (Bayesian model beating baselines) are still valid.
The question is whether those advantages carry over when the intervention space becomes
open-ended (code changes, not schema cells). That question is what v4 is testing in
production.
