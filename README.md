# autoresearcher2

A structured Bayesian research agent with learntropy-inspired appraisal. Active-inference-rooted, but approximate in production. Designed to accumulate transferable research intuition.

[![The Evolution of AI Research - from autoresearch to active inference researcher](https://img.youtube.com/vi/hGScpUahPUo/maxresdefault.jpg)](https://www.youtube.com/watch?v=hGScpUahPUo)

## Current Status

**v1 validates the core architecture on synthetic environments.** The controller, Bayesian model, appraisal module, and memory integrate correctly and outperform random and greedy baselines across 3 synthetic environments × 20 seeds. Factor structure is learned, epistemic uncertainty decreases, and learntropy marks belief-changing events. See `artifacts/v1_validation/results.json` for full results.

The intended first practical proving ground is the same target family as [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): optimizing a single-file GPT training pipeline (`train.py`) against validation bits-per-byte (val_bpb) on the [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) dataset, with ~5-minute experiments per trial. **This repository does not yet integrate with `train.py` — that is the next implementation phase.**

## Environments

Three distinct environments exist (or are planned) in this project:

| Environment | What it is | Status |
|---|---|---|
| **Toy validation** | Synthetic 27-cell POMDP | Implemented, validated |
| **Synthetic environment** | Controlled factorial simulation with known ground-truth effects | Implemented, validated — v1 exit criteria pass |
| **GPT training pipeline** | Real `train.py` edits, real val_bpb measurements | **Implemented** — SSH-based runner, tested on 2× RTX PRO 6000 |

The toy validates the theory. The synthetic validates the plumbing. The GPT pipeline is where it counts — and initial integration is working.

## The Problem

[autoresearch](https://github.com/karpathy/autoresearch) is an effective agentic system: an LLM proposes code edits to `train.py`, trains for 5 minutes, measures val_bpb, keeps improvements, reverts failures, and repeats indefinitely. The LLM brings world knowledge, reads previous results, and can reason about what to try next. This is substantially stronger than random or grid search.

But it has no structured persistent model of *why* things work. It sees a flat results log, not factor effects, interaction structure, or uncertainty estimates. It can't say "positional encoding matters more than optimizer for this architecture size" — only "this edit improved val_bpb by 0.02." Knowledge lives in the LLM's context window and the results log; structural insight doesn't accumulate across sessions.

autoresearcher2 operates on the same substrate but adds structured inference. The agent maintains a generative model of the research space — factor effects, interactions, uncertainty — and notices when its model can't explain the results. The LLM proposes what the missing dimension might be. Discovery is grounded in an explicit world model rather than implicit pattern matching in a flat log, and structural insight transfers across campaigns.

Bayesian optimization (BO) is better than unstructured search but treats the space as a black box. It doesn't build *structural* knowledge ("RoPE is generally better than learned positional encodings for this model size") that transfers to the next campaign.

## What This Does

autoresearcher2 adds four things:

1. **A generative model** that infers latent structure (factor effects, regimes, proxy fidelity) rather than just predicting outcomes directly
2. **Policy evaluation via expected free energy** — the agent evaluates short action sequences, not just single next steps, balancing pragmatic value and epistemic value
3. **An appraisal module** that measures learntropy-like signals: how much did this result change beliefs, contradict expectations, or reorganize knowledge?
4. **Persistent memory** with activation/decay dynamics that carries episodic knowledge across experiments and campaigns

### Two Modes

**Canonical mode** (toy environment): a small POMDP where policies minimize expected free energy under an explicit generative model. Used to validate theoretical claims. **Implemented.**

**Applied mode** (GPT pipeline): a research controller operating on the same substrate as autoresearch that preserves the same design intention — balancing preference-seeking and information gain — but uses approximations where exact active inference becomes intractable. **Not yet implemented — currently validated on synthetic environments only.**

The long-term ambition is not merely to borrow the language of active inference, but to progressively replace approximations with explicit generative-model components wherever tractability permits.

### The Two Loops (Design Intent)

**Outer loop** (learntropy-driven): "What should I investigate next?"
- Controller evaluates candidate interventions by expected free energy
- Appraisal module scores results for epistemic salience — not just "was the outcome good?" but "did it change what I believe about the research space?"
- Memory provides episodic context to the LLM for proposal generation

**Inner loop** (execution): "What happened when I tried it?"
- Run the selected intervention (currently: synthetic environment; planned: `train.py` edit + val_bpb measurement)
- Update generative model (factor effects, interactions, precisions)
- Appraisal module computes learntropy signals
- Store result in memory

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              LLM (Proposal Generator)                   │
│  Suggests candidate policies, interprets patterns,      │
│  proposes schema extensions. Does NOT do inference.      │
│  Receives appraisal scores + memory context as input.   │
└──────────────────┬──────────────────────────────────────┘
                   │ candidate policies (2-3 step sequences)
                   ▼
┌─────────────────────────────────────────────────────────┐
│      Layer A: Controller (EFE Policy Evaluation)        │
│  Evaluates policies by expected free energy:            │
│    G(π) = risk + ambiguity                              │
│  Simulates belief updates per step to score sequences   │
│  Selects policy with lowest EFE                         │
└──────────────────┬──────────────────────────────────────┘
                   │ selected policy (execute step by step)
                   ▼
┌──────────────┐   ┌──────────────────────────────────────┐
│  Environment │──→│  Layer B: Generative Model           │
│  (run expt)  │   │  Latent states:                      │
│              │   │    factor effects, interactions,      │
│  returns o   │   │    regime, proxy fidelity, precisions │
│              │   │  Observation model: latent → expected │
└──────────────┘   │  Posterior update after each obs      │
                   └──────────────┬───────────────────────┘
                                  │
                   ┌──────────────┼──────────────────────┐
                   ▼              ▼                       │
    ┌────────────────────┐  ┌──────────────────────────┐ │
    │  Appraisal Module  │  │  Layer C: Memory         │ │
    │  Computes:         │  │  Experiment history      │ │
    │  - surprise        │  │  Activation / decay      │ │
    │  - belief revision │  │  Pattern retrieval       │ │
    │  - knowledge reorg │  │  Feeds LLM with context  │ │
    │  - transfer breadth│  │                          │ │
    │  = learntropy score│  │                          │ │
    └────────────────────┘  └──────────────────────────┘ │
                                                         │
                   ┌─────────────────────────────────────┘
                   ▼
    ┌──────────────────────────────────────┐
    │  Transfer Subsystem                  │
    │  Cross-campaign prior blending       │
    │  Negative transfer guards            │
    │  Proxy fidelity is a latent variable │
    └──────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | Does NOT |
|---|---|---|
| **A: Controller** | Evaluates policies by EFE, selects best sequence | Know about memory, LLMs, or language |
| **B: Generative model** | Infers latent structure, predicts observations, updates beliefs | Make decisions |
| **Appraisal** | Measures epistemic salience of results (learntropy signals) | Make decisions or update beliefs |
| **C: Memory** | Stores episodic history, retrieves patterns, feeds LLM | Do inference or define latent states |
| **Transfer** | Reuses knowledge across campaigns with guards | Operate without explicit proxy fidelity estimates |

## The Generative Model

Not a predictor that maps interventions to outcomes. A model of *hidden causes* that explains why outcomes happen.

### Latent States

```python
latent_states = {
    # What structure does the research space have?
    factor_effects:   N(μ_f, Σ_f),       # "how much does each factor matter?"
    interactions:     N(μ_i, Σ_i),       # "which factor pairs interact?"
    regime:           Cat(π_regime),      # "which regime am I in?"
                                          #  e.g., "attention-dominated" vs
                                          #  "optimizer-dominated"

    # How reliable are my observations?
    outcome_noise:    Gamma(α_on, β_on),  # "how noisy are experiments?"
    proxy_fidelity:   Beta(α_pf, β_pf),  # "do proxy results predict real results?"
    proxy_per_factor: {                   # "does the proxy lie about some factors?"
        "optimizer":    Beta(α, β),
        "pos_encoding": Beta(α, β),
        ...
    },
}
```

### Observation Model

The generative model specifies *how latent states produce observations*:

```python
# Given latent states, what observations do I expect?
P(outcome | intervention, latent_states) = f(
    intervention,
    factor_effects,     # main effects
    interactions,       # pairwise interactions
    regime,             # which regime is active
    outcome_noise       # how noisy
)
```

This is the key difference from a surrogate predictor. The model doesn't just predict — it explains. When a result is surprising, the agent can ask *which latent belief was wrong*: was it the factor effect, the regime, or the proxy fidelity?

### Precision (First-Class)

Precision is not decorative. It determines how much prediction errors drive belief updates:

```python
belief_update = precision × prediction_error

# High precision → prediction errors drive large updates
# Low precision → prediction errors are discounted (noisy source)
```

Precisions are inferred, not assumed:
- **Outcome noise precision**: learned from experimental variance. Early: low (cautious updates). Later: higher (sharper learning).
- **Proxy fidelity precision**: learned by comparing proxy vs. expensive runs. If they disagree, proxy precision drops.
- **Transfer precision**: how stable is cross-campaign knowledge? Drops if transferred priors mislead.

### Why This Matters

The design aims to let the agent attribute surprising outcomes to specific latent causes — not just "RoPE is good" but "positional encoding matters more than optimizer in this regime." Whether this is achievable depends on the expressiveness and identifiability of the latent model, which is why we validate on the toy environment first.

## Policy Evaluation (Expected Free Energy)

The controller evaluates short policies (2-3 step sequences), not just single next actions.

```python
candidate_policies = [
    # Policy 1: explore unknown factor combination
    [try(rope, lion, lr=1e-3), try(rope, lion, lr=3e-3)],

    # Policy 2: test proxy fidelity for n_layers
    [try_proxy(12_layers, rope), try_expensive(12_layers, rope)],

    # Policy 3: exploit known good region
    [try(rope, adamw, lr=3e-4), try(rope, adamw, lr=1e-3)],
]

for π in candidate_policies:
    G(π) = Σ_t [
        risk_t(π)       # D_KL[ Q(o_t|π) ‖ P(o_t) ] — distance from preferred outcomes
      + ambiguity_t(π)  # E_Q(s_t|π)[ H[P(o_t|s_t)] ] — expected observation noise
    ]
    # Crucially: beliefs at step 2 are conditioned on what we'd learn at step 1

best_policy = argmin(G)
```

A one-step agent can't plan "run a cheap experiment to learn about transfer, then use that to pick a better expensive experiment." A policy-planning agent can. This is **active belief shaping** — the agent designs experiments to improve its own model, not just to find good outcomes.

### Where canonical, where approximate

| Aspect | Canonical (toy env) | Applied (GPT pipeline) |
|---|---|---|
| Latent states | Full discrete POMDP | Factored Gaussians + categorical regime |
| Policy horizon | Full planning horizon | 2-3 steps |
| EFE computation | Exact | Simulated belief trajectories |
| Precision | Full Bayesian | Gamma priors, point estimates where needed |
| Proxy fidelity | N/A | Latent Beta variable |

The seam between canonical and approximate is explicit. Where the latent space is tractable, we use faithful active inference. Where it isn't, we approximate and document the deviation.

## Appraisal Module (Learntropy)

After each experiment, the appraisal module computes *epistemic salience* — how much this result matters for the agent's understanding, not just whether the outcome was good.

### Why This Exists

Standard BO asks "was the outcome good?" and "where am I uncertain?" Learntropy ([Wozniak](https://supermemo.guru/wiki/Pleasure_of_learning)) suggests a richer signal: learning has an intrinsic attractiveness structure. Results that arrive at the *boundary* of current knowledge — neither obvious nor incomprehensible — drive the most learning.

The appraisal module makes this measurable.

### Signals

```python
appraisal = {
    # How much did beliefs change?
    surprise:           KL(posterior_after ‖ posterior_before),

    # Did this contradict a strong prior belief?
    theory_conflict:    max(prior_confidence × prediction_error),

    # How many other beliefs were affected?
    knowledge_reorg:    count(beliefs_revised > threshold),

    # Could this insight transfer to other regions of the schema?
    transfer_breadth:   count(cells_whose_prediction_changed > threshold),

    # Is this in the learntropy sweet spot?
    # (not obvious, not incomprehensible)
    learntropy:         surprise × (1 - ambiguity),
    # High surprise + low ambiguity = the model was confidently wrong
    #   → maximum learning signal
    # High surprise + high ambiguity = noisy, can't tell what happened
    #   → low learning signal
    # Low surprise = expected, nothing to learn
    #   → low learning signal
}
```

### How Appraisal Feeds the System

- **Feeds the LLM**: appraisal scores are included in the context for proposal generation. The LLM sees: "Experiment 47 had high theory_conflict — the belief that optimizer matters more than pos_encoding was contradicted." This guides the LLM toward proposing follow-up experiments that resolve the conflict.
- **Feeds memory activation**: high-appraisal results get a boost in memory activation. They persist longer and surface more readily. Low-appraisal results (boring, expected) decay faster.
- **Does NOT feed the controller directly**: the controller operates on EFE over the generative model. Appraisal is a *reporting* layer, not a decision layer. It prevents the "memory quietly becomes the generative model" failure.
- **Feeds visualization**: appraisal scores over time show whether the agent is in the learntropy sweet spot — learning efficiently at the boundary of its knowledge — or stuck in obvious/incomprehensible territory.

### LLM Verbalization (Reporting Channel, Not Control Signal)

LLM verbal reactions may contain useful epistemic appraisal — "Interesting, that changes everything" could reflect genuine surprise at a posterior shift. But surface wording alone is too style-contaminated to trust directly. LLMs are trained for social fluency; "Interesting!" can be cheap style, not genuine epistemic rupture.

Therefore the system grounds appraisal in evidence and posterior change, then allows the LLM to express it richly in language:

```json
{
  "surprise": 0.81,
  "theory_conflict": 0.74,
  "transfer_breadth": 0.68,
  "learntropy": 0.72,
  "utterance": "This overturns the assumption that optimizer choice matters more than positional encoding. Worth investigating whether this holds across context lengths."
}
```

The verbal output is a *rendering* of measured appraisal variables, not the variable itself. This keeps the biological and cognitive inspiration alive without collapsing into ungrounded vibes.

## Intervention Schema

The schema is a fixed, engineered factorial grid. Each cell maps to a specific combination of factor levels. The generative model reasons about factor effects and interactions across cells.

**v1 synthetic validation** uses a small schema (3 factors × 3 levels = 27 cells) to validate the architecture:

```python
schema = InterventionSchema(factors={
    "optimizer": ["adam", "adamw", "sgd"],
    "lr": ["1e-4", "3e-4", "1e-3"],
    "batch_size": ["32", "64", "128"],
})
```

**Current `train.py` schema** (implemented, running on real hardware):

```python
schema = InterventionSchema(factors={
    "DEPTH": ["6", "8", "10"],
    "MATRIX_LR": ["0.02", "0.04", "0.08"],
    "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
})
# 27 cells, ~5 min per experiment, val_bpb ~1.0-1.1
```

The key idea: autoresearch edits anything in the file; autoresearcher2 maps edits to a structured factor space so it can reason about effects, interactions, and uncertainty. The first real run confirmed this works — the Bayesian model receives real val_bpb measurements and updates factor posteriors accordingly.

## Memory System

The generative model infers *latent structure* (factor effects, regimes, precisions). Memory stores *episodic knowledge* that the generative model can't capture: which experiments were run, what the LLM's reasoning was, which combinations were tried and abandoned, and why.

### What memory does that the generative model doesn't

| Need | Generative model | Memory |
|---|---|---|
| "Is RoPE generally good?" | Yes — it's a latent factor effect | No |
| "Did I already try this exact config?" | No — it has no episodic recall | Yes — deduplication |
| "What happened last time I explored this region?" | No | Yes — retrieval by similarity |
| "What should the LLM know for its next proposal?" | No — it outputs distributions, not context | Yes — structured summaries |
| "Was I calibrated last week?" | No — it only has current posterior | Yes — historical tracking |
| "Which results were most surprising?" | No | Yes — via appraisal scores |

### How memory interacts with the other layers

- **Feeds LLM**: memory provides episodic context + appraisal history for proposal generation.
- **Informs transfer**: campaign-level summaries and calibration history for the transfer subsystem.
- **Receives from appraisal**: high-appraisal results get activation boosts; low-appraisal results decay faster.
- **Does NOT feed Layer A or B directly**: the controller and generative model operate on the structured posterior. Memory influences decisions only indirectly, through LLM proposals and transfer priors.

### Dynamics (neuroscience as metaphor, not claim)

- **Activation decay** (Ebbinghaus): old, unretrieved results lose accessibility. The system naturally forgets dead ends.
- **Recall boost** (spaced repetition): results that keep being useful persist longer.
- **Association** (Hebbian): experiments that co-occur in successful runs link together.
- **Appraisal-weighted**: high-learntropy results start with higher activation and decay slower.

## Transfer Across Campaigns

Transfer is a first-class subsystem. Proxy fidelity is a latent variable in the generative model, not an assumption.

```python
# What transfers: factor effect posteriors, interaction structure, regime beliefs
# What the agent can also learn about transfer itself:
#   proxy_fidelity — "do small-run results predict large-run results?"
#   proxy_per_factor — "does the proxy lie about some factors more than others?"

# The agent can actively test transfer by running:
#   [proxy_experiment, expensive_validation]
# and updating proxy_fidelity from the comparison.

# Guards against negative transfer:
#   - Schema must be compatible
#   - Tasks must be similar (metadata check)
#   - Source must have been well-calibrated (Brier score check)
#   - Proxy fidelity must be above threshold
```

## Evaluation

### v1 Synthetic Results (proven)

v1 has been validated on 3 synthetic environments × 20 seeds × 50 experiments each. All exit criteria pass:

| Metric | env_a (main effects) | env_b (interaction) | env_c (high noise) |
|---|---|---|---|
| Beats random (regret) | 6.5 vs 17.5 | 5.4 vs 13.4 | 7.6 vs 15.9 |
| Competitive w/ greedy | 6.5 vs 9.4 | 5.4 vs 7.2 | 7.6 vs 9.3 |
| Factor rank accuracy | 95% | 100% | 95% |
| Epistemic decrease | 100% | 100% | 100% |

Full results: `artifacts/v1_validation/results.json`. Runner: `scripts/run_v1_synthetic_validation.py`.

### Baselines

**Implemented (v1):**
- **Random search** — random schema cells
- **Greedy exploitation** — always try predicted best

**Planned (not yet implemented):**
- **GP-UCB** — Bayesian optimization (black-box, no factor structure)
- **ASHA / Hyperband** — multi-fidelity scheduling
- **autoresearch-style** — LLM agentic loop with results log (the direct comparison)

### Falsification Criteria

These apply to the **future `train.py` integration**, not to synthetic validation:

| Criterion | Condition |
|---|---|
| Beats random search | ≥20% better compute-normalized best-found val_bpb |
| Beats autoresearch-style | Better val_bpb at same experiment count, or same val_bpb with fewer experiments |
| Structural knowledge | Agent correctly identifies top-2 factor effects |
| Stable across seeds | Crossover point variance <30% across 5 random seeds |

These criteria cannot be evaluated until `train.py` integration exists. No claiming victory on synthetic alone.

## Toy Validation Environment

A 27-cell synthetic POMDP (3 factors × 3 levels) with canonical active inference (expected free energy, policy-conditioned beliefs, explicit precision).

This exists **only to validate the theory** — it is not the GPT pipeline, not a component of the practical system, and not a surrogate predictor. It is a theoretical sandbox where we verify that canonical active inference produces the expected epistemic → pragmatic shift, proper calibration, and baseline-beating convergence. If these properties don't emerge in the toy environment, the applied system inherits no credibility from "active inference."

## Theoretical Roots and Deliberate Departures

This project draws on three theoretical traditions. It takes them seriously — not as decoration, but as design constraints. Where the current system departs from canonical formulations, the departure is deliberate and documented.

### Active Inference (Karl Friston)

**Contribution:** The core idea that experiment selection should unify epistemic value (uncertainty reduction) and pragmatic value (preference-seeking) under a single objective — expected free energy. An agent with a generative model of hidden causes can evaluate policies by asking: "will this action sequence move me toward preferred outcomes *and* teach me something about the world?"

**Current status:** The applied system uses the same generative-model structure — latent states, observation model, prior preferences, policy-conditioned belief updates — with approximations where exact inference is intractable. The toy validation environment uses canonical active inference without approximation.

**Deliberate departure:** The applied system's EFE-based policy evaluation over short horizons is an approximation of full active inference, not a repudiation of it. The long-term goal is to progressively replace heuristics with explicit generative-model components wherever tractability permits. The seam between canonical and approximate is part of the contribution, not an embarrassment.

### Learntropy (Piotr Wozniak)

**Contribution:** The idea that some learning opportunities are intrinsically attractive — not because they are novel per se, but because they are *surprising, decodable, structurally meaningful, and growth-inducing*. ([Pleasure of learning](https://supermemo.guru/wiki/Pleasure_of_learning))

**Why learntropy is not the same as novelty:** Random novelty is not enough. A completely unfamiliar, chaotic signal is novel but has zero learntropy — the system can't decode it, can't connect it to existing knowledge, can't use it to grow. Learntropy requires a zone where the signal is surprising *but decodable* and *structurally useful*. This is why the appraisal module needs both positive terms (surprise, transfer breadth) and negative terms (ambiguity, noise). The sweet spot is: confidently wrong → maximum learning. Hopelessly confused → no learning. Already knew that → no learning.

**Current status:** The appraisal module operationalizes learntropy by measuring posterior change magnitude, theory conflict, knowledge reorganization, and transfer breadth. The system does not yet directly optimize learntropy as an objective — it uses it as a reporting and memory-weighting signal. Moving learntropy into the EFE computation itself is an open research direction.

### Compression Progress (Juergen Schmidhuber)

**Contribution:** Intrinsic reward = the first derivative of model improvement. The agent should focus where its model is improving fastest, not where uncertainty is highest per se. The appraisal module's `learntropy = surprise × (1 - ambiguity)` captures this: maximum signal when the model was *confidently wrong*, not when observations are just noisy.

### What the practical system is today

The practical controller is an approximation designed for tractability, not a repudiation of the theory:

- Active inference contributes the generative-model structure and the EFE decomposition
- Learntropy contributes the appraisal module and the insight that epistemic value has internal structure (not all uncertainty reduction is equally valuable)
- The current system is not yet canonical active inference and does not yet directly optimize learntropy
- The toy validation environment is the canonical reference; the GPT pipeline system is the working approximation
- The gap between them is where the research contribution lives
- The first proving ground — same substrate as autoresearch — is intentionally chosen to make the comparison concrete

### Key Papers
- Friston et al. 2019 — [Generalised free energy and active inference](https://link.springer.com/article/10.1007/s00422-019-00805-w)
- Da Costa et al. 2020 — [Active inference on discrete state-spaces](https://arxiv.org/abs/2001.07203)
- Champion et al. 2024 — [Reframing the Expected Free Energy](https://arxiv.org/abs/2402.14460)
- Millidge et al. 2021 — [Whence the Expected Free Energy?](https://arxiv.org/abs/2004.08128)
- Sajid et al. 2021 — [Active inference, Bayesian optimal design, and expected utility](https://arxiv.org/abs/2110.04074)
- Li et al. 2026 — [Curiosity is Knowledge](https://arxiv.org/abs/2602.06029)
- Parr, Pezzulo & Friston 2022 — [Active Inference (MIT Press, open access)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)

## What This Is and Isn't

This is a serious attempt to bring active inference and learntropy closer to an applied self-improving researcher. Currently validated on synthetic environments; the real test — same substrate as autoresearch — hasn't happened yet.

**The core bet**: an agent that maintains a structured model of factor effects, uncertainty, and epistemic value — and uses learntropy-inspired appraisal to weight what's worth learning from — will outperform a flat agentic edit loop on the same GPT training pipeline. **This bet is not yet tested on real substrates.**

**What works today**: the architecture is coherent. On synthetic environments with known ground truth, the Bayesian model learns factor structure, the controller outperforms random and greedy baselines, and the appraisal module marks belief-changing events.

**What's new**: `train.py` integration via SSH runner, LLM proposal generation via `claude -p`, two-step policy lookahead, and memory-grounded LLM context. Three experiment loops completed on RTX PRO 6000 Blackwell GPUs. The LLM v2 loop (lookahead + enhanced prompt with learntropy-ranked experiments and coverage gaps) reached val_bpb=1.034 by experiment 5 — the same quality that LLM v1 needed 15 experiments to achieve. Best overall: val_bpb=1.0331 (DEPTH=8, MATRIX_LR=0.04, WEIGHT_DECAY=0.2). A head-to-head comparison with Karpathy's autoresearch-style baseline remains the next milestone.

It is NOT:
- Pure active inference (yet)
- Just BO with memory
- Claiming generality across domains — the first proving ground is deliberately narrow
- Going to discover fundamentally new architectures *within* a fixed schema
- Claiming biological fidelity (neuroscience terms are metaphors, not claims)
- **Claiming to work on real substrates until a real run proves it**

## Implementation Status

### v1 — Synthetic Validation (complete)

v1 validates the core architecture on synthetic environments. It proves the plumbing works.

**v1 includes:**
- Bayesian linear model with conjugate updates over one-hot factor features
- Thompson-sampling controller with EFE-style diagnostics
- Learntropy-inspired appraisal (surprise, theory_conflict, prediction_impact_breadth, learntropy)
- Simple episodic memory with appraisal-weighted retrieval
- Baseline evaluation harness: random, greedy
- Toy validation environment (27-cell canonical active inference)
- 3 synthetic environments with known ground truth, validated across 20 seeds
- All exit criteria passing

**v1 does NOT include:**
- GP-UCB / ASHA baselines
- Transfer across campaigns
- Multi-step policy planning
- Memory dynamics (activation decay, Hebbian linking)

### v1.5 — Real Substrate + LLM Integration (in progress)

Building on v1, this phase connects the Bayesian engine to real GPU training:

**v1.5 includes:**
- `TrainPyEnvironment`: SSH-based runner that patches `train.py` knobs, runs real training, parses val_bpb
- LLM proposal module: `claude -p` generates experiment suggestions based on history + factor importances
- Dual-GPU parallel execution for 2× throughput
- Two-step policy lookahead: simulates belief updates to value information gain
- Memory-grounded LLM context: high-signal experiments (by learntropy) + coverage gaps
- 88 tests passing (69 core + 16 LLM + 3 lookahead)
- **Three experiment loops completed**: pure Bayesian, LLM v1, LLM v2 (lookahead + enhanced prompt)

**First real loop results** (15 experiments, pure Bayesian Thompson sampling):

| Metric | Value |
|---|---|
| Best val_bpb | 1.053 (DEPTH=6, MATRIX_LR=0.02, WEIGHT_DECAY=0.2) |
| Worst val_bpb | 1.081 |
| Factor importances | DEPTH=0.302, MATRIX_LR=0.096, WEIGHT_DECAY=0.035 |
| Unique cells visited | 5 / 27 |
| Failures | 0 / 15 |
| Total time | 83 min (~5.5 min per experiment) |

Full results: `artifacts/trainpy_loop/first_real_loop.json`

**Three-way comparison** (pure Bayesian vs LLM v1 vs LLM v2):

| Metric | Pure Bayesian | LLM v1 | LLM v2 |
|---|---|---|---|
| Best val_bpb | 1.0530 | 1.0347 | **1.0331** |
| Mean val_bpb | 1.0656 | 1.0514 | **1.0491** |
| Experiments | 15 | 20 | 20 |
| Unique cells | 5 / 27 | 12 / 27 | **13 / 27** |
| Total time | 83 min | 108 min | 109 min |

**Convergence speed** (best val_bpb reached by experiment N):

| Experiment | Pure Bayesian | LLM v1 | LLM v2 |
|---|---|---|---|
| 5 | 1.0530 | 1.0532 | **1.0339** |
| 10 | 1.0530 | 1.0455 | **1.0339** |
| 15 | 1.0530 | 1.0384 | **1.0336** |
| 20 | — | 1.0347 | **1.0331** |

The LLM v2 loop (two-step lookahead + memory-grounded prompt) found val_bpb=1.034 by experiment 5 — it took LLM v1 until experiment 15 to reach the same quality. The key improvements:

1. **Two-step lookahead** values information gain: the controller picks experiments that improve future decisions, not just the immediate outcome
2. **Coverage gap detection** in the LLM prompt: Claude sees which factor levels have never been tested and targets them
3. **High-signal experiment context**: the LLM receives the top experiments ranked by learntropy, focusing its reasoning on the most informative results

Best config found: **DEPTH=8, MATRIX_LR=0.04, WEIGHT_DECAY=0.2** (val_bpb=1.0331, LLM suggestion). All top-6 configs have DEPTH=8 — the LLM discovered this sweet spot when the pure Bayesian loop had only explored DEPTH=6 and DEPTH=10.

**Source breakdown** (LLM v2):
- LLM suggestions: 8 runs, mean val_bpb = 1.0446
- Lookahead: 12 runs, mean val_bpb = 1.0521

Full results: `artifacts/trainpy_llm_loop/results.json`, `artifacts/trainpy_llm_loop/results_v1.json`, `artifacts/comparison.json`

**v1.5 does NOT yet include:**
- Full head-to-head comparison with autoresearch (different substrate approach)
- GP-UCB / ASHA baselines
- Transfer across campaigns

### Next: Head-to-Head Comparison with autoresearch

Three loops are complete with real artifacts. Next steps:
- Run autoresearch-style baseline (unstructured LLM edit loop) on the same hardware + budget
- Head-to-head: autoresearcher2 vs autoresearch on compute-normalized val_bpb
- Evaluate falsification criteria on real substrates
- Schema extension: add more `train.py` knobs (n_heads, context_length, etc.)

## Implementation Phases (Full Roadmap)

1. **Toy canonical validation** — 27-cell synthetic POMDP with canonical active inference. Verify: calibration, epistemic→pragmatic shift, baselines. Theoretical sandbox only.
2. **Synthetic plumbing validation** — Controlled factorial environment with known ground-truth effects. Verify that controller, model, appraisal, and memory integrate correctly. No real training runs.
3. **GPT pipeline integration** — Connect to the same substrate as autoresearch: `train.py` edits → val_bpb measurement → structured model update. Schema maps to real `train.py` levers.
4. **Head-to-head comparison** — Run autoresearcher2 and autoresearch-style baseline on the same GPT pipeline, same compute budget. Measure val_bpb convergence, experiment efficiency, structural knowledge quality.
5. **Appraisal + memory** — Learntropy signals feed memory weighting and LLM proposal context. Ablation studies: +/- appraisal, +/- memory.
6. **Stronger baselines** — Add GP-UCB, ASHA/Hyperband for rigorous comparison beyond autoresearch-style.
7. **Transfer + proxy fidelity** — Cross-campaign transfer with guards. Test whether structural knowledge from one `train.py` campaign improves the next.
8. **Visualization + evaluation** — Dashboard, calibration plots, regret curves, appraisal trajectories, full report.
9. **Scale validation** — Test whether knowledge from 5-minute proxy experiments transfers to longer, more expensive training runs.

## License

MIT
