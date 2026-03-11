# autoresearcher2

An autonomous ML research agent that learns *what to try next* instead of searching blindly.

It uses a generative model of the research space — inspired by active inference and learntropy — to explore efficiently, exploit what it learns, and know when to stop.

## The Problem

Current autonomous research agents (including [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) run experiments in a semi-random loop: mutate code, train, evaluate, repeat. Each session starts from scratch. There's no cumulative learning, no principled exploration, and no way to know when the search space is sufficiently explored.

Bayesian optimization (BO) is better but treats the search space as a black box. It doesn't build *structural* knowledge ("RoPE is generally better than learned positional encodings") that transfers across campaigns.

## Three Distinct Layers

This project uses multiple layers, each serving a different purpose. They should not be confused:

| Layer | What it is | Purpose |
|---|---|---|
| **Toy validation environment** | Synthetic 27-cell POMDP | Test whether canonical active inference behaves as claimed. Theoretical sandbox only — not used in production. |
| **Proxy workload** | Cheap real ML experiments (small/short NanoGPT-style runs, ~5 min each) | Gather practical evidence cheaply. Stand-in for expensive full-scale training. |
| **Generative model** | Bayesian model over latent research-space structure | Infer hidden causes (factor effects, regimes, proxy fidelity), predict outcomes, evaluate policies. |

The toy environment validates the theory. The proxy workload provides real data cheaply. The generative model learns the hidden structure of the research space and steers exploration.

## What This Does

autoresearcher2 adds four things:

1. **A generative model** that infers latent structure (factor effects, regimes, proxy fidelity) rather than just predicting outcomes directly
2. **Policy evaluation via expected free energy** — the agent evaluates short action sequences, not just single next steps, balancing pragmatic value and epistemic value
3. **An appraisal module** that measures learntropy-like signals: how much did this result change beliefs, contradict expectations, or reorganize knowledge?
4. **Persistent memory** with activation/decay dynamics that carries episodic knowledge across experiments and campaigns

### Two Modes

**Canonical mode** (toy environment): a small POMDP where policies minimize expected free energy under an explicit generative model. Used to validate theoretical claims.

**Applied mode** (proxy workload): a larger-scale research controller that preserves the same design intention — balancing preference-seeking and information gain — but uses approximations where exact active inference becomes intractable.

The long-term ambition is not merely to borrow the language of active inference, but to progressively replace approximations with explicit generative-model components wherever tractability permits.

### The Two Loops

**Outer loop** (learntropy-driven): "What should I investigate next?"
- Controller evaluates candidate policies (2-3 step sequences) by expected free energy
- Appraisal module scores results for epistemic salience — not just "was it good?" but "did it change what I believe?"
- Memory provides episodic context to the LLM for proposal generation

**Inner loop** (val_bpb-driven): "What happened when I tried it?"
- Run a proxy workload experiment, observe outcome
- Update generative model (latent states, precisions, proxy fidelity)
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

The agent doesn't just learn "RoPE is good." It infers *why* — because it maintains beliefs about the latent regime. It can then predict what happens in unseen combinations, know when to distrust proxy results, and detect when the regime has shifted.

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

| Aspect | Canonical (toy env) | Applied (proxy workload) |
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

Standard BO and even basic active inference only ask "was the outcome good?" and "was I uncertain?" Learntropy ([Wozniak](https://supermemo.guru/wiki/Pleasure_of_learning)) suggests a richer signal: learning has an intrinsic attractiveness structure. Results that arrive at the *boundary* of current knowledge — neither obvious nor incomprehensible — drive the most learning.

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

### LLM Verbalization (Optional)

The LLM can render appraisal scores as natural language:

```json
{
  "surprise": 0.81,
  "theory_conflict": 0.74,
  "transfer_breadth": 0.68,
  "learntropy": 0.72,
  "utterance": "This overturns the assumption that optimizer choice matters more than positional encoding. Worth investigating whether this holds across context lengths."
}
```

The verbal output is a *rendering* of measured appraisal variables, not the variable itself. The agent doesn't trust "Interesting!" as a signal — it trusts the posterior change that "Interesting!" is a report of.

## Intervention Schema

A fixed, engineered factorial grid:

```python
schema = {
    "optimizer":    ["adam", "adamw", "sgd", "lion"],
    "lr_bucket":    ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"],
    "context_len":  [256, 512, 1024],
    "pos_encoding": ["rope", "learned", "alibi", "none"],
    "n_heads":      [2, 4, 8, 16],
    "n_layers":     [4, 6, 8, 12],
    "hidden_dim":   [128, 256, 512],
}
# 11,520 cells. Most will never be visited — that's the point.
# The generative model generalizes across cells via latent factor structure.
```

The LLM can suggest which cells to visit and propose schema extensions between phases, but does not define the search space.

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

### Baselines

- Random search
- Greedy exploitation (always try predicted best)
- GP-UCB (Bayesian optimization)
- ASHA / Hyperband (multi-fidelity)
- Karpathy-style autoresearch

### Falsification Criteria (preregistered)

| Criterion | Condition |
|---|---|
| Posterior calibration improves | Brier score rolling average decreases over experiment blocks |
| Exploration → exploitation shift | Predictive variance decreases while pragmatic score increases (emergent, not scheduled) |
| Beats random search | ≥20% better compute-normalized best-found val_bpb |
| Beats GP-UCB | ≥10% better or equivalent with better calibration |
| Beats ASHA/Hyperband | Comparable best-found with better sample efficiency in sparse regions |
| Cumulative regret | Lower than GP-UCB baseline |
| Memory ablation matters | +memory outperforms -memory by ≥15% (ablation) |
| Appraisal ablation matters | +appraisal outperforms -appraisal on proposal quality |
| Detects diminishing returns | Agent stops when: low uncertainty in promising regions AND expected improvement below threshold AND best-so-far plateaued AND calibration healthy AND diversity exhausted |
| Proxy fidelity is learned | Agent's proxy fidelity estimates correlate with actual proxy-to-real transfer accuracy |
| Stable across seeds | Crossover point variance <30% across 5 random seeds |

If these criteria are not met, the system does not work. No moving goalposts.

## Toy Validation Environment

A 27-cell synthetic POMDP (3 factors × 3 levels) with canonical active inference (expected free energy, policy-conditioned beliefs, explicit precision).

This exists **only to validate the theory** — it is not a proxy workload, not a component of the practical system, and not a surrogate predictor. It is a theoretical sandbox where we verify that canonical active inference produces the expected epistemic → pragmatic shift, proper calibration, and baseline-beating convergence. If these properties don't emerge in the toy environment, the applied system inherits no credibility from "active inference."

## Theoretical Roots

### Active Inference (Karl Friston)
Expected free energy decomposes into pragmatic value (preference-seeking) and epistemic value (uncertainty-reduction), or equivalently risk and ambiguity. The applied system uses the same generative-model structure — latent states, observation model, prior preferences, policy-conditioned belief updates — with approximations where exact inference is intractable. The approximations are explicit.

### Learntropy (Piotr Wozniak)
Not all surprise is equal. The brain rewards information at the *boundary* of current knowledge — structured novelty, not random noise. Learning has an intrinsic attractiveness landscape: results that are neither obvious nor incomprehensible drive the most model improvement. The appraisal module operationalizes this by measuring posterior change magnitude, theory conflict, knowledge reorganization, and transfer breadth. ([Pleasure of learning](https://supermemo.guru/wiki/Pleasure_of_learning))

### Compression Progress (Juergen Schmidhuber)
Intrinsic reward = the first derivative of model improvement. The agent should focus where its model is improving fastest, not where uncertainty is highest per se. The appraisal module's `learntropy = surprise × (1 - ambiguity)` captures this: maximum signal when the model was *confidently wrong*, not when observations are just noisy.

### Key Papers
- Friston et al. 2019 — [Generalised free energy and active inference](https://link.springer.com/article/10.1007/s00422-019-00805-w)
- Da Costa et al. 2020 — [Active inference on discrete state-spaces](https://arxiv.org/abs/2001.07203)
- Champion et al. 2024 — [Reframing the Expected Free Energy](https://arxiv.org/abs/2402.14460)
- Millidge et al. 2021 — [Whence the Expected Free Energy?](https://arxiv.org/abs/2004.08128)
- Sajid et al. 2021 — [Active inference, Bayesian optimal design, and expected utility](https://arxiv.org/abs/2110.04074)
- Li et al. 2026 — [Curiosity is Knowledge](https://arxiv.org/abs/2602.06029)
- Parr, Pezzulo & Friston 2022 — [Active Inference (MIT Press, open access)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)

## What This Is and Isn't

This project has two modes:

**Canonical mode**: a toy active-inference environment in which policies minimize expected free energy under an explicit generative model. Used to validate theoretical claims.

**Applied mode**: a larger-scale research controller that preserves the same generative-model structure — latent states, observation model, precision, policy evaluation — but uses approximations where exact active inference becomes intractable.

It is NOT:
- A replacement for domain expertise in designing the intervention schema
- Going to discover fundamentally new architectures
- Going to beat ASHA on the narrow task ASHA was designed for
- Claiming biological fidelity (neuroscience terms are metaphors, not claims)

## Implementation Phases

1. **Toy validation environment** — 27-cell synthetic POMDP with canonical active inference. Verify: calibration, crossover, baselines, stopping. Theoretical sandbox only.
2. **Generative model + controller** — Latent factor structure, precision inference, EFE-based policy evaluation over proxy ML workload. Benchmark against BO, ASHA, random.
3. **Appraisal module** — Learntropy signals: surprise, theory conflict, knowledge reorganization, transfer breadth. Validate that appraisal improves proposal quality.
4. **Memory + LLM** — Episodic memory with appraisal-weighted activation/decay. LLM proposal generation from structured context. Memory ablation study.
5. **Transfer + proxy fidelity** — Cross-campaign transfer with guards. Proxy fidelity as latent variable. Test on related task pairs.
6. **Visualization + evaluation** — Dashboard, calibration plots, regret curves, appraisal trajectories, full report.
7. **Scale validation** — Test whether knowledge from proxy workloads transfers to more expensive target runs.

## License

MIT
