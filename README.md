# autoresearcher2

An autonomous ML research agent that learns *what to try next* instead of searching blindly.

It uses structured Bayesian experiment selection with persistent memory — inspired by active inference and learntropy — to explore efficiently, exploit what it learns, and know when to stop.

## The Problem

Current autonomous research agents (including [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) run experiments in a semi-random loop: mutate code, train, evaluate, repeat. Each session starts from scratch. There's no cumulative learning, no principled exploration, and no way to know when the search space is sufficiently explored.

Bayesian optimization (BO) is better but treats the search space as a black box. It doesn't build *structural* knowledge ("RoPE is generally better than learned positional encodings") that transfers across campaigns.

## Three Kinds of Simplification

This project uses multiple layers, each serving a different purpose. They should not be confused:

| Layer | What it is | Purpose |
|---|---|---|
| **Toy validation environment** | Synthetic 27-cell POMDP | Test whether canonical active inference behaves as claimed. Theoretical sandbox only — not used in production. |
| **Proxy workload** | Cheap real ML experiments (small/short NanoGPT-style runs, ~5 min each) | Gather practical evidence cheaply. Stand-in for expensive full-scale training. |
| **Surrogate predictor** | Bayesian linear model learned from experiment history | Predict outcomes and choose the next experiment. Generalizes across the intervention space. |

The toy environment validates the theory. The proxy workload provides real data cheaply. The surrogate predictor learns from that data and steers exploration.

## What This Does

autoresearcher2 adds three things:

1. **A surrogate predictor** that learns factor-level effects and interactions across the intervention space — not per-cell, but across the whole schema
2. **Thompson sampling** for experiment selection — exploration emerges naturally from posterior uncertainty, no hand-tuned explore/exploit schedule
3. **Persistent memory** with activation/decay dynamics that carries knowledge across experiments and campaigns

### The Two Loops

**Outer loop** (learntropy-inspired): "What should I investigate next?"
- Thompson sampling from the surrogate posterior drives exploration when uncertain, exploitation when confident
- Memory consolidation builds cumulative understanding
- LLM proposes candidate interventions from structured context

**Inner loop** (val_bpb-driven): "What happened when I tried it?"
- Run a proxy workload experiment, observe outcome
- Update surrogate posterior (exact Bayesian update)
- Store result in memory

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           LLM (Proposal Generator)                  │
│  Suggests cells to evaluate, interprets patterns,   │
│  proposes schema extensions. Does NOT do inference.  │
└──────────────────┬──────────────────────────────────┘
                   │ candidate cells
                   ▼
┌─────────────────────────────────────────────────────┐
│     Layer A: Controller (Thompson Sampling)         │
│  Samples from surrogate posterior                   │
│  Selects cell with highest sampled expected utility │
│  No explicit explore/exploit weights                │
└──────────────────┬──────────────────────────────────┘
                   │ selected intervention
                   ▼
┌──────────────┐   ┌──────────────────────────────────┐
│  Environment │──→│  Layer B: Surrogate Model        │
│  (run expt)  │   │  Bayesian linear + interactions  │
│              │   │  Posterior update after each obs  │
│  returns o   │   │  Provides: predictions, variance,│
│              │   │  factor importances, residuals    │
└──────────────┘   └──────────────┬───────────────────┘
                                  │ structured outputs
                                  ▼
                   ┌──────────────────────────────────┐
                   │  Layer C: Memory + Summarization │
                   │  Experiment history + activation  │
                   │  Pattern tracking, dedup, decay   │
                   │  Feeds LLM with structured context│
                   └──────────────────────────────────┘
```

### Three Layers (cleanly separated)

| Layer | Responsibility | Does NOT |
|---|---|---|
| **A: Controller** | Selects next experiment via Thompson sampling | Know about memory, LLMs, or language |
| **B: Surrogate predictor** | Predicts outcomes, provides uncertainty estimates | Make decisions |
| **C: Memory** | Stores history, retrieves patterns, feeds LLM | Do inference or define latent states |

Plus a **Transfer** subsystem for reusing knowledge across campaigns (with negative-transfer guards).

## The Surrogate Predictor

A Bayesian linear model over one-hot encoded factors with pairwise interactions:

```python
# Feature vector: ~30 main effects + ~200 interactions ≈ 230 parameters
x(c) = [one_hot(optimizer), one_hot(lr), ...,
         one_hot(optimizer) ⊗ one_hot(lr), ...]

# Posterior over weights
W ~ N(μ_W, Σ_W)

# Predictions
P(outcome | c) = softmax(x(c)^T W + b)

# Uncertainty (analytic)
predictive_variance(c) = x(c)^T Σ_W x(c)
```

Why Bayesian linear:
- Learns "RoPE is good" (factor effect), not just "RoPE + adam + lr=3e-4 is good" (one cell)
- Exact conjugate updates — no retraining
- Analytic uncertainty — no sampling for variance estimates
- Interpretable — read off factor importances directly
- Upgrade path to random forest, GP, or neural surrogate when residuals demand it

## Thompson Sampling (Why Not Acquisition Function Weights)

Instead of `score = w_explore × uncertainty + w_exploit × quality` (where the crossover is a hyperparameter artifact), we use Thompson sampling:

```python
def select_next_experiment(surrogate, preferences):
    W_sample = sample_normal(surrogate.μ_W, surrogate.Σ_W)

    for c in candidate_cells:
        pred = softmax(x(c) @ W_sample)
        scores[c] = dot(preferences, pred)

    return argmax(scores)
```

- **Early** (high Σ_W): posterior samples are wild → different cells win each time → exploration
- **Late** (low Σ_W): samples converge → best cells consistently win → exploitation
- The shift from exploration to exploitation **emerges from Bayesian updating**, not from a schedule

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
# The surrogate generalizes across cells via factor-level learning.
```

The LLM can suggest which cells to visit and propose schema extensions between phases, but does not define the search space.

## Memory System

Memory serves the agent — it does not *be* the agent.

Inspired by neuroscience (as metaphor, not claim):
- **Activation decay** (Ebbinghaus): unused results fade. This is the complexity penalty — the system forgets dead ends.
- **Recall boost** (spaced repetition): frequently retrieved results persist longer.
- **Association** (Hebbian): co-occurring successes link together.

Memory helps with: avoiding repeats, retrieving related results, forming summaries for LLM context, tracking calibration over time.

## Transfer Across Campaigns

Transfer is a first-class subsystem, not implicit in memory.

```python
# What transfers: factor importances, interaction effects, prior weights
# Guards against negative transfer:
#   - Schema must be compatible
#   - Tasks must be similar (metadata check)
#   - Source must have been well-calibrated (Brier score check)
# Blending: source posterior becomes weak prior for new campaign
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
| Memory matters | +memory outperforms -memory by ≥15% (ablation) |
| Detects diminishing returns | Agent stops when: low uncertainty in promising regions AND expected improvement below threshold AND best-so-far plateaued AND calibration healthy AND diversity exhausted |
| Stable across seeds | Crossover point variance <30% across 5 random seeds |

If these criteria are not met, the system does not work. No moving goalposts.

## Toy Validation Environment

A 27-cell synthetic POMDP (3 factors × 3 levels) with canonical active inference (expected free energy, policy-conditioned beliefs).

This exists **only to validate the theory** — it is not a proxy workload, not a component of the practical system, and not a surrogate predictor. It is a theoretical sandbox where we verify that canonical active inference produces the expected epistemic → pragmatic shift, proper calibration, and baseline-beating convergence. If these properties don't emerge in the toy environment, the practical system inherits no credibility from "active inference inspiration."

## Theoretical Roots

### Active Inference (Karl Friston)
The decomposition of expected free energy into pragmatic value (preference-seeking) and epistemic value (uncertainty-reduction). We use this as design inspiration for the practical controller. Canonical implementation only in the toy environment.

### Learntropy (Piotr Wozniak)
Not all surprise is equal. The brain rewards information at the *boundary* of current knowledge — structured novelty, not random noise. Thompson sampling operationalizes this: posterior uncertainty naturally directs attention to where learning would be most valuable.

### Compression Progress (Juergen Schmidhuber)
Intrinsic reward = the first derivative of model improvement. The agent should focus where its model is improving fastest, not where uncertainty is highest per se.

### Key Papers
- Friston et al. 2019 — [Generalised free energy and active inference](https://link.springer.com/article/10.1007/s00422-019-00805-w)
- Da Costa et al. 2020 — [Active inference on discrete state-spaces](https://arxiv.org/abs/2001.07203)
- Champion et al. 2024 — [Reframing the Expected Free Energy](https://arxiv.org/abs/2402.14460)
- Millidge et al. 2021 — [Whence the Expected Free Energy?](https://arxiv.org/abs/2004.08128)
- Li et al. 2026 — [Curiosity is Knowledge](https://arxiv.org/abs/2602.06029)
- Parr, Pezzulo & Friston 2022 — [Active Inference (MIT Press, open access)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)

## What This Is NOT

- Not a faithful implementation of canonical active inference (except in the toy environment)
- Not a replacement for domain expertise in designing the intervention schema
- Not going to discover fundamentally new architectures
- Not going to beat ASHA on the narrow task ASHA was designed for

It is: a structured acquisition function with memory, inspired by active inference, that learns factor-level structure, transfers knowledge, explains its reasoning, and knows when to stop.

## Implementation Phases

1. **Toy validation environment** — 27-cell synthetic POMDP with canonical active inference. Verify: calibration, crossover, baselines, stopping. Theoretical sandbox only.
2. **Surrogate predictor + controller** — Bayesian linear model, Thompson sampling, wired to proxy ML workload (small NanoGPT-style runs). Benchmark against BO, ASHA, random.
3. **Memory + LLM** — Experiment memory with decay. LLM proposal generation. Memory ablation study.
4. **Transfer** — Cross-campaign transfer with guards. Related task pairs. Negative transfer measurement.
5. **Visualization + evaluation** — Dashboard, calibration plots, regret curves, full report.
6. **Scale validation** — Test whether knowledge from proxy workloads transfers to more expensive target runs.

## License

MIT
