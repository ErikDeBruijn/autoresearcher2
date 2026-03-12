# Guardrails — Current Phase

For normative principles, see CONSTITUTION.md and CHARTER.md.
This document contains only the operational constraints for the current evaluation phase.

## Current phase: Evidence-quality evaluation (v1.5)

**Objective:** Produce a clean head-to-head comparison of four approaches on real train.py.

**What is stable (do not change):**
- 3-factor schema: DEPTH × MATRIX_LR × WEIGHT_DECAY (27 cells)
- Outcome metric: val_bpb (lower is better)
- Hardware: GPU 1 on dllm-experiment.home
- Seed: 42

**What is being measured:**
- random baseline vs Bayesian-only vs flat LLM vs Bayesian+LLM
- Per experiment: val_bpb, tokens_M, tok/sec, mfu, num_steps, wall_time_s, decision_id, trial_id, source

**What is not allowed in this phase:**
- New schema knobs
- Throughput optimization
- Multi-worker scheduling
- Multi-agent architecture
- Schema or objective changes

**What is allowed:**
- Bug fixes and instrumentation improvements
- Observing and recording throughput (but not optimizing for it)

## Phase transition rule

This phase ends when:
1. All four approaches have completed 20 experiments each
2. Results are committed with full instrumentation
3. An evaluation document exists with honest conclusions about what the LLM adds, whether structured signals help, and which claims are justified

## Next phases

Each phase requires evidence-quality validation of the previous one.

### Intelligence track (primary — making the agent smarter)

**v2.0 — Unconstrained research agent**
- LLM proposes "research steps" instead of just configs (analysis, hypothesis, experiment)
- Code execution on dllm-experiment.home sandbox (Python scripts via SSH — already isolated)
- LLM can use statistical tools (regression, power analysis, factorial design)
- LLM can modify train.py: architecture changes (e.g. transformer → Mamba), optimizer,
  data pipeline — like Karpathy's autoresearch but with epistemic governance
- Must still solve the current problem (val_bpb optimization) as a baseline
- Evidence test: does richer LLM interaction improve sample efficiency vs v1.5?

**v2.1 — Domain transfer test**
- Same pipeline, different dataset (Wikipedia, code, or multilingual)
- Same schema to isolate dataset-specific vs generalizable findings
- Evidence test: do optimal configs transfer, or are they dataset-specific?
- This is the G-factor test — fluid intelligence vs crystallized

**v2.2 — Cross-domain generalization**
- Apply the agent to a non-ML verifiable problem (compiler flags, material properties)
- Minimal harness changes — same governance, same meta-loop, different environment
- Evidence test: does the research methodology generalize beyond ML?

### Infrastructure track (secondary — scaling what works)

**v3.0 — Throughput-aware reasoning**
- Expose tokens_M, tok/sec, wall_time to LLM analysis (partially done in v1.5)
- Throughput-affecting knobs as new factors

**v3.1 — Multi-worker execution**
- Arbitrarily many workers: unprivileged VMs/containers with GPU passthrough
- Workers run on Proxmox cluster (or any host with GPU passthrough)
- Central controller distributes experiments to available workers
- Workers are stateless and disposable — results flow back to controller

**v3.2 — Multi-agent social research**
- Multiple agents with different priors
- Coordinated exploration with shared evidence grammar

