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

Expansion path (each phase requires evidence-quality validation of the previous one):

0. Multi-runner controller (local + remote workers)
1. Throughput-aware reasoning (expose metadata to analysis and LLM)
2. Throughput-affecting knobs (new explicit phase)
3. Multi-worker execution under one controller
4. Heterogeneous workers / compute-aware generalization
5. Multi-agent social research

