# v2.0 Design: Unconstrained Research Agent

## Summary

Expand the LLM from suggesting 3 hyperparameter configs to proposing **research steps**:
experiments, analyses, hypotheses, or schema changes. The sandbox is dllm-experiment.home
(already isolated, has GPU, no network access).

## Research Step Types

### 1. Experiment (existing, formalized)
```json
{"type": "experiment", "payload": {"config": {...}, "config_justification": "..."}}
```
Same as today but with explicit confidence and preconditions.

### 2. Analysis (new — code execution)
```json
{"type": "analysis", "payload": {
  "question": "Does val_bpb improve per-token or per-wall-clock-time?",
  "code": "import numpy as np; ...",
  "timeout_seconds": 60
}}
```
LLM writes Python code, executed in sandbox. Read-only access to results.
Can use numpy, scipy, pandas, statsmodels. Cannot use torch, subprocess, os.system.

### 3. Hypothesis (new — testable claims)
```json
{"type": "hypothesis", "payload": {
  "claim": "DEPTH × MATRIX_LR interaction explains 10% of residual variance",
  "test_type": "statistical",
  "proposed_test": "Fit y ~ depth + lr + depth*lr; compute r² gain",
  "acceptance_threshold": "p < 0.05"
}}
```

### 4. Schema Change (new — requires human approval)
```json
{"type": "schema_change", "payload": {
  "changes": [{"factor": "MATRIX_LR", "operation": "refine_levels",
               "new_levels": ["0.015", "0.02", "0.035", "0.05", "0.07", "0.10"]}],
  "approval_required": true
}}
```

## What stays hard-coded (the harness)

1. **Experiment execution + data storage** — deterministic, append-only, reproducible
2. **Epistemic governance** — source labeling, decision audit, CONSTITUTION/CHARTER
3. **Meta-loop** — when the LLM is consulted, resource budgets, termination
4. **Safety gates** — code sandbox, schema change approval, resource limits

## What's free for the LLM (the agent space)

- Choose research step type (experiment, analysis, hypothesis, schema change)
- Write and execute analysis code
- Formulate and test hypotheses
- Propose schema changes (with human approval gate)
- Reason about what it doesn't know

## Code Execution Sandbox

The dllm-experiment VM is already isolated:
- No network access
- GPU available for training experiments
- Filesystem isolated (git checkout resets)

Additional sandbox constraints for analysis code:
- Memory: 2 GB max
- CPU time: 60 seconds (configurable)
- Filesystem: read-only except artifacts/analyses/
- Module allowlist: numpy, scipy, pandas, statsmodels, sklearn
- Blocked: torch, tensorflow, subprocess, os.system, network

## Minimal Architecture Change

```
Current:  LLM → [3 configs]     → hard-coded loop → results → LLM
Proposed: LLM → [3 research steps] → router        → results → LLM
                                       ├─ experiment  → TrainPyEnv
                                       ├─ analysis    → CodeSandbox
                                       ├─ hypothesis  → StatisticalTest
                                       └─ schema_change → ApprovalQueue
```

The main loop barely changes. The router dispatches based on step type.
Experiments use existing code. Analyses run in subprocess. Hypotheses
are evaluated against existing data. Schema changes queue for approval.

## Comparison with Karpathy's autoresearch

| Aspect | Karpathy | autoresearcher2 v2.0 |
|--------|----------|---------------------|
| LLM freedom | Rewrites train.py freely | Proposes structured research steps |
| Governance | None | CONSTITUTION + CHARTER + per-step audit |
| Model changes | Implicit (edit code) | Explicit (schema_change with approval) |
| Reproducibility | Git diff only | Structured log with decision IDs |
| Statistical tools | Manual | LLM orchestrates scipy/statsmodels |
| Safety | Trust the LLM | Sandbox + allowlist + approval gates |

## Evidence Test

v2.0 must still solve the current problem (val_bpb optimization on 27-cell schema).
Success metric: **sample efficiency** — does richer LLM interaction find the optimum
in fewer experiments than v1.5?

## Sources

- Karpathy's autoresearch: https://github.com/karpathy/autoresearch
- OpenAI Agents SDK: code execution as first-class primitive
- E2B / Google Agent Sandbox: microVM isolation patterns
- Self-Organized Agents (SoA): role-based multi-agent specialization
