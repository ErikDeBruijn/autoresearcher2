# Guardrails

## Warning

This project is ambitious enough to become unstable or unmeasurable if multiple new ideas are introduced at once.

New capabilities are welcome, but only after the previous layer has evidence-quality validation. Pilot and debugging runs are useful for observability, bug-finding, and qualitative signal checks, but they do not count as definitive evidence. The project loses value as soon as it becomes harder to understand why something works.

Avoid:
- multiple independent changes in the same evaluation phase
- README/design claims that exceed current evidence
- conclusions drawn from pilot/debug runs
- capability expansion without a clear measurement strategy
- "impressive but unstable" architectural drift

## Guardrails

- Fix obvious bugs and missing instrumentation immediately.
- Treat instrumentation as part of reliability, not as scope creep.
- Only add a new capability after the previous layer has evidence-quality validation.
- Keep schema, objective, and agent architecture stable during an evaluation phase.
- Prefer changing only one main axis at a time:
  - controller logic
  - schema
  - workload
  - instrumentation
  - parallelism model
  - agent architecture
- Parallel execution comes before multi-agent social search.
- Throughput should first be observed and analyzed before it is optimized.
- Heterogeneous workers are allowed only after stable multi-worker execution under one controller.
- If a bug or instrumentation gap invalidates a run, relabel that run as pilot/debugging rather than using it as evidence.
- Every important claim should be backed by runnable behavior, logs, or artifacts.

## Long-term expansion path

1. Throughput-aware reasoning
   Expose runtime metadata (`tokens_M`, `tok/sec`, `mfu`, `num_steps`, `runtime`) to analysis and LLM reflection.
2. Throughput-affecting knobs
   Add knobs such as batch size or compiler-related settings only as a new explicit phase.
3. Multi-worker execution under one controller
   One controller, one shared world model, multiple workers, explicit batch diversity, no accidental duplicates.
4. Heterogeneous workers / compute-aware generalization
   Learn what depends on model/dataset vs compute environment by comparing runs across different worker types.
5. Multi-agent social research
   Explorer / exploiter / critic / synthesizer architectures only after the single-controller multi-worker path is stable and measured.

## Why This Matters

autoresearcher2 should become more capable without becoming harder to reason about.

The project wins if it can say, with evidence:
- what worked
- why it worked
- what generalizes
- what remains uncertain

It loses if it becomes an impressive but unstable pile of interacting ideas.

Stay ambitious. Advance in layers. Do not let exciting directions erase interpretability.
