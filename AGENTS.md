# AGENTS.md

Follow CONSTITUTION.md as the highest normative layer and CHARTER.md as the governing epistemic layer for research, judgment, and self-improvement.

When uncertain whether a claim is justified, apply CHARTER.md §Claim discipline. When uncertain whether an action is responsible, apply CONSTITUTION.md §Autonomy as deserved trust.

## Core Rule

**The README must never claim more than the code proves.** Every claim about real-world behavior requires a runnable script that produces output artifacts. No artifact = soften the claim.

## Anti-Slop Checklist

Before any PR, answer these honestly:

1. **What runs outside the test suite?** Name a specific command a human can execute and observe new behavior.
2. **Which README claim does this PR prove?** Point to the artifact that proves it.
3. **What is explicitly NOT implemented?** List it. Don't let omissions hide behind abstractions.
4. **Does the README need softening?** If you added infrastructure but not execution, the README must say "planned" not "does."

## Forbidden Patterns

- **Architecture-only PRs** — interfaces, abstractions, future hooks without a runnable execution path. Every PR must add behavior OR measurable evidence.
- **README-ahead-of-code** — claiming train.py/val_bpb/real substrate works when only synthetic infra exists.
- **Premature abstraction** — plugin systems, orchestration layers, async frameworks before the basic happy path works end-to-end.
- **Tolerance-widening** — never relax test thresholds to make things pass. Fix the code instead.
- **Hardcoded test outputs** — never hardcode expected output to make a test green. A failing honest test is better than a passing fake one.

## Development Style

- **Seed everything.** All stochastic components use `np.random.default_rng(seed)`. No bare `np.random` calls.
- **Keep it simple.** Three similar lines > one premature abstraction. Add complexity only when the current task demands it.
- **Tests prove behavior, not structure.** Tests should verify what the system does, not that classes exist.
- **Smallest possible integration.** First real-substrate PR should be: fixed knobs, simple runner, simple parsing, real end-to-end execution. No LLM, no transfer, no orchestration.

## The Simple Test

> "If I ignore all tests, what concrete new behavior can I run and observe?"

If the answer is vague, the PR is probably slop.
