# Run Context

## Hypothesis
Structured Bayesian inference combined with LLM reasoning outperforms either alone
for hyperparameter search on a real GPT training pipeline.

## World model relationship
This run tests the core 2×2 ablation of autoresearcher2:
- Does a structured generative model (factor effects, uncertainty) help beyond random search?
- Does LLM world knowledge add value beyond what Thompson sampling captures?
- Does the combination (Bayesian + LLM with appraisal signals) beat either alone?

The theoretical claim is that LLMs bring prior knowledge about the search space
(e.g., "deeper models train fewer tokens per unit time") while Bayesian models bring
systematic uncertainty tracking. The combination should allow the LLM to focus on
regions where the Bayesian model is most uncertain or most wrong.

## Approach design
The four approaches form a 2×2 factorial:

|                | No LLM        | LLM                |
|----------------|---------------|--------------------|
| No model       | random        | autoresearch       |
| Bayesian model | bayesian      | full               |

- **random**: uniform cell selection (control)
- **bayesian**: Thompson sampling + two-step lookahead, linear factor model
- **autoresearch**: LLM sees flat results table, suggests configs (Karpathy-style)
- **full**: Bayesian model + LLM with appraisal signals (surprise, learntropy, factor importances, coverage gaps)

## Known confounds
- **Token budget varies by config**: DEPTH=6 trains ~323M tokens, DEPTH=8 ~176M, DEPTH=10 ~101M
  in the same wall time. Better val_bpb for shallower models may partly reflect more training.
- **Single seed**: no statistical significance claims possible.
- **Small schema**: 27 cells with one dominant factor (depth) may be too easy for LLM priors.
- **Sequential execution**: approaches run one after another, not interleaved. Hardware state may drift.

## Methodology notes
Random and bayesian ran in the original process (run_id 5d2ffa21). A silent-fallback
bug was discovered: LLM failures were labeled as "llm_flat" instead of "llm_fallback_*".
Process was killed after bayesian completed. Autoresearch and full were restarted
with fixed source-labeling code (run_id 65018a7c, commit ce7d19a).

## Success criteria
From GUARDRAILS.md phase transition rules:
1. All four approaches complete 20 experiments each
2. Results committed with full instrumentation
3. Evaluation document with honest conclusions about what the LLM adds
