"""Smoke test: run one 50-experiment synthetic loop and print diagnostics."""

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.core.loop import ResearchLoop


def main():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "adamw", "sgd"],
            "lr": ["1e-4", "3e-4", "1e-3"],
            "batch_size": ["32", "64", "128"],
        }
    )

    env = SyntheticEnvironment(
        schema=schema,
        true_effects={
            "optimizer": {"adam": 0.2, "adamw": 0.25, "sgd": -0.2},
            "lr": {"1e-4": -0.1, "3e-4": 0.15, "1e-3": 0.0},
            "batch_size": {"32": -0.05, "64": 0.05, "128": 0.0},
        },
        baseline=0.5,
        noise_std=0.05,
        seed=42,
    )

    model = BayesianLinearModel(schema, noise_variance=0.05**2)
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    memory = MemoryStore()

    loop = ResearchLoop(schema, model, controller, memory, env)
    results = loop.run(n_experiments=50)

    # Best outcome
    best = max(results, key=lambda r: r["outcome"])
    print(f"Best outcome: {best['outcome']:.4f} at config {best['config']}")

    # Factor importances (heuristic — not causal claims)
    importances = model.factor_importances()
    print("\nFactor importances (heuristic, not causal):")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # Top 5 appraisal events by learntropy
    top = memory.top_by_appraisal("learntropy", n=5)
    print("\nTop 5 experiments by learntropy:")
    for r in top:
        print(
            f"  cell={r['cell_index']} outcome={r['outcome']:.3f} "
            f"surprise={r['appraisal']['surprise']:.3f} "
            f"learntropy={r['appraisal']['learntropy']:.3f}"
        )

    # Summary
    summary = memory.summary()
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
