# tests/eval/test_harness.py
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.eval.harness import run_evaluation


def test_evaluation_runs_and_compares():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    env_config = {
        "true_effects": {"optimizer": {"adam": 0.3, "sgd": -0.3}},
        "baseline": 0.5,
        "noise_std": 0.05,
    }

    report = run_evaluation(
        schema=schema,
        env_config=env_config,
        n_experiments=30,
        seed=42,
    )

    assert "autoresearcher2" in report
    assert "random" in report
    assert "greedy" in report

    for agent_name, metrics in report.items():
        assert "best_outcome" in metrics
        assert "cumulative_regret" in metrics
        assert "outcomes" in metrics


def test_evaluation_autoresearcher_beats_random():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    env_config = {
        "true_effects": {
            "optimizer": {"adam": 0.4, "sgd": -0.4},
            "lr": {"low": -0.1, "high": 0.1},
        },
        "baseline": 0.5,
        "noise_std": 0.02,
    }

    report = run_evaluation(
        schema=schema,
        env_config=env_config,
        n_experiments=80,
        seed=42,
    )

    ar2_best = report["autoresearcher2"]["best_outcome"]
    random_best = report["random"]["best_outcome"]
    assert ar2_best >= random_best - 0.05
