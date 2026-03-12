"""Compare autoresearcher2 approaches from artifact files.

Reads results from:
- artifacts/trainpy_loop/first_real_loop.json (pure Bayesian, Thompson sampling)
- artifacts/trainpy_llm_loop/results.json (LLM v1, Thompson + claude -p)
- artifacts/trainpy_llm_v2_loop/results.json (LLM v2, lookahead + enhanced prompt)

Produces a comparison table and saves summary to artifacts/comparison.json.
"""

import json
from pathlib import Path


def load_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_run(data: dict) -> dict:
    """Extract key metrics from a run's results."""
    results = data.get("results", [])
    successful = [r for r in results if r.get("error") is None and r.get("val_bpb") is not None]

    if not successful:
        return {"n_experiments": 0}

    val_bpbs = [r["val_bpb"] for r in successful]
    cells = [r["cell"] for r in successful]

    # Track best-so-far at each experiment
    best_so_far = []
    running_best = float("inf")
    for bpb in val_bpbs:
        running_best = min(running_best, bpb)
        best_so_far.append(running_best)

    # Experiment where best was found
    best_idx = val_bpbs.index(min(val_bpbs))
    best_config = successful[best_idx].get("config", {})

    # Source breakdown
    labels = [r.get("label", "unknown") for r in successful]
    by_label = {}
    for label in set(labels):
        label_bpbs = [r["val_bpb"] for r, l in zip(successful, labels) if l == label]
        by_label[label] = {
            "count": len(label_bpbs),
            "mean": sum(label_bpbs) / len(label_bpbs),
            "best": min(label_bpbs),
        }

    return {
        "n_experiments": len(successful),
        "n_unique_cells": len(set(cells)),
        "best_val_bpb": min(val_bpbs),
        "worst_val_bpb": max(val_bpbs),
        "mean_val_bpb": sum(val_bpbs) / len(val_bpbs),
        "best_at_experiment": best_idx + 1,
        "best_config": best_config,
        "best_so_far": best_so_far,
        "by_label": by_label,
        "total_time_s": data.get("total_time_s", "?"),
        "factor_importances": data.get("factor_importances", {}),
    }


def main():
    base = Path("artifacts")

    approaches = {
        "Pure Bayesian (Thompson)": base / "trainpy_loop" / "first_real_loop.json",
        "LLM v1 (Thompson + claude)": base / "trainpy_llm_loop" / "results_v1.json",
        "LLM v2 (Lookahead + enhanced)": base / "trainpy_llm_loop" / "results.json",
        "Autoresearch baseline": base / "autoresearch_baseline" / "results.json",
    }

    results = {}
    for name, path in approaches.items():
        data = load_results(path)
        if data:
            results[name] = analyze_run(data)
        else:
            print(f"  [{name}] No results found at {path}")

    if not results:
        print("No results to compare.")
        return

    # Print comparison
    print("=" * 70)
    print("AUTORESEARCHER2 — APPROACH COMPARISON")
    print("=" * 70)

    header = f"{'Metric':<30}"
    for name in results:
        short = name.split("(")[0].strip()
        header += f" | {short:>16}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Best val_bpb", "best_val_bpb", ".4f"),
        ("Mean val_bpb", "mean_val_bpb", ".4f"),
        ("Experiments", "n_experiments", "d"),
        ("Unique cells", "n_unique_cells", "d"),
        ("Best found at exp #", "best_at_experiment", "d"),
        ("Total time (s)", "total_time_s", ".0f"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<30}"
        for name, r in results.items():
            val = r.get(key, "?")
            if val != "?" and isinstance(val, (int, float)):
                row += f" | {val:>16{fmt}}"
            else:
                row += f" | {'?':>16}"
        print(row)

    # Best config per approach
    print()
    for name, r in results.items():
        config = r.get("best_config", {})
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        print(f"  {name}: best config = {config_str}")

    # Source breakdown
    print()
    print("SOURCE BREAKDOWN:")
    for name, r in results.items():
        by_label = r.get("by_label", {})
        if by_label:
            parts = []
            for label, stats in sorted(by_label.items()):
                parts.append(f"{label}: {stats['count']} runs, mean={stats['mean']:.4f}")
            print(f"  {name}:")
            for p in parts:
                print(f"    {p}")

    # Convergence comparison (best-so-far at experiment 5, 10, 15, 20)
    print()
    print("CONVERGENCE (best val_bpb at experiment N):")
    checkpoints = [5, 10, 15, 20]
    header = f"{'Experiment':<15}"
    for name in results:
        short = name.split("(")[0].strip()
        header += f" | {short:>16}"
    print(header)
    for n in checkpoints:
        row = f"{n:<15}"
        for name, r in results.items():
            bsf = r.get("best_so_far", [])
            if len(bsf) >= n:
                row += f" | {bsf[n-1]:>16.4f}"
            else:
                row += f" | {'—':>16}"
        print(row)

    # Save comparison
    output = base / "comparison.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
