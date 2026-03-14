"""Visualize evidence run results: convergence curves, factor effects, model state."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path(__file__).parent.parent / "artifacts" / "evidence"


def load_all_results():
    """Load and merge results from all JSON files."""
    approaches = {}
    for f in sorted(ARTIFACTS.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        for name, adata in data.get("approaches", {}).items():
            approaches[name] = adata
    return approaches


def plot_convergence(approaches, ax):
    """Best-found val_bpb over experiment index per approach."""
    colors = {"random": "#888888", "bayesian": "#2196F3", "autoresearch": "#FF9800", "full": "#4CAF50"}
    for name in ["random", "bayesian", "autoresearch", "full"]:
        if name not in approaches:
            continue
        results = approaches[name]["results"]
        ok = [r for r in results if r.get("val_bpb") is not None]
        if not ok:
            continue
        best_so_far = []
        current_best = float("inf")
        for r in ok:
            current_best = min(current_best, r["val_bpb"])
            best_so_far.append(current_best)
        ax.plot(range(1, len(best_so_far) + 1), best_so_far,
                label=name, color=colors.get(name, "#000"), linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Best val_bpb found")
    ax.set_title("Convergence: best val_bpb over experiments")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_factor_effects(approaches, ax):
    """Mean val_bpb per factor level across all experiments (all approaches combined)."""
    all_results = []
    for adata in approaches.values():
        all_results.extend([r for r in adata["results"] if r.get("val_bpb") is not None])

    factors = {}
    for r in all_results:
        config = r.get("config", {})
        for fname, fval in config.items():
            if fname not in factors:
                factors[fname] = {}
            if fval not in factors[fname]:
                factors[fname][fval] = []
            factors[fname][fval].append(r["val_bpb"])

    n_factors = len(factors)
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, (fname, levels) in enumerate(sorted(factors.items())):
        level_names = sorted(levels.keys(), key=lambda x: float(x))
        means = [np.mean(levels[l]) for l in level_names]
        stds = [np.std(levels[l]) for l in level_names]
        x = np.arange(len(level_names)) + i * 0.25
        ax.bar(x, means, 0.2, yerr=stds, label=fname, color=colors[i % len(colors)], alpha=0.8)
        for j, (m, s, ln) in enumerate(zip(means, stds, level_names)):
            ax.text(x[j], m + s + 0.001, ln, ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Mean val_bpb (lower is better)")
    ax.set_title("Factor effects (all approaches pooled)")
    ax.set_xticks([])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


def plot_exploration_heatmap(approaches, axes):
    """Which cells were explored by each approach."""
    approach_order = ["random", "bayesian", "autoresearch", "full"]
    for idx, name in enumerate(approach_order):
        ax = axes[idx]
        if name not in approaches:
            ax.set_title(f"{name} (no data)")
            continue
        results = approaches[name]["results"]
        cell_counts = {}
        for r in results:
            c = r.get("cell", -1)
            cell_counts[c] = cell_counts.get(c, 0) + 1

        grid = np.zeros(27)
        for c, count in cell_counts.items():
            if 0 <= c < 27:
                grid[c] = count

        # Reshape to 3x9 (DEPTH × (MATRIX_LR × WEIGHT_DECAY))
        grid_2d = grid.reshape(3, 9)
        im = ax.imshow(grid_2d, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(3, grid.max()))
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("DEPTH")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["6", "8", "10"])
        ax.set_xlabel("LR×WD")
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"{lr},{wd}" for lr in ["0.02", "0.04", "0.08"]
                            for wd in ["0.1", "0.2", "0.4"]], rotation=45, fontsize=6)
        # Annotate counts
        for i in range(3):
            for j in range(9):
                if grid_2d[i, j] > 0:
                    ax.text(j, i, f"{int(grid_2d[i, j])}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=axes, shrink=0.6, label="# experiments")


def plot_bayesian_surprise(approaches, ax):
    """Surprise trajectory for bayesian approach (model convergence)."""
    if "bayesian" not in approaches:
        ax.set_title("Bayesian surprise (no data)")
        return
    results = approaches["bayesian"]["results"]
    surprises = []
    for r in results:
        apr = r.get("appraisal", {})
        s = apr.get("surprise", None)
        if s is not None:
            surprises.append(s)
    if surprises:
        ax.semilogy(range(1, len(surprises) + 1), surprises, "o-", color="#2196F3", markersize=4)
        ax.set_xlabel("Experiment #")
        ax.set_ylabel("Surprise (log scale)")
        ax.set_title("Bayesian model convergence (surprise)")
        ax.grid(True, alpha=0.3)


def main():
    approaches = load_all_results()
    if not approaches:
        print("No data found in", ARTIFACTS)
        sys.exit(1)

    print(f"Loaded approaches: {list(approaches.keys())}")
    for name, adata in approaches.items():
        n = len(adata["results"])
        ok = [r for r in adata["results"] if r.get("val_bpb") is not None]
        best = min(r["val_bpb"] for r in ok) if ok else "?"
        print(f"  {name}: {n} experiments, best={best}")

    fig = plt.figure(figsize=(16, 12))

    # Layout: 2x2 top, 4x1 bottom
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    ax_conv = fig.add_subplot(gs[0, :2])
    plot_convergence(approaches, ax_conv)

    ax_factor = fig.add_subplot(gs[0, 2:])
    plot_factor_effects(approaches, ax_factor)

    ax_surprise = fig.add_subplot(gs[1, :2])
    plot_bayesian_surprise(approaches, ax_surprise)

    # Source distribution pie for autoresearch/full
    ax_sources = fig.add_subplot(gs[1, 2:])
    source_data = {}
    for name in ["autoresearch", "full"]:
        if name in approaches:
            for r in approaches[name]["results"]:
                s = r.get("source", "unknown")
                source_data[s] = source_data.get(s, 0) + 1
    if source_data:
        colors_src = {"llm_flat": "#FF9800", "llm_augmented": "#4CAF50", "lookahead": "#2196F3",
                       "random_init": "#888888", "llm_fallback_error": "#F44336",
                       "llm_fallback_empty": "#E91E63", "llm_fallback": "#FF5722"}
        labels = list(source_data.keys())
        sizes = list(source_data.values())
        cols = [colors_src.get(l, "#999") for l in labels]
        ax_sources.bar(labels, sizes, color=cols)
        ax_sources.set_title("Decision sources (autoresearch + full)")
        ax_sources.set_ylabel("# experiments")
        plt.setp(ax_sources.xaxis.get_majorticklabels(), rotation=30, ha="right")
    else:
        ax_sources.set_title("Decision sources (no LLM data yet)")

    # Exploration heatmaps
    heatmap_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]
    plot_exploration_heatmap(approaches, heatmap_axes)

    outpath = ARTIFACTS.parent / "evidence_dashboard.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nDashboard saved to: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
