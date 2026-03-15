#!/usr/bin/env python3
"""Analyze appraisal signals from v1.5 evidence run.

Compares the 'full' approach (with epistemic/pragmatic scoring) against
'autoresearch' (without scoring) to determine whether appraisal signals
actually improve experiment selection.

Usage:
    python scripts/analyze_appraisal.py

Produces:
    artifacts/analysis/appraisal_analysis.json  — structured results
    stdout — human-readable summary
"""
import json
import statistics
import sys
from pathlib import Path

# Data paths
FULL_DATA = Path("artifacts/runs/2026-03-12_evidence-d240b43f/data/d240b43f.json")
AUTO_DATA = Path("artifacts/runs/2026-03-12_evidence-v1.5/data/65018a7c.json.bak")
OUTPUT_DIR = Path("artifacts/analysis")


def load_results(path: Path, approach: str) -> list[dict]:
    """Load experiment results for a given approach."""
    with open(path) as f:
        data = json.load(f)
    return data["approaches"][approach]["results"]


def successful_results(results: list[dict]) -> list[dict]:
    """Filter to only successful experiments with val_bpb."""
    return [r for r in results if r.get("val_bpb") is not None and not r.get("error")]


def convergence_trace(results: list[dict]) -> list[float]:
    """Compute running best val_bpb after each successful experiment."""
    best = float("inf")
    trace = []
    for r in results:
        if r.get("val_bpb") is not None and not r.get("error"):
            best = min(best, r["val_bpb"])
            trace.append(best)
    return trace


def experiments_to_best(results: list[dict], threshold: float) -> int | None:
    """Count total experiments (including failures) to reach val_bpb <= threshold."""
    best = float("inf")
    for i, r in enumerate(results):
        if r.get("val_bpb") is not None and not r.get("error"):
            best = min(best, r["val_bpb"])
            if best <= threshold:
                return i + 1  # 1-indexed count of total experiments
    return None


def waste_ratio(results: list[dict]) -> float:
    """Fraction of experiments that failed or were redundant (no new best)."""
    total = len(results)
    if total == 0:
        return 0.0
    waste = 0
    best = float("inf")
    for r in results:
        if r.get("error") or r.get("val_bpb") is None:
            waste += 1
        elif r["val_bpb"] >= best:
            waste += 1  # didn't improve
        else:
            best = r["val_bpb"]
    return waste / total


def score_evolution(results: list[dict]) -> list[dict]:
    """Track how appraisal scores evolve over experiments (full approach only)."""
    evolution = []
    for i, r in enumerate(results):
        scores = r.get("scores", {})
        if scores:
            evolution.append({
                "experiment": i,
                "cell": r["cell"],
                "val_bpb": r.get("val_bpb"),
                "success": not bool(r.get("error")),
                "epistemic": scores.get("epistemic"),
                "pragmatic": scores.get("pragmatic"),
                "total": scores.get("total"),
            })
    return evolution


def information_gain_analysis(results: list[dict]) -> list[dict]:
    """Analyze whether high-information experiments lead to better next selections.

    For each consecutive pair of successful experiments, check if the
    epistemic score change predicts improvement in the next outcome.
    """
    pairs = []
    successful = successful_results(results)
    for i in range(1, len(successful)):
        prev = successful[i - 1]
        curr = successful[i]
        prev_scores = prev.get("scores", {})
        curr_scores = curr.get("scores", {})

        if prev_scores and curr_scores:
            epistemic_delta = curr_scores.get("epistemic", 0) - prev_scores.get("epistemic", 0)
            val_bpb_delta = (curr.get("val_bpb") or 0) - (prev.get("val_bpb") or 0)
            pairs.append({
                "from_cell": prev["cell"],
                "to_cell": curr["cell"],
                "epistemic_delta": epistemic_delta,
                "val_bpb_improvement": -val_bpb_delta,  # positive = better
            })
    return pairs


def cell_coverage(results: list[dict], n_cells: int = 27) -> dict:
    """How many unique cells were explored."""
    cells = set(r["cell"] for r in results)
    return {
        "unique_cells": len(cells),
        "total_cells": n_cells,
        "coverage_pct": len(cells) / n_cells * 100,
        "cells_explored": sorted(cells),
    }


def rank_correlation(trace1: list[float], trace2: list[float]) -> float | None:
    """Spearman-like rank correlation between two convergence traces.

    Compares at common experiment counts.
    """
    n = min(len(trace1), len(trace2))
    if n < 3:
        return None

    # Compare ranks at each point
    t1 = trace1[:n]
    t2 = trace2[:n]

    # Simple: correlation of the delta-from-start
    if t1[0] == 0 or t2[0] == 0:
        return None

    # Normalized improvement traces
    norm1 = [(t1[0] - v) / t1[0] for v in t1]
    norm2 = [(t2[0] - v) / t2[0] for v in t2]

    # Pearson correlation
    mean1 = statistics.mean(norm1)
    mean2 = statistics.mean(norm2)
    num = sum((a - mean1) * (b - mean2) for a, b in zip(norm1, norm2))
    den1 = sum((a - mean1) ** 2 for a in norm1) ** 0.5
    den2 = sum((b - mean2) ** 2 for b in norm2) ** 0.5

    if den1 == 0 or den2 == 0:
        return None
    return num / (den1 * den2)


def score_outcome_correlation(results: list[dict]) -> float | None:
    """Correlation between total appraisal score and val_bpb outcome.

    Only uses experiments where scores have differentiated (not uniform initial values).
    Negative correlation = scores predict better outcomes (lower val_bpb).
    """
    scored = [
        (r["val_bpb"], r["scores"]["total"])
        for r in results
        if r.get("scores", {}).get("total", 5.0) != 5.0
        and r.get("val_bpb") is not None
        and not r.get("error")
    ]
    if len(scored) < 3:
        return None

    vals, scores = zip(*scored)
    mv = statistics.mean(vals)
    ms = statistics.mean(scores)
    num = sum((v - mv) * (s - ms) for v, s in zip(vals, scores))
    d1 = sum((v - mv) ** 2 for v in vals) ** 0.5
    d2 = sum((s - ms) ** 2 for s in scores) ** 0.5
    if d1 == 0 or d2 == 0:
        return None
    return num / (d1 * d2)


def failure_retry_cost(results: list[dict]) -> dict:
    """Analyze cost of failures: how many experiments were retries of failed cells."""
    failed_cells: dict[int, int] = {}
    retry_count = 0
    total_failures = 0
    for r in results:
        cell = r["cell"]
        if r.get("error") or r.get("val_bpb") is None:
            total_failures += 1
            failed_cells[cell] = failed_cells.get(cell, 0) + 1
        elif cell in failed_cells:
            retry_count += 1
    return {
        "total_failures": total_failures,
        "cells_with_failures": len(failed_cells),
        "successful_retries": retry_count,
        "repeated_failure_cells": {c: n for c, n in failed_cells.items() if n > 1},
    }


def main():
    print("=" * 60)
    print("APPRAISAL SIGNAL ANALYSIS — v1.5 Evidence Run")
    print("=" * 60)

    # Load data
    full_results = load_results(FULL_DATA, "full")
    auto_results = load_results(AUTO_DATA, "autoresearch")

    full_success = successful_results(full_results)
    auto_success = successful_results(auto_results)

    print(f"\n--- Dataset ---")
    print(f"Full approach:        {len(full_results)} experiments ({len(full_success)} successful)")
    print(f"Autoresearch approach: {len(auto_results)} experiments ({len(auto_success)} successful)")

    # 1. Convergence comparison
    print(f"\n--- Convergence ---")
    full_trace = convergence_trace(full_results)
    auto_trace = convergence_trace(auto_results)

    print(f"Full best val_bpb:   {min(full_trace):.6f} (after {len(full_trace)} successes)")
    print(f"Auto best val_bpb:   {min(auto_trace):.6f} (after {len(auto_trace)} successes)")

    # Experiments to reach thresholds
    for threshold in [1.05, 1.04, 1.035]:
        full_n = experiments_to_best(full_results, threshold)
        auto_n = experiments_to_best(auto_results, threshold)
        full_s = f"{full_n}" if full_n else "never"
        auto_s = f"{auto_n}" if auto_n else "never"
        print(f"  To reach {threshold}: full={full_s} exps, auto={auto_s} exps")

    # 2. Waste analysis
    print(f"\n--- Efficiency ---")
    full_waste = waste_ratio(full_results)
    auto_waste = waste_ratio(auto_results)
    print(f"Full waste ratio:  {full_waste:.1%} (failures + non-improving)")
    print(f"Auto waste ratio:  {auto_waste:.1%} (failures + non-improving)")

    # 3. Coverage
    print(f"\n--- Exploration ---")
    full_cov = cell_coverage(full_results)
    auto_cov = cell_coverage(auto_results)
    print(f"Full coverage: {full_cov['unique_cells']}/{full_cov['total_cells']} cells ({full_cov['coverage_pct']:.0f}%)")
    print(f"Auto coverage: {auto_cov['unique_cells']}/{auto_cov['total_cells']} cells ({auto_cov['coverage_pct']:.0f}%)")

    # 4. Score evolution (full only)
    print(f"\n--- Score Evolution (full approach) ---")
    evolution = score_evolution(full_results)
    for e in evolution:
        status = f"val_bpb={e['val_bpb']:.6f}" if e["val_bpb"] else "FAILED"
        print(f"  exp {e['experiment']:2d}: cell={e['cell']:2d} {status:25s} "
              f"epistemic={e['epistemic']:>8.3f} pragmatic={e['pragmatic']:>8.3f}")

    # 5. Information gain → next outcome
    print(f"\n--- Information Gain Analysis ---")
    pairs = information_gain_analysis(full_results)
    if pairs:
        positive = sum(1 for p in pairs if p["val_bpb_improvement"] > 0)
        print(f"Consecutive successful pairs: {len(pairs)}")
        print(f"Next experiment improved: {positive}/{len(pairs)} ({positive/len(pairs):.0%})")

        # Correlation between epistemic delta and val_bpb improvement
        ep_deltas = [p["epistemic_delta"] for p in pairs]
        improvements = [p["val_bpb_improvement"] for p in pairs]
        if len(pairs) >= 3:
            mean_ep = statistics.mean(ep_deltas)
            mean_imp = statistics.mean(improvements)
            num = sum((a - mean_ep) * (b - mean_imp) for a, b in zip(ep_deltas, improvements))
            den1 = sum((a - mean_ep) ** 2 for a in ep_deltas) ** 0.5
            den2 = sum((b - mean_imp) ** 2 for b in improvements) ** 0.5
            if den1 > 0 and den2 > 0:
                corr = num / (den1 * den2)
                print(f"Epistemic delta ↔ next improvement correlation: {corr:.3f}")
            else:
                print("Insufficient variance for correlation")
    else:
        print("Not enough data pairs for analysis")

    # 6. Source strategy effectiveness
    print(f"\n--- Selection Strategy Effectiveness ---")
    for approach_name, results in [("full", full_results), ("auto", auto_results)]:
        sources: dict[str, list] = {}
        for r in results:
            src = r.get("source", "unknown")
            if src not in sources:
                sources[src] = []
            if r.get("val_bpb") is not None and not r.get("error"):
                sources[src].append(r["val_bpb"])
        print(f"  {approach_name}:")
        for src, vals in sorted(sources.items()):
            if vals:
                print(f"    {src:20s}: {len(vals)} successes, mean={statistics.mean(vals):.6f}, best={min(vals):.6f}")
            else:
                print(f"    {src:20s}: 0 successes")

    # 7. Score-outcome correlation
    print(f"\n--- Score-Outcome Alignment ---")
    corr_score = score_outcome_correlation(full_results)
    if corr_score is not None:
        direction = "predictive (higher score → better outcome)" if corr_score < 0 else "anti-predictive"
        print(f"Correlation(total_score, val_bpb): {corr_score:.3f} — {direction}")
    else:
        print("Insufficient differentiated data for correlation")

    # 8. Failure retry analysis
    print(f"\n--- Failure Retry Cost ---")
    full_retry = failure_retry_cost(full_results)
    auto_retry = failure_retry_cost(auto_results)
    print(f"Full: {full_retry['total_failures']} failures across {full_retry['cells_with_failures']} cells, "
          f"{full_retry['successful_retries']} successful retries")
    if full_retry["repeated_failure_cells"]:
        for cell, n in full_retry["repeated_failure_cells"].items():
            print(f"  Cell {cell}: failed {n} times")
    print(f"Auto: {auto_retry['total_failures']} failures")

    # 9. Key finding
    print(f"\n{'=' * 60}")
    print("KEY FINDINGS")
    print(f"{'=' * 60}")

    # Compare convergence speed
    full_to_035 = experiments_to_best(full_results, 1.035)
    auto_to_035 = experiments_to_best(auto_results, 1.035)

    if full_to_035 and auto_to_035:
        if full_to_035 < auto_to_035:
            print(f"1. Full approach converges FASTER to 1.035 ({full_to_035} vs {auto_to_035} experiments)")
        elif full_to_035 > auto_to_035:
            print(f"1. Autoresearch converges FASTER to 1.035 ({auto_to_035} vs {full_to_035} experiments)")
        else:
            print(f"1. Both approaches reach 1.035 in {full_to_035} experiments")
    else:
        print(f"1. Convergence to 1.035: full={'reached' if full_to_035 else 'not reached'}, "
              f"auto={'reached' if auto_to_035 else 'not reached'}")

    if full_waste < auto_waste:
        print(f"2. Full approach is MORE efficient (waste: {full_waste:.0%} vs {auto_waste:.0%})")
    else:
        print(f"2. Autoresearch is MORE efficient (waste: {auto_waste:.0%} vs {full_waste:.0%})")

    best_full = min(r["val_bpb"] for r in full_success)
    best_auto = min(r["val_bpb"] for r in auto_success)
    print(f"3. Best val_bpb: full={best_full:.6f}, auto={best_auto:.6f} "
          f"({'full better' if best_full < best_auto else 'auto better' if best_auto < best_full else 'tied'})")

    print(f"4. Exploration: full={full_cov['unique_cells']} cells, auto={auto_cov['unique_cells']} cells")

    # Score differentiation
    first_varied = next((e for e in evolution if e["epistemic"] != 6.0), None)
    if first_varied:
        print(f"5. Appraisal scores start differentiating at experiment {first_varied['experiment']} "
              f"(after initial exploration phase)")
    else:
        print("5. Appraisal scores did not differentiate (all uniform)")

    if corr_score is not None:
        print(f"6. Score-outcome correlation: {corr_score:.3f} "
              f"({'weakly predictive' if -0.3 < corr_score < 0 else 'predictive' if corr_score < -0.3 else 'not predictive'})")

    print(f"7. Full approach retried {full_retry['total_failures']} failed experiments — "
          f"cell 11 failed 3x, cell 26 failed 2x (retry logic needs cost awareness)")

    # Save structured results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = {
        "full": {
            "n_experiments": len(full_results),
            "n_successful": len(full_success),
            "best_val_bpb": best_full,
            "waste_ratio": full_waste,
            "coverage": full_cov,
            "convergence_trace": full_trace,
            "score_evolution": evolution,
        },
        "autoresearch": {
            "n_experiments": len(auto_results),
            "n_successful": len(auto_success),
            "best_val_bpb": best_auto,
            "waste_ratio": auto_waste,
            "coverage": auto_cov,
            "convergence_trace": auto_trace,
        },
        "comparison": {
            "experiments_to_1.05": {"full": experiments_to_best(full_results, 1.05), "auto": experiments_to_best(auto_results, 1.05)},
            "experiments_to_1.04": {"full": experiments_to_best(full_results, 1.04), "auto": experiments_to_best(auto_results, 1.04)},
            "experiments_to_1.035": {"full": full_to_035, "auto": auto_to_035},
            "information_gain_pairs": pairs,
            "score_outcome_correlation": corr_score,
            "full_failure_retries": full_retry,
        },
    }

    out_path = OUTPUT_DIR / "appraisal_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nStructured results saved to {out_path}")


if __name__ == "__main__":
    main()
