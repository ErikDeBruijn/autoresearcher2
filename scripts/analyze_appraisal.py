"""Analyze whether appraisal signals improve experiment selection.

Uses v1.5 evidence data to determine if learntropy, surprise, and
theory_conflict signals actually correlate with better convergence.

Method:
1. From evidence data: extract appraisal traces from 'full' approach
   (which has appraisal) vs 'autoresearch' (which doesn't)
2. After high-surprise experiments, does the next selection improve
   more than after low-surprise experiments?
3. Compute correlation between cumulative learntropy and convergence speed
4. Compare: does the full approach converge faster *because* of
   appraisal signals, or despite them?

Usage:
    uv run python scripts/analyze_appraisal.py
    uv run python scripts/analyze_appraisal.py --evidence-file artifacts/evidence/d240b43f.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_evidence(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_approach_data(evidence: dict, approach: str) -> list[dict]:
    """Get results for an approach, filtering failed experiments."""
    approaches = evidence.get("approaches", {})
    if approach not in approaches:
        return []
    return [r for r in approaches[approach].get("results", [])
            if r.get("error") is None]


def compute_convergence_curve(results: list[dict]) -> list[float]:
    """Best outcome so far at each step."""
    best_so_far = []
    current_best = float("-inf")
    for r in results:
        outcome = r.get("outcome", 0)
        if outcome is not None and outcome > current_best:
            current_best = outcome
        best_so_far.append(current_best)
    return best_so_far


def analyze_surprise_improvement(results: list[dict]) -> dict:
    """After high-surprise experiments, does the next selection improve more?"""
    if len(results) < 3:
        return {"error": "too few experiments"}

    improvements_after_high_surprise = []
    improvements_after_low_surprise = []

    # Median split on surprise
    surprises = [r.get("appraisal", {}).get("surprise", 0) for r in results]
    median_surprise = np.median(surprises)

    for i in range(len(results) - 1):
        current = results[i]
        next_r = results[i + 1]
        surprise = current.get("appraisal", {}).get("surprise", 0)

        # Improvement = how much better the next outcome is
        curr_outcome = current.get("outcome", 0)
        next_outcome = next_r.get("outcome", 0)
        if curr_outcome is None or next_outcome is None:
            continue
        improvement = next_outcome - curr_outcome

        if surprise >= median_surprise:
            improvements_after_high_surprise.append(improvement)
        else:
            improvements_after_low_surprise.append(improvement)

    return {
        "median_surprise": float(median_surprise),
        "n_high": len(improvements_after_high_surprise),
        "n_low": len(improvements_after_low_surprise),
        "mean_improvement_after_high_surprise": float(np.mean(improvements_after_high_surprise))
            if improvements_after_high_surprise else None,
        "mean_improvement_after_low_surprise": float(np.mean(improvements_after_low_surprise))
            if improvements_after_low_surprise else None,
    }


def analyze_learntropy_convergence(results: list[dict]) -> dict:
    """Correlation between cumulative learntropy and convergence speed."""
    if len(results) < 3:
        return {"error": "too few experiments"}

    cumulative_learntropy = []
    best_so_far = []
    running_learntropy = 0.0
    current_best = float("-inf")

    for r in results:
        learntropy = r.get("appraisal", {}).get("learntropy", 0)
        outcome = r.get("outcome", 0)
        if outcome is None:
            continue
        running_learntropy += learntropy
        cumulative_learntropy.append(running_learntropy)
        if outcome > current_best:
            current_best = outcome
        best_so_far.append(current_best)

    if len(cumulative_learntropy) < 3:
        return {"error": "too few valid experiments"}

    # Pearson correlation
    x = np.array(cumulative_learntropy)
    y = np.array(best_so_far)
    if np.std(x) == 0 or np.std(y) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])

    return {
        "correlation": correlation,
        "n_points": len(cumulative_learntropy),
        "total_learntropy": float(running_learntropy),
        "final_best": float(current_best),
    }


def compare_convergence(evidence: dict) -> dict:
    """Compare convergence speed between approaches."""
    results = {}
    for approach in ["random", "bayesian", "autoresearch", "full"]:
        data = extract_approach_data(evidence, approach)
        if not data:
            continue
        curve = compute_convergence_curve(data)
        results[approach] = {
            "n_experiments": len(data),
            "best_outcome": max(curve) if curve else None,
            "convergence_curve": curve,
        }

        # How quickly does it find a "good" outcome (>= 0.85)?
        good_threshold = 0.85
        first_good = next((i for i, v in enumerate(curve) if v >= good_threshold), None)
        results[approach]["first_good_at"] = first_good

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze appraisal signal contribution")
    parser.add_argument("--evidence-dir", type=str,
                        default="artifacts/evidence",
                        help="Directory with evidence JSON files")
    parser.add_argument("--evidence-file", type=str,
                        help="Specific evidence file to analyze")
    args = parser.parse_args()

    # Load evidence
    if args.evidence_file:
        evidence_files = [Path(args.evidence_file)]
    else:
        evidence_dir = Path(args.evidence_dir)
        evidence_files = sorted(evidence_dir.glob("*.json"))

    if not evidence_files:
        print("No evidence files found")
        sys.exit(1)

    print("=" * 70)
    print("APPRAISAL SIGNAL ANALYSIS")
    print("=" * 70)

    for evidence_file in evidence_files:
        print(f"\n{'─' * 70}")
        print(f"Evidence: {evidence_file.name}")
        print(f"{'─' * 70}")

        evidence = load_evidence(evidence_file)
        run_id = evidence.get("run_id", "unknown")

        # 1. Convergence comparison
        print(f"\n1. Convergence Comparison (run {run_id})")
        convergence = compare_convergence(evidence)
        print(f"   {'Approach':<15} {'Best':>8} {'First good':>12} {'N':>5}")
        print(f"   {'-'*45}")
        for approach, data in convergence.items():
            best = f"{data['best_outcome']:.4f}" if data['best_outcome'] else "?"
            first = str(data['first_good_at']) if data['first_good_at'] is not None else "never"
            print(f"   {approach:<15} {best:>8} {first:>12} {data['n_experiments']:>5}")

        # 2. Surprise → improvement analysis (full approach only)
        full_data = extract_approach_data(evidence, "full")
        if full_data and any(r.get("appraisal") for r in full_data):
            print(f"\n2. Surprise → Next Improvement (full approach)")
            surprise_analysis = analyze_surprise_improvement(full_data)
            if "error" not in surprise_analysis:
                print(f"   Median surprise threshold: {surprise_analysis['median_surprise']:.4f}")
                print(f"   After high surprise (n={surprise_analysis['n_high']}): "
                      f"mean improvement = {surprise_analysis['mean_improvement_after_high_surprise']:.4f}"
                      if surprise_analysis['mean_improvement_after_high_surprise'] is not None
                      else "   After high surprise: no data")
                print(f"   After low surprise  (n={surprise_analysis['n_low']}): "
                      f"mean improvement = {surprise_analysis['mean_improvement_after_low_surprise']:.4f}"
                      if surprise_analysis['mean_improvement_after_low_surprise'] is not None
                      else "   After low surprise: no data")

                high = surprise_analysis['mean_improvement_after_high_surprise']
                low = surprise_analysis['mean_improvement_after_low_surprise']
                if high is not None and low is not None:
                    if high > low:
                        print(f"   → High surprise leads to BETTER next selections (+{high-low:.4f})")
                    else:
                        print(f"   → High surprise does NOT lead to better next selections ({high-low:.4f})")
            else:
                print(f"   {surprise_analysis['error']}")

            # 3. Learntropy → convergence correlation
            print(f"\n3. Learntropy ↔ Convergence Correlation (full approach)")
            learntropy_analysis = analyze_learntropy_convergence(full_data)
            if "error" not in learntropy_analysis:
                r = learntropy_analysis["correlation"]
                print(f"   Pearson r = {r:.4f} (n={learntropy_analysis['n_points']})")
                print(f"   Total learntropy accumulated: {learntropy_analysis['total_learntropy']:.4f}")
                if abs(r) > 0.5:
                    print(f"   → {'Strong' if abs(r) > 0.7 else 'Moderate'} correlation")
                else:
                    print(f"   → Weak correlation — learntropy may not drive convergence")
            else:
                print(f"   {learntropy_analysis['error']}")

        else:
            print("\n2-3. No appraisal data in full approach — skipping")

        # 4. Full vs Autoresearch comparison
        full_conv = convergence.get("full", {})
        auto_conv = convergence.get("autoresearch", {})
        if full_conv and auto_conv:
            print(f"\n4. Full vs Autoresearch (appraisal vs no appraisal)")
            full_best = full_conv.get("best_outcome")
            auto_best = auto_conv.get("best_outcome")
            if full_best and auto_best:
                diff = full_best - auto_best
                print(f"   Full best:         {full_best:.4f}")
                print(f"   Autoresearch best: {auto_best:.4f}")
                print(f"   Difference:        {diff:+.4f}")
                if diff > 0.01:
                    print(f"   → Full approach (with appraisal) finds BETTER configs")
                elif diff < -0.01:
                    print(f"   → Autoresearch (no appraisal) finds better configs")
                else:
                    print(f"   → No meaningful difference")

            full_first = full_conv.get("first_good_at")
            auto_first = auto_conv.get("first_good_at")
            if full_first is not None and auto_first is not None:
                print(f"   Full first good:   experiment {full_first}")
                print(f"   Auto first good:   experiment {auto_first}")
                if full_first < auto_first:
                    print(f"   → Full converges FASTER ({auto_first - full_first} experiments earlier)")
                elif auto_first < full_first:
                    print(f"   → Autoresearch converges faster ({full_first - auto_first} experiments earlier)")

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print("=" * 70)
    print("Check above for:")
    print("  - Does high surprise predict better next selections?")
    print("  - Does cumulative learntropy correlate with convergence?")
    print("  - Does the full approach (with appraisal) outperform autoresearch?")
    print("If yes to all → appraisal signals are load-bearing")
    print("If no to all → appraisal signals are decorative (remove them)")


if __name__ == "__main__":
    main()
