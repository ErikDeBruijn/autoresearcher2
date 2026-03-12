"""Generate a 1-2 page run report as LaTeX → PDF.

Two-stage pipeline:
1. Data stage: reads JSON results, produces structured tables and figures (deterministic)
2. Discussion stage: an LLM reads the data sections and writes interpretation,
   bound by CHARTER claim discipline (what is evidence vs. speculation)

The LLM doesn't just summarize — it's asked to think as a scientist:
what patterns matter, what confounds exist, what to investigate next.

Usage:
    uv run python scripts/generate_run_paper.py [--run-dir artifacts/runs/YYYY-MM-DD_name]
                                                 [--no-llm]
"""

import argparse
import json
import logging
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_approaches(run_dir: Path) -> dict:
    """Load all approach results from JSON files in data/."""
    approaches = {}
    for f in sorted((run_dir / "data").glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        for name, adata in data.get("approaches", {}).items():
            approaches[name] = adata
    return approaches


def approach_stats(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("val_bpb") is not None]
    if not ok:
        return {}
    bpbs = [r["val_bpb"] for r in ok]
    tokens = [r["tokens_M"] for r in ok if r.get("tokens_M")]
    tps = [r["tok_per_sec"] for r in ok if r.get("tok_per_sec")]
    sources = {}
    for r in results:
        s = r.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1

    # Best-found trajectory
    best_at = []
    current_best = float("inf")
    for r in ok:
        current_best = min(current_best, r["val_bpb"])
        best_at.append(current_best)
    # At which experiment did it first reach within 0.001 of final best?
    final_best = best_at[-1]
    convergence_exp = next((i + 1 for i, b in enumerate(best_at)
                           if b <= final_best + 0.001), len(best_at))

    return {
        "n": len(results),
        "n_ok": len(ok),
        "best": min(bpbs),
        "mean": np.mean(bpbs),
        "std": np.std(bpbs),
        "median": np.median(bpbs),
        "unique_cells": len(set(r["cell"] for r in ok)),
        "mean_tokens_M": np.mean(tokens) if tokens else None,
        "mean_tps": np.mean(tps) if tps else None,
        "convergence_exp": convergence_exp,
        "sources": sources,
    }


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(char, f"\\{char}")
    return s


DISCUSSION_INSTRUCTIONS = (
    "You are a research scientist writing the Discussion section of an experiment report.\n"
    "\n"
    "CLAIM DISCIPLINE (from project CHARTER):\n"
    "- Distinguish observation from interpretation\n"
    "- Label confidence: observed / supported / plausible / speculative\n"
    "- Name confounds explicitly\n"
    "- Say what you don't know\n"
    "\n"
    "Your task: write a Discussion section (LaTeX formatted) that interprets\n"
    "the results below. Not a summary — an analysis. What do these results\n"
    "mean for the hypothesis? What mechanisms explain the patterns? What\n"
    "confounds exist? What should we investigate next?\n"
    "\n"
    "Be concise (15-30 lines of LaTeX). No filler. Real uncertainty is fine;\n"
    "performative hedging is not.\n"
    "\n"
    "Output ONLY LaTeX body text — no section header, no preamble.\n"
    "Use \\textbf, \\emph, \\begin{itemize} as needed.\n"
)


def _build_discussion_prompt(data_sections: str, run_dir: Path) -> str:
    """Build the LLM prompt from run context + data.

    Reads context.md from the run directory for hypothesis, world model relationship,
    known confounds, and methodology notes. This keeps run-specific information
    out of the script.
    """
    context = ""
    context_path = run_dir / "context.md"
    if context_path.exists():
        context = "RUN CONTEXT (from context.md):\n" + context_path.read_text() + "\n\n"
    else:
        context = (
            "RUN CONTEXT: No context.md found in run directory.\n"
            "The discussion should focus on what the data shows.\n\n"
        )

    narrative = ""
    narrative_path = run_dir / "narrative.log"
    if narrative_path.exists():
        text = narrative_path.read_text()
        narrative = "NARRATIVE LOG (observations during the run):\n" + text[-2000:] + "\n\n"

    return DISCUSSION_INSTRUCTIONS + "\n" + context + "DATA:\n\n" + data_sections + "\n\n" + narrative


def generate_discussion_llm(data_sections: str,
                            run_dir: Path,
                            ssh_host: str = "root@dllm-experiment.home",
                            ssh_key: str = "~/.ssh/pve03_key") -> str | None:
    """Ask an LLM to write the Discussion section based on the data."""
    prompt = _build_discussion_prompt(data_sections, run_dir)

    ssh_cmd = [
        "ssh", "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        ssh_host,
        "claude -p --output-format json 2>/dev/null",
    ]

    try:
        result = subprocess.run(
            ssh_cmd, input=prompt,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning("LLM discussion failed (rc=%d): %s",
                           result.returncode, result.stderr[:200])
            return None

        raw = result.stdout
        try:
            outer = json.loads(raw)
            text = outer.get("result", raw)
        except (json.JSONDecodeError, AttributeError):
            text = raw

        # Strip any markdown code fences the LLM might add
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return text.strip()

    except Exception:
        logger.warning("LLM discussion call failed", exc_info=True)
        return None


def generate_latex(approaches: dict, run_dir: Path, use_llm: bool = True) -> str:
    """Generate the LaTeX document."""
    stats = {name: approach_stats(data["results"]) for name, data in approaches.items()}

    # Determine run metadata
    run_name = run_dir.name
    has_figure = (run_dir / "figures" / "evidence_dashboard.png").exists()
    figure_path = str(run_dir / "figures" / "evidence_dashboard.png") if has_figure else None

    # Status
    complete_approaches = [n for n, s in stats.items() if s.get("n_ok", 0) == 20]
    in_progress = [n for n, s in stats.items() if 0 < s.get("n_ok", 0) < 20]
    all_complete = len(complete_approaches) == 4

    # Detect fallbacks
    fallback_count = 0
    for s in stats.values():
        for src, count in s.get("sources", {}).items():
            if "fallback" in src:
                fallback_count += count

    # Build results table
    table_rows = []
    approach_order = ["random", "bayesian", "autoresearch", "full"]
    for name in approach_order:
        if name not in stats or not stats[name]:
            table_rows.append(f"    {name} & -- & -- & -- & -- & -- \\\\")
            continue
        s = stats[name]
        tok = f"{s['mean_tokens_M']:.0f}" if s.get("mean_tokens_M") else "--"
        table_rows.append(
            f"    {name} & {s['n_ok']}/20 & {s['best']:.6f} & {s['mean']:.6f} "
            f"& {s['unique_cells']} & {tok} \\\\"
        )

    # Source integrity table
    source_rows = []
    for name in approach_order:
        if name not in stats or not stats[name]:
            continue
        s = stats[name]
        sources_str = ", ".join(f"{k}: {v}" for k, v in sorted(s["sources"].items()))
        source_rows.append(f"    {name} & {escape_latex(sources_str)} \\\\")

    # Convergence comparison
    convergence_rows = []
    for name in approach_order:
        if name not in stats or not stats[name]:
            continue
        s = stats[name]
        convergence_rows.append(f"    {name} & {s['convergence_exp']} & {s['best']:.6f} \\\\")

    status_str = "COMPLETE" if all_complete else f"IN PROGRESS ({', '.join(in_progress)} running)"

    latex = textwrap.dedent(r"""
    \documentclass[10pt,a4paper,twocolumn]{article}
    \usepackage[margin=1.8cm]{geometry}
    \usepackage{booktabs}
    \usepackage{graphicx}
    \usepackage{xcolor}
    \usepackage{hyperref}
    \usepackage[small]{titlesec}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.4em}

    \title{\textbf{Evidence Run Report}\\[0.3em]
           \large """ + escape_latex(run_name) + r"""}
    \author{autoresearcher2}
    \date{""" + datetime.now().strftime("%Y-%m-%d %H:%M") + r"""}

    \begin{document}
    \maketitle
    \thispagestyle{empty}

    \section*{Status: """ + status_str + r"""}

    Head-to-head comparison of four experiment-selection approaches on real \texttt{train.py}.
    Schema: DEPTH $\times$ MATRIX\_LR $\times$ WEIGHT\_DECAY (27 cells).
    Budget: 20 experiments per approach. Hardware: GPU~1 (RTX PRO 6000), seed~42.

    \section{Results}

    \begin{table}[h]
    \centering
    \small
    \begin{tabular}{lrrrrr}
    \toprule
    Approach & Done & Best & Mean & Cells & Tok/exp \\
    \midrule
    """ + "\n".join(table_rows) + r"""
    \bottomrule
    \end{tabular}
    \caption{Performance summary. Best/mean = val\_bpb (lower is better). Cells = unique cells explored. Tok/exp = mean tokens trained per experiment (millions).}
    \end{table}

    \section{Convergence}

    Experiment number at which each approach first reached within 0.001 of its final best:

    \begin{table}[h]
    \centering
    \small
    \begin{tabular}{lrr}
    \toprule
    Approach & Conv.\ exp & Final best \\
    \midrule
    """ + "\n".join(convergence_rows) + r"""
    \bottomrule
    \end{tabular}
    \end{table}

    \section{Decision Source Integrity}

    Every experiment carries a source label indicating what method actually chose it.
    """ + (r"\textbf{No fallbacks detected.}" if fallback_count == 0
           else r"\textcolor{red}{\textbf{WARNING: " + str(fallback_count) + r" fallback(s) detected.}}") + r"""

    \begin{table}[h]
    \centering
    \small
    \begin{tabular}{ll}
    \toprule
    Approach & Sources \\
    \midrule
    """ + "\n".join(source_rows) + r"""
    \bottomrule
    \end{tabular}
    \end{table}

    \section{Observations (not claims)}

    These are patterns visible in the data. They are \textbf{not} generalization claims ---
    this is one run with one seed on one schema.

    \begin{itemize}
    \item DEPTH=8 consistently produces the best val\_bpb across all approaches.
          DEPTH=10 is worst (fewer tokens trained in fixed wall time).
    \item WEIGHT\_DECAY=0.4 appears in most best-performing configurations.
    \item Reproducibility is strong: repeated cell visits show $\Delta$val\_bpb $< 0.001$.
    \item Throughput varies $\sim$3$\times$ across depths (300K--1M tok/s), meaning
          deeper models train fewer tokens in the same wall time.
    \end{itemize}

    \section{What this does NOT prove}

    \begin{itemize}
    \item Generalization beyond this schema, seed, or hardware.
    \item That any approach is \emph{better} in a statistically significant sense
          (single seed, small budget).
    \item That the LLM's advantage (if confirmed) comes from world knowledge vs.\ lucky guessing.
    \item That structured signals (appraisal, factor importances) help the LLM.
    \end{itemize}

    \section{Methodology notes}

    """ + _methodology_from_context(run_dir) + r"""

    """)

    # Generate LLM discussion
    if use_llm:
        # Build a plain-text summary of the data sections for the LLM
        data_summary = _build_data_summary(stats, approach_order)
        narrative_path = run_dir / "narrative.log"
        if narrative_path.exists():
            data_summary += "\n\nNARRATIVE LOG (observations during the run):\n"
            data_summary += narrative_path.read_text()[-2000:]  # last 2000 chars

        print("Asking LLM for Discussion section...")
        discussion = generate_discussion_llm(data_summary, run_dir)
        if discussion:
            latex += r"""
    \section{Discussion}
    """ + discussion + "\n"
            print("  Discussion generated.")
        else:
            latex += r"""
    \section{Discussion}
    \textit{LLM discussion generation failed. Data sections above stand on their own.}
    """
            print("  Discussion generation failed, continuing without.")
    else:
        latex += r"""
    \section{Discussion}
    \textit{LLM discussion skipped (--no-llm flag).}
    """

    if has_figure:
        latex += r"""
    \begin{figure*}[b]
    \centering
    \includegraphics[width=\textwidth]{""" + figure_path + r"""}
    \caption{Evidence dashboard. Top-left: convergence curves. Top-right: factor effects.
    Bottom: exploration heatmaps per approach (cell visit counts).}
    \end{figure*}
    """

    latex += r"""
    \end{document}
    """
    return latex


def _methodology_from_context(run_dir: Path) -> str:
    """Extract methodology notes from context.md, or return a generic message."""
    context_path = run_dir / "context.md"
    if not context_path.exists():
        return "No context.md found in run directory."

    text = context_path.read_text()
    # Extract the "Methodology notes" section
    in_section = False
    lines = []
    for line in text.split("\n"):
        if line.strip().startswith("## Methodology"):
            in_section = True
            continue
        elif line.strip().startswith("## ") and in_section:
            break
        elif in_section:
            lines.append(line)

    if not lines:
        return "See context.md for methodology details."

    content = "\n".join(lines).strip()
    return escape_latex(content)


def _build_data_summary(stats: dict, approach_order: list[str]) -> str:
    """Build a plain-text summary of the data for the LLM discussion prompt."""
    lines = ["RESULTS SUMMARY:"]
    lines.append(f"{'Approach':<15} {'Done':>6} {'Best':>10} {'Mean':>10} {'Cells':>6} {'Conv':>5}")
    for name in approach_order:
        if name not in stats or not stats[name]:
            lines.append(f"{name:<15} {'--':>6} {'--':>10} {'--':>10} {'--':>6} {'--':>5}")
            continue
        s = stats[name]
        lines.append(
            f"{name:<15} {s['n_ok']:>4}/20 {s['best']:>10.6f} {s['mean']:>10.6f} "
            f"{s['unique_cells']:>6} {s['convergence_exp']:>5}"
        )

    lines.append("\nSOURCE INTEGRITY:")
    for name in approach_order:
        if name not in stats or not stats[name]:
            continue
        s = stats[name]
        sources = ", ".join(f"{k}: {v}" for k, v in sorted(s["sources"].items()))
        lines.append(f"  {name}: {sources}")

    lines.append("\nEXPERIMENTAL SETUP:")
    lines.append("- Schema: DEPTH={6,8,10} × MATRIX_LR={0.02,0.04,0.08} × WEIGHT_DECAY={0.1,0.2,0.4}")
    lines.append("- 20 experiments per approach, ~5 min each, same GPU (RTX PRO 6000), same seed (42)")
    lines.append("- val_bpb = validation bits per byte (lower is better)")
    lines.append("- Throughput varies ~3x: DEPTH=6 ~1M tok/s, DEPTH=8 ~540K, DEPTH=10 ~300K")
    lines.append("- This means DEPTH=6 trains ~3x more tokens in the same wall time")
    lines.append("- Single seed, single schema — no statistical significance claims possible")

    lines.append("\nIMPORTANT CONFOUND:")
    lines.append("- Wall-time budget is fixed, but token budget varies by config.")
    lines.append("  DEPTH=6 trains ~323M tokens, DEPTH=8 ~176M, DEPTH=10 ~101M.")
    lines.append("  A 'better val_bpb' for DEPTH=8 vs DEPTH=10 may partly reflect")
    lines.append("  more training tokens, not just better architecture.")

    lines.append("\nAPPROACH DETAILS:")
    lines.append("- random: uniform random cell selection, no learning")
    lines.append("- bayesian: Thompson sampling + two-step lookahead via Bayesian linear model")
    lines.append("  (conjugate Gaussian updates over one-hot factor features)")
    lines.append("- autoresearch: LLM (Claude) sees flat results table, suggests 3 configs per call")
    lines.append("  (mirrors Karpathy's autoresearch approach)")
    lines.append("- full: Bayesian model + LLM with appraisal signals (surprise, learntropy,")
    lines.append("  factor importances, coverage gaps) — the autoresearcher2 approach")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=Path("artifacts/runs/2026-03-12_evidence-v1.5"))
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM discussion generation")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = Path("/Users/erik/github.com/erikdebruijn/autoresearcher2") / run_dir

    approaches = load_approaches(run_dir)
    if not approaches:
        print(f"No data in {run_dir}/data/")
        return

    latex = generate_latex(approaches, run_dir, use_llm=not args.no_llm)

    tex_path = run_dir / "report.tex"
    tex_path.write_text(latex)
    print(f"LaTeX written to: {tex_path}")

    # Build PDF with tectonic
    try:
        result = subprocess.run(
            ["tectonic", str(tex_path)],
            capture_output=True, text=True, timeout=60,
            cwd=str(run_dir),
        )
        if result.returncode == 0:
            pdf_path = tex_path.with_suffix(".pdf")
            print(f"PDF generated: {pdf_path}")
        else:
            print(f"tectonic failed:\n{result.stderr[:500]}")
    except FileNotFoundError:
        print("tectonic not found — LaTeX file written, build manually")


if __name__ == "__main__":
    main()
