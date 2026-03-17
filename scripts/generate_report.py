#!/usr/bin/env python3
"""Generate a research progress report from live production data.

Connects to the VM's SQLite database, extracts project data,
generates figures with matplotlib, writes LaTeX, compiles PDF.

Usage:
    # Full report (connects to VM via SSH)
    uv run python scripts/generate_report.py

    # From local DB copy
    uv run python scripts/generate_report.py --db research_v4.db

    # Skip LLM discussion
    uv run python scripts/generate_report.py --no-llm
"""

import argparse
import json
import platform
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

# Lazy matplotlib import (needs viz extra)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fetch_data_ssh(ssh_host="root@dllm-experiment.local", ssh_key="~/.ssh/pve03_key"):
    """Pull project data from the VM via SSH + Python one-liner."""
    script = r'''
import json
from autoresearcher2.v3.store import Store
s = Store("research_v4.db")
result = {"projects": []}
for p in s.list_projects(active_only=False):
    pid = p["id"]
    wm = s.load_world_model(project_id=pid)
    obs = [o for o in s.list_observations() if o.project_id == pid]
    success = sorted([o for o in obs if o.outcome_success], key=lambda o: o.created_at)

    trajectory = []
    target = (p.get("domain_config") or {}).get("target_metric", "val_bpb")
    maximize = (p.get("domain_config") or {}).get("optimize") == "maximize"
    best_so_far = -float("inf") if maximize else float("inf")
    for i, o in enumerate(success):
        val = (o.outcome_metrics or {}).get(target)
        if val is not None:
            if (maximize and val > best_so_far) or (not maximize and val < best_so_far):
                best_so_far = val
        trajectory.append({
            "run": i+1, "value": val, "best": best_so_far,
            "type": o.intervention_type,
            "spec": {k: v for k, v in o.intervention_spec.items() if k != "file_changes"},
        })

    beliefs = sorted(wm.beliefs, key=lambda b: -float(b["confidence"]))
    result["projects"].append({
        "name": p["name"], "id": pid,
        "domain_config": p.get("domain_config"),
        "target_metric": target,
        "optimize": "maximize" if maximize else "minimize",
        "total_obs": len(obs),
        "success_obs": len(success),
        "failed_obs": len(obs) - len(success),
        "wm_version": wm.version,
        "trajectory": trajectory,
        "beliefs": [{"id": b["id"], "claim": b["claim"], "confidence": float(b["confidence"])} for b in beliefs],
        "tensions": [t if isinstance(t, str) else json.dumps(t) for t in wm.tensions],
        "cost_beliefs": wm.cost_beliefs,
        "totals": {
            "energy_kwh": sum(o.energy_kwh or 0 for o in obs),
            "cost_eur": sum(o.cost_eur or 0 for o in obs),
            "wall_s": sum(o.wall_time_s or 0 for o in obs),
        },
    })
s.close()
print(json.dumps(result, default=str))
'''
    cmd = [
        "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10", ssh_host,
        f"cd /root/github.com/erikdebruijn/autoresearcher2 && uv run python -c '{script}'"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"SSH data fetch failed: {result.stderr[:300]}")
    return json.loads(result.stdout)


def fetch_data_local(db_path):
    """Pull data from a local DB copy."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from autoresearcher2.v3.store import Store
    s = Store(db_path)
    # Same logic as SSH version but local
    result = {"projects": []}
    for p in s.list_projects(active_only=False):
        pid = p["id"]
        wm = s.load_world_model(project_id=pid)
        obs = [o for o in s.list_observations() if o.project_id == pid]
        success = sorted([o for o in obs if o.outcome_success], key=lambda o: o.created_at)

        target = (p.get("domain_config") or {}).get("target_metric", "val_bpb")
        maximize = (p.get("domain_config") or {}).get("optimize") == "maximize"
        best_so_far = -float("inf") if maximize else float("inf")
        trajectory = []
        for i, o in enumerate(success):
            val = (o.outcome_metrics or {}).get(target)
            if val is not None:
                if (maximize and val > best_so_far) or (not maximize and val < best_so_far):
                    best_so_far = val
            trajectory.append({
                "run": i+1, "value": val, "best": best_so_far,
                "type": o.intervention_type,
                "spec": {k: v for k, v in o.intervention_spec.items() if k != "file_changes"},
            })

        beliefs = sorted(wm.beliefs, key=lambda b: -float(b["confidence"]))
        result["projects"].append({
            "name": p["name"], "id": pid,
            "domain_config": p.get("domain_config"),
            "target_metric": target,
            "optimize": "maximize" if maximize else "minimize",
            "total_obs": len(obs),
            "success_obs": len(success),
            "failed_obs": len(obs) - len(success),
            "wm_version": wm.version,
            "trajectory": trajectory,
            "beliefs": [{"id": b["id"], "claim": b["claim"], "confidence": float(b["confidence"])} for b in beliefs],
            "tensions": [t if isinstance(t, str) else json.dumps(t) for t in wm.tensions],
            "cost_beliefs": wm.cost_beliefs,
            "totals": {
                "energy_kwh": sum(o.energy_kwh or 0 for o in obs),
                "cost_eur": sum(o.cost_eur or 0 for o in obs),
                "wall_s": sum(o.wall_time_s or 0 for o in obs),
            },
        })
    s.close()
    return result


def generate_figures(data, output_dir):
    """Generate matplotlib figures for each project."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    for proj in data["projects"]:
        if not proj["trajectory"]:
            continue

        name = proj["name"]
        slug = name.lower().replace(" ", "_")
        target = proj["target_metric"]
        maximize = proj["optimize"] == "maximize"
        traj = proj["trajectory"]

        # --- Convergence plot ---
        fig, ax = plt.subplots(figsize=(8, 3.5))
        runs = [t["run"] for t in traj if t["value"] is not None]
        vals = [t["value"] for t in traj if t["value"] is not None]
        bests = [t["best"] for t in traj if t["value"] is not None]

        # Scatter: color by intervention type
        colors = {"probe": "#6b7280", "config_change": "#3b82f6", "code_change": "#a855f7"}
        for t in traj:
            if t["value"] is None:
                continue
            c = colors.get(t["type"], "#6b7280")
            ax.scatter(t["run"], t["value"], c=c, s=20, alpha=0.7, zorder=3)

        # Step function for records
        ax.step(runs, bests, where="post", color="#22c55e", linewidth=2, label="Best so far", zorder=2)
        ax.set_xlabel("Experiment #", fontsize=10)
        ax.set_ylabel(target, fontsize=10)
        ax.set_title(f"{name}: Convergence", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # Legend for types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b7280", markersize=6, label="probe"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b82f6", markersize=6, label="config_change"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#a855f7", markersize=6, label="code_change"),
            Line2D([0], [0], color="#22c55e", linewidth=2, label="best so far"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=8)

        fig.tight_layout()
        conv_path = output_dir / f"{slug}_convergence.pdf"
        fig.savefig(conv_path, dpi=150)
        plt.close(fig)
        figures[f"{slug}_convergence"] = conv_path

        # --- Heatmap: val_bpb by DEPTH × LR (NanoGPT specific) ---
        if target == "val_bpb":
            depths = sorted(set(t["spec"].get("DEPTH") for t in traj if t["spec"].get("DEPTH") is not None))
            lrs = sorted(set(t["spec"].get("MATRIX_LR") for t in traj if t["spec"].get("MATRIX_LR") is not None))

            if depths and lrs:
                # Build grid: best val_bpb per (depth, lr)
                grid = np.full((len(depths), len(lrs)), np.nan)
                for t in traj:
                    d, lr = t["spec"].get("DEPTH"), t["spec"].get("MATRIX_LR")
                    if d is not None and lr is not None and t["value"] is not None:
                        di, li = depths.index(d), lrs.index(lr)
                        if np.isnan(grid[di, li]) or t["value"] < grid[di, li]:
                            grid[di, li] = t["value"]

                fig, ax = plt.subplots(figsize=(6, 3))
                im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
                ax.set_xticks(range(len(lrs)))
                ax.set_xticklabels([f"{lr}" for lr in lrs], fontsize=8, rotation=45)
                ax.set_yticks(range(len(depths)))
                ax.set_yticklabels([f"D={d}" for d in depths], fontsize=9)
                ax.set_xlabel("MATRIX_LR", fontsize=10)
                ax.set_ylabel("DEPTH", fontsize=10)
                ax.set_title(f"{name}: Best val_bpb by DEPTH × LR", fontsize=11, fontweight="bold")

                for i in range(len(depths)):
                    for j in range(len(lrs)):
                        if not np.isnan(grid[i, j]):
                            ax.text(j, i, f"{grid[i,j]:.4f}", ha="center", va="center", fontsize=7,
                                    color="white" if grid[i, j] > np.nanmedian(grid) else "black")

                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                heat_path = output_dir / f"{slug}_heatmap.pdf"
                fig.savefig(heat_path, dpi=150)
                plt.close(fig)
                figures[f"{slug}_heatmap"] = heat_path

        # --- Belief confidence distribution ---
        if proj["beliefs"]:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            confs = [b["confidence"] for b in proj["beliefs"]]
            ax.hist(confs, bins=np.arange(0, 1.05, 0.05), color="#3b82f6", edgecolor="#1e3a5f", alpha=0.8)
            ax.set_xlabel("Confidence", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{name}: Belief Confidence Distribution ({len(confs)} beliefs)", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1)
            fig.tight_layout()
            bel_path = output_dir / f"{slug}_beliefs.pdf"
            fig.savefig(bel_path, dpi=150)
            plt.close(fig)
            figures[f"{slug}_beliefs"] = bel_path

    return figures


def escape_latex(s):
    for ch in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(ch, f"\\{ch}")
    # Replace common Unicode that LaTeX can't handle
    s = s.replace("→", r"$\to$")
    s = s.replace("←", r"$\leftarrow$")
    s = s.replace("≈", r"$\approx$")
    s = s.replace("×", r"$\times$")
    s = s.replace("≥", r"$\geq$")
    s = s.replace("≤", r"$\leq$")
    s = s.replace("---", "---")
    s = s.replace("–", "--")
    s = s.replace("'", "'")
    s = s.replace("'", "'")
    s = s.replace(""", "``")
    s = s.replace(""", "''")
    s = s.replace("~", r"\textasciitilde{}")
    return s


def _author_name():
    """Return author based on hostname."""
    host = platform.node().lower()
    if "erik" in host or "dllm" in host:
        return r"Erik de Bruijn \and autoresearcher2 (autonomous)"
    return "autoresearcher2 (autonomous)"


def _get_version():
    """Get version from latest git tag or branch name."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_latex(data, figures, output_dir):
    """Build the LaTeX document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    version = _get_version()

    # Filter to projects with actual results
    active_projects = [p for p in data["projects"] if p["trajectory"] and p["success_obs"] >= 3]

    # Project sections
    project_sections = ""
    for proj in active_projects:
        name = proj["name"]
        slug = name.lower().replace(" ", "_")
        target = proj["target_metric"]
        direction = proj["optimize"]
        n_success = proj["success_obs"]
        n_fail = proj["failed_obs"]
        n_beliefs = len(proj["beliefs"])
        n_tensions = len(proj["tensions"])
        totals = proj["totals"]

        # Best result
        traj = proj["trajectory"]
        vals = [t["value"] for t in traj if t["value"] is not None]
        if vals:
            if direction == "maximize":
                best_val = max(vals)
            else:
                best_val = min(vals)
            best_run = next(t for t in traj if t["value"] == best_val)
            best_spec = ", ".join(f"{k}={v}" for k, v in best_run["spec"].items() if k not in ("file_changes",))
            best_result_section = f"""\\subsubsection*{{Best result}}
\\texttt{{{escape_latex(target)}}} = \\textbf{{{best_val:.6f}}} at run {best_run["run"]} ({escape_latex(best_run["type"])}) \\\\
Config: \\texttt{{{escape_latex(best_spec)}}}"""
        else:
            best_result_section = "\\subsubsection*{Best result}\nNo successful observations with target metric yet."

        # Convergence figure
        conv_fig = figures.get(f"{slug}_convergence")
        heat_fig = figures.get(f"{slug}_heatmap")
        bel_fig = figures.get(f"{slug}_beliefs")

        domain = escape_latex(name)
        project_sections += f"""
\\subsection{{{domain}}}

\\textbf{{Target:}} {direction} \\texttt{{{escape_latex(target)}}} \\\\
\\textbf{{Runs:}} {n_success} successful, {n_fail} failed \\\\
\\textbf{{World model:}} v{proj["wm_version"]} --- {n_beliefs} beliefs, {n_tensions} tensions \\\\
\\textbf{{Cost:}} {totals["energy_kwh"]:.3f} kWh, \\euro{{}}{totals["cost_eur"]:.2f}, {totals["wall_s"]/3600:.1f} hours wall time

{best_result_section}
"""
        if conv_fig:
            conv_rel = Path(conv_fig).relative_to(output_dir)
            project_sections += f"""
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\columnwidth]{{{conv_rel}}}
\\caption{{{escape_latex(name)}: convergence over {n_success} experiments. Green line = cumulative best.}}
\\end{{figure}}
"""
        if heat_fig:
            heat_rel = Path(heat_fig).relative_to(output_dir)
            project_sections += f"""
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.85\\columnwidth]{{{heat_rel}}}
\\caption{{Best {escape_latex(target)} per DEPTH $\\times$ MATRIX\\_LR cell. Lower (green) is better.}}
\\end{{figure}}
"""

        # Key beliefs (top 5, with observation/inference labels)
        if proj["beliefs"]:
            project_sections += """
\\subsubsection*{Key beliefs}
\\begin{enumerate}
\\small
"""
            for b in proj["beliefs"][:5]:
                conf = b["confidence"]
                claim = escape_latex(b["claim"][:180])
                project_sections += f"  \\item \\textbf{{[{conf:.2f}]}} {claim}\n"
            if len(proj["beliefs"]) > 5:
                project_sections += f"  \\item[] \\textit{{... and {len(proj['beliefs']) - 5} more beliefs}}\n"
            project_sections += "\\end{enumerate}\n"

        # Tensions (top 3)
        if proj["tensions"]:
            project_sections += """
\\subsubsection*{Open questions}
\\begin{itemize}
\\small
"""
            for t in proj["tensions"][:3]:
                if "nature" in str(t):
                    import ast
                    try:
                        td = ast.literal_eval(t) if isinstance(t, str) else t
                        nature = td.get("nature", str(t))[:180]
                    except Exception:
                        nature = str(t)[:180]
                else:
                    nature = str(t)[:180]
                project_sections += f"  \\item {escape_latex(nature)}\n"
            project_sections += "\\end{itemize}\n"

        # Cost beliefs
        if proj["cost_beliefs"]:
            project_sections += """
\\subsubsection*{Measured cost per experiment}
\\begin{table}[h]
\\centering
\\small
\\begin{tabular}{lrrr}
\\toprule
Type & Wall time (s) & Energy (Wh) & Cost (\\euro{{}}) \\\\
\\midrule
"""
            for itype, costs in proj["cost_beliefs"].items():
                wt = costs.get("wall_time_s") or 0
                en = (costs.get("energy_kwh") or 0) * 1000
                co = costs.get("cost_eur") or 0
                project_sections += f"  {escape_latex(itype)} & {wt:.0f} & {en:.1f} & {co:.4f} \\\\\n"
            project_sections += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    # Summary stats (only active projects)
    n_domains = len(active_projects)
    total_obs = sum(p["total_obs"] for p in active_projects)
    total_beliefs = sum(len(p["beliefs"]) for p in active_projects)
    total_cost = sum(p["totals"]["cost_eur"] for p in active_projects)
    total_energy = sum(p["totals"]["energy_kwh"] for p in active_projects)

    # Mention early-stage domains if any were filtered
    early_domains = [p for p in data["projects"] if p["trajectory"] and p["success_obs"] < 3]
    early_note = ""
    if early_domains:
        names = ", ".join(escape_latex(p["name"]) for p in early_domains)
        early_note = f"Additionally, {names} {'is' if len(early_domains) == 1 else 'are'} in early exploration (fewer than 3 completed runs) and not yet reported."

    latex = textwrap.dedent(r"""
    \documentclass[10pt,a4paper,twocolumn]{article}
    \usepackage[margin=1.8cm]{geometry}
    \usepackage{booktabs}
    \usepackage{graphicx}
    \usepackage{xcolor}
    \usepackage{hyperref}
    \usepackage{eurosym}
    \usepackage[small]{titlesec}
    \usepackage[T1]{fontenc}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.4em}

    \title{\textbf{AutoResearcher2: Research Progress Memo}\\[0.3em]
           \large """ + escape_latex(version) + r""" --- Generator-Critic Pipeline}
    \author{""" + _author_name() + r"""}
    \date{""" + now + r"""}

    \begin{document}
    \maketitle
    \thispagestyle{empty}

    \section*{Executive Summary}

    An autonomous research agent, starting with zero domain knowledge,
    designed and executed \textbf{""" + str(total_obs) + r"""} GPU experiments
    across """ + str(n_domains) + r""" project""" + ("s" if n_domains != 1 else "") + r""",
    costing \textbf{\euro{}""" + f"{total_cost:.2f}" + r"""} in electricity
    (""" + f"{total_energy:.3f}" + r""" kWh). """ + early_note + r"""

    Through a generator-critic pipeline with an evolving world model,
    the system developed \textbf{""" + str(total_beliefs) + r""" explicit beliefs}
    about how training parameters interact --- each with a confidence score,
    supporting evidence, and natural language explanation.

    All results are exploratory (single-seed, no repetitions).
    The value is not in any individual finding, but in the agent's ability
    to build and communicate a structured understanding autonomously.

    \smallskip
    \noindent\textit{Source:} \url{https://github.com/erikdebruijn/autoresearcher2}

    \subsection*{Glossary}
    \small

    \textbf{val\_bpb} --- validation bits-per-byte; lower = better language model.

    \textbf{mean\_reward} --- average game score; higher = better agent.

    \textbf{DEPTH} --- number of transformer layers (model capacity).

    \textbf{MATRIX\_LR} --- learning rate for the main weight matrices.

    \textbf{WEIGHT\_DECAY} --- regularization strength (prevents overfitting).

    \textbf{Confidence} --- heuristic 0--1 score assigned by the LLM based on evidence count and consistency. Not a calibrated probability; treat as a ranking signal.

    \textbf{Belief} --- an LLM-generated claim about the domain, updated after each experiment. May mix observation and inference.

    \textbf{Tension} --- an open question the agent has identified but not yet resolved.
    \normalsize

    \section{Projects}

    """ + project_sections + r"""

    \section{Method}

    \subsection*{Pipeline}
    The system follows an OODA loop:
    \textbf{Orient} (update world model from latest result) $\to$
    \textbf{Generate} (LLM proposes 5 experiments, reasoning from epistemic intent) $\to$
    \textbf{Critique} (LLM ranks by information value, selects top 3) $\to$
    \textbf{Execute} (run on GPU, parse metrics, feed back).

    Each proposal follows: epistemic intent $\to$ rationale $\to$
    expected learning $\to$ intervention $\to$ cost estimate.
    This ensures experiments are designed to \emph{learn something regardless of outcome}.

    \subsection*{Experiment protocol}
    Each run trains a single model configuration to completion (single seed, no repetition).
    The agent controls hyperparameters (config\_change) or full training scripts (code\_change).
    A run is ``successful'' if it completes without error and produces a metric;
    ``failed'' runs (OOM, script errors) still inform the world model.
    Energy consumption is measured per-GPU via nvidia-smi polling at 1\,Hz.

    \subsection*{Confidence scores}
    Confidence values (0--1) are assigned by the LLM during world-model updates.
    They reflect evidence count and internal consistency, not frequentist probabilities.
    They are useful for ranking beliefs but should not be interpreted as calibrated.

    \section{Limitations}

    \begin{itemize}
        \item \textbf{Single seed}: no statistical significance; any result could be noise
        \item \textbf{No ablation}: the LLM's contribution vs.\ simpler optimization is untested
        \item \textbf{Hardware-specific}: all runs on one GPU (RTX PRO 6000, 96\,GB VRAM)
        \item \textbf{Beliefs mix observation and inference}: claims like ``X explains Y''
              are the agent's interpretation, not proven causal links
        \item \textbf{Exploratory}: this memo reports what the agent found, not what is established
    \end{itemize}

    \end{document}
    """)
    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate research progress PDF")
    parser.add_argument("--db", type=Path, default=None, help="Local DB path (skips SSH)")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    args = parser.parse_args()

    # Fetch data
    print("Fetching data...")
    if args.db:
        data = fetch_data_local(str(args.db))
    else:
        data = fetch_data_ssh()

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d")
    report_dir = args.output_dir / f"{timestamp}_progress"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures"

    # Generate figures
    print("Generating figures...")
    figures = generate_figures(data, fig_dir)
    print(f"  {len(figures)} figures generated")

    # Generate LaTeX
    print("Generating LaTeX...")
    latex = generate_latex(data, figures, report_dir)
    tex_path = report_dir / "report.tex"
    tex_path.write_text(latex)
    print(f"  Written to {tex_path}")

    # Save data snapshot
    data_path = report_dir / "data.json"
    data_path.write_text(json.dumps(data, indent=2, default=str))

    # Compile PDF
    print("Compiling PDF...")
    try:
        result = subprocess.run(
            ["tectonic", "report.tex"],
            capture_output=True, text=True, timeout=60,
            cwd=str(report_dir),
        )
        if result.returncode == 0:
            pdf_path = tex_path.with_suffix(".pdf")
            print(f"PDF generated: {pdf_path}")
            return str(pdf_path)
        else:
            print(f"tectonic failed:\n{result.stderr[:500]}")
    except FileNotFoundError:
        print("tectonic not found --- install with: brew install tectonic")
        print(f"LaTeX file ready at: {tex_path}")

    return str(tex_path)


if __name__ == "__main__":
    main()
