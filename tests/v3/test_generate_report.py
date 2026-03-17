"""Tests for domain-agnostic generate_report.py."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path so we can import the script
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import generate_report


# --- Default metric ---

class TestDefaultMetric:
    """The default target metric should be 'outcome', not 'val_bpb'."""

    def _make_project(self, domain_config=None):
        """Helper: minimal project dict as returned by fetch_data_*."""
        dc = domain_config or {}
        target = dc.get("target_metric", "outcome")
        return {
            "name": "Test Project",
            "id": "proj_test",
            "domain_config": dc,
            "target_metric": target,
            "optimize": dc.get("optimize", "minimize"),
            "total_obs": 0,
            "success_obs": 0,
            "failed_obs": 0,
            "wm_version": 1,
            "trajectory": [],
            "beliefs": [],
            "tensions": [],
            "cost_beliefs": {},
            "totals": {"energy_kwh": 0, "cost_eur": 0, "wall_s": 0},
        }

    def test_default_metric_is_outcome_not_val_bpb(self):
        """When domain_config has no target_metric, default should be 'outcome'."""
        # Check that the source code uses "outcome" as default, not "val_bpb"
        src = Path(__file__).parent.parent.parent / "scripts" / "generate_report.py"
        content = src.read_text()
        # Both fetch functions should default to "outcome"
        assert '"val_bpb"' not in content, (
            "generate_report.py still contains hardcoded 'val_bpb' default"
        )


# --- Heatmap: data-driven spec keys ---

class TestHeatmapGeneric:
    """Heatmap should use the top 2 most-varied spec keys from data, not hardcoded DEPTH/MATRIX_LR."""

    def test_heatmap_not_hardcoded_to_nanogpt(self):
        """Heatmap section should not reference DEPTH or MATRIX_LR literally."""
        src = Path(__file__).parent.parent.parent / "scripts" / "generate_report.py"
        content = src.read_text()
        # The heatmap generation code should not contain hardcoded spec keys
        # (The old code had: t["spec"].get("DEPTH") and t["spec"].get("MATRIX_LR"))
        # We check generate_figures function only
        import ast
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate_figures":
                func_src = ast.get_source_segment(content, node)
                assert '"DEPTH"' not in func_src, "generate_figures still has hardcoded DEPTH"
                assert '"MATRIX_LR"' not in func_src, "generate_figures still has hardcoded MATRIX_LR"
                break

    def test_heatmap_uses_top_two_varied_keys(self, tmp_path):
        """Heatmap should be generated from whatever spec keys vary most."""
        data = {"projects": [{
            "name": "Generic Domain",
            "id": "proj_gen",
            "domain_config": {},
            "target_metric": "score",
            "optimize": "maximize",
            "total_obs": 6,
            "success_obs": 6,
            "failed_obs": 0,
            "wm_version": 1,
            "trajectory": [
                {"run": 1, "value": 0.5, "best": 0.5, "type": "config_change",
                 "spec": {"param_a": 1, "param_b": 10, "param_c": "fixed"}},
                {"run": 2, "value": 0.6, "best": 0.6, "type": "config_change",
                 "spec": {"param_a": 1, "param_b": 20, "param_c": "fixed"}},
                {"run": 3, "value": 0.7, "best": 0.7, "type": "config_change",
                 "spec": {"param_a": 2, "param_b": 10, "param_c": "fixed"}},
                {"run": 4, "value": 0.8, "best": 0.8, "type": "config_change",
                 "spec": {"param_a": 2, "param_b": 20, "param_c": "fixed"}},
            ],
            "beliefs": [],
            "tensions": [],
            "cost_beliefs": {},
            "totals": {"energy_kwh": 0, "cost_eur": 0, "wall_s": 0},
        }]}
        figures = generate_report.generate_figures(data, tmp_path)
        # Should have a heatmap since param_a and param_b each have 2 unique values
        assert any("heatmap" in k for k in figures), (
            f"No heatmap generated for generic data. Keys: {list(figures.keys())}"
        )


# --- Glossary: data-driven ---

class TestGlossaryGeneric:
    """Glossary should be built from actual data, not hardcoded NanoGPT terms."""

    def test_no_hardcoded_glossary_terms(self):
        """The glossary section should not contain hardcoded NanoGPT-specific terms."""
        src = Path(__file__).parent.parent.parent / "scripts" / "generate_report.py"
        content = src.read_text()
        # Old hardcoded glossary had these literal strings
        assert "val\\_bpb" not in content or "val\\_bpb" in content.split("def generate_latex")[0], \
            "Hardcoded val_bpb glossary entry still present"
        assert "MATRIX\\_LR" not in content, "Hardcoded MATRIX_LR glossary entry still present"
        assert "WEIGHT\\_DECAY" not in content, "Hardcoded WEIGHT_DECAY glossary entry still present"


# --- Hardware mention ---

class TestHardwareMention:
    """Hardware-specific mentions should be generic."""

    def test_no_rtx_pro_6000_mention(self):
        src = Path(__file__).parent.parent.parent / "scripts" / "generate_report.py"
        content = src.read_text()
        assert "RTX PRO 6000" not in content, "Still contains hardware-specific GPU mention"
        assert "96" not in content or "96\\,GB" not in content, "Still contains VRAM-specific mention"
