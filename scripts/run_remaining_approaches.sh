#!/bin/bash
# Run autoresearch + full approaches with fixed source-labeling code.
# Random + bayesian data is already in artifacts/evidence/5d2ffa21.json
set -e

cd /Users/erik/github.com/erikdebruijn/autoresearcher2

RUN_DIR="artifacts/runs/2026-03-12_evidence-v1.5"

echo "$(date): Starting autoresearch approach..."
PYTHONUNBUFFERED=1 uv run python scripts/run_evidence.py --approach autoresearch 2>&1 | tee -a artifacts/evidence_run_remaining.log

echo "$(date): Autoresearch complete. Syncing data and generating report..."
cp artifacts/evidence/*.json "$RUN_DIR/data/" 2>/dev/null || true
uv run python scripts/generate_run_paper.py --run-dir "$RUN_DIR" 2>&1 | tee -a artifacts/evidence_run_remaining.log
echo "$(date): Report generated after autoresearch."

echo "$(date): Starting full approach..."
PYTHONUNBUFFERED=1 uv run python scripts/run_evidence.py --approach full 2>&1 | tee -a artifacts/evidence_run_remaining.log

echo "$(date): Full complete. Syncing data and generating final report..."
cp artifacts/evidence/*.json "$RUN_DIR/data/" 2>/dev/null || true
uv run python scripts/generate_run_paper.py --run-dir "$RUN_DIR" 2>&1 | tee -a artifacts/evidence_run_remaining.log

echo "$(date): Both approaches complete. Final report at $RUN_DIR/report.pdf"
