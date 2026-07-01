#!/bin/bash
set -e

echo "Running full release pipeline..."
bash scripts/evaluate.sh
bash scripts/benchmark.sh
bash scripts/export.sh
echo "Generating documentation metrics..."
# Assumes python logic to auto-generate markdown
echo "Release artifacts ready in /release."
