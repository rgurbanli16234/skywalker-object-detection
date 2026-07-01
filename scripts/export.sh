#!/bin/bash
set -e

echo "Starting export pipeline..."
source venv/bin/activate
python scripts/run_export.py
echo "Export complete."
