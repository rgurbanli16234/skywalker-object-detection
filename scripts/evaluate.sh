#!/bin/bash
set -e

echo "Starting evaluation..."
source venv/bin/activate
python src/evaluate.py --data data.yaml
echo "Evaluation complete."
