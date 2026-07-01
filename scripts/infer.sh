#!/bin/bash
set -e

echo "Starting inference..."
source venv/bin/activate
python src/infer.py --source dataset/images/val
echo "Inference complete."
