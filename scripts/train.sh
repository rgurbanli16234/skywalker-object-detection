#!/bin/bash
set -e

echo "Starting distributed training pipeline..."
source venv/bin/activate
python src/train.py --data data.yaml --epochs 100 --batch 16
echo "Training complete."
