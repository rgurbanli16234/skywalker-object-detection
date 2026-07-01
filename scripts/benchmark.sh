#!/bin/bash
set -e

echo "Running benchmark suite..."
source venv/bin/activate
python src/benchmark.py
echo "Benchmarking complete."
