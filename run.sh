#!/bin/bash
# Convenience script to run experiments

set -e

# Default values
BENCHMARK="${1:-data/example_benchmark.jsonl}"
CONFIG="${2:-configs/baseline.yaml}"
OUTPUT_DIR="${3:-results/experiment_$(date +%Y%m%d_%H%M%S)}"

echo "Running experiment:"
echo "  Benchmark: $BENCHMARK"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo ""

python -m specdec_study.run_experiment \
    --benchmark "$BENCHMARK" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR"
