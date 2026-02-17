# Speculative Decoding Study

An experimentation environment for evaluating speculative decoding algorithms using vLLM.

## Features

- **Custom Benchmarks**: Load benchmarks from JSONL format
- **Comprehensive Metrics**: Track walltime, acceptance rates, and tokens/second
- **Response Storage**: Save model responses for each benchmark run
- **Flexible Configuration**: Run various vLLM settings from config files or command-line

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run an experiment with a benchmark file:

```bash
python -m specdec_study.run_experiment \
    --benchmark data/benchmark.jsonl \
    --model meta-llama/Llama-2-7b-chat-hf \
    --output-dir results/experiment_001
```

Or use the convenience script:

```bash
./run.sh data/benchmark.jsonl configs/baseline.yaml results/my_experiment
```

### Using Configuration Files

Create a vLLM configuration file (YAML or JSON):

```yaml
# configs/baseline.yaml
model: meta-llama/Llama-2-7b-chat-hf

generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 1.0

engine:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
```

Run with config:

```bash
python -m specdec_study.run_experiment \
    --benchmark data/benchmark.jsonl \
    --config configs/baseline.yaml \
    --output-dir results/baseline
```

### Batch Experiments

Run multiple configurations:

```bash
python run_batch.py \
    --benchmark data/benchmark.jsonl \
    --configs configs/baseline.yaml configs/speculative.yaml \
    --output-base results/batch_run
```

### Benchmark Format

Benchmarks should be in JSONL format, with each line containing:

```json
{"prompt": "Your prompt text here", "id": "example_001"}
```

The `id` field is optional and will be auto-generated if not provided.

## Project Structure

```
spec-dec-study/
├── specdec_study/
│   ├── __init__.py
│   ├── benchmark.py      # Benchmark loader
│   ├── config.py         # Configuration management
│   ├── metrics.py        # Metrics collection
│   ├── runner.py         # Experiment runner
│   └── utils.py          # Utility functions
├── configs/              # vLLM configuration files
│   ├── qwen2_7b_*.yaml   # Qwen2-7B configurations
│   ├── qwen2_72b_*.yaml  # Qwen2-72B configurations
│   ├── qwen3_32b_*.yaml  # Qwen3-32B configurations
│   └── llama3_*.yaml     # Llama3 configurations
├── data/                 # Benchmark data files
├── results/              # Experiment results (gitignored)
├── run_all_experiments.py # Comprehensive experiment runner
├── run_batch.py          # Batch experiment runner
├── requirements.txt
├── README.md
├── EXPERIMENTS.md        # Experiment documentation
└── QUICK_REFERENCE.md    # Quick reference guide
```

## Pre-configured Experiments

The project includes ready-to-use configurations for:

- **Qwen2-7B-Instruct** and **Qwen2-72B-Instruct**
- **Qwen3-32B**
- **Llama3-8B-Instruct** and **Llama3-70B-Instruct**

Each model has configurations for:
- Baseline (no speculative decoding)
- N-gram self-drafting (prompt-based)
- EAGLE self-drafting (where compatible)

See `EXPERIMENTS.md` for details.

## Metrics

The system tracks:
- **Walltime**: Total execution time per request
- **Acceptance Rate**: Token acceptance rate for speculative decoding
- **Tokens/Second**: Generation throughput
- **Model Responses**: Full responses saved for analysis

## Output Format

Each experiment creates an output directory with:
- `metrics.json`: Aggregated metrics and per-request details
- `responses.jsonl`: Individual responses (one per line)
- `config.yaml`: Configuration used for reproducibility

## Notes

- **vLLM API Compatibility**: The exact vLLM API for speculative decoding may vary by version. Check the [vLLM documentation](https://docs.vllm.ai/) for the latest API and adjust the configuration accordingly.
- **Token Counting**: Token counts use the model's tokenizer, which may differ from other tokenizers.
- **Timing**: For batched requests, walltime is approximated by dividing total time evenly across requests. For more accurate per-request timing, consider running requests individually.
