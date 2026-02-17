# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running Your First Experiment

1. **Prepare a benchmark file** (JSONL format):
   ```bash
   echo '{"prompt": "What is machine learning?", "id": "test_001"}' > data/my_benchmark.jsonl
   ```

2. **Create a config file** (optional, or use `--model` directly):
   ```yaml
   # configs/my_config.yaml
   model: meta-llama/Llama-2-7b-chat-hf
   generation:
     max_tokens: 256
     temperature: 0.7
   ```

3. **Run the experiment**:
   ```bash
   python -m specdec_study.run_experiment \
       --benchmark data/my_benchmark.jsonl \
       --config configs/my_config.yaml \
       --output-dir results/my_experiment
   ```

## Understanding Results

After running an experiment, check the output directory:

- `metrics.json`: Aggregated metrics (walltime, tokens/sec, acceptance rates)
- `responses.jsonl`: Individual responses for each prompt
- `config.yaml`: Configuration used (for reproducibility)

## Key Metrics

- **Walltime**: Time taken per request (seconds)
- **Tokens/Second**: Generation throughput
- **Acceptance Rate**: For speculative decoding, percentage of tokens accepted
- **Total Tokens**: Prompt + generated tokens

## Speculative Decoding Configuration

To enable speculative decoding in vLLM, configure it in your YAML:

```yaml
model: meta-llama/Llama-2-7b-chat-hf

speculative_decoding:
  model: meta-llama/Llama-2-7b-chat-hf  # Draft model
  num_tokens: 5  # Number of speculative tokens

generation:
  max_tokens: 512
```

**Note**: The exact vLLM API for speculative decoding may vary by version. Check the [vLLM documentation](https://docs.vllm.ai/) for the latest API.

## Batch Experiments

Compare multiple configurations:

```bash
python run_batch.py \
    --benchmark data/benchmark.jsonl \
    --configs configs/baseline.yaml configs/speculative.yaml \
    --output-base results/comparison
```
