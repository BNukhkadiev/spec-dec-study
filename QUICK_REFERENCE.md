# Quick Reference: Running Experiments

## Available Configurations

### Qwen2-7B-Instruct
- `configs/qwen2_7b_baseline.yaml` - Baseline (no speculative decoding)
- `configs/qwen2_7b_ngram.yaml` - N-gram self-drafting
- `configs/qwen2_7b_eagle.yaml` - EAGLE self-drafting (requires model conversion)

### Qwen2-72B-Instruct
- `configs/qwen2_72b_baseline.yaml` - Baseline
- `configs/qwen2_72b_ngram.yaml` - N-gram self-drafting
- `configs/qwen2_72b_eagle.yaml` - EAGLE self-drafting (requires model conversion)

### Qwen3-32B
- `configs/qwen3_32b_baseline.yaml` - Baseline
- `configs/qwen3_32b_ngram.yaml` - N-gram self-drafting

### Llama3-8B-Instruct
- `configs/llama3_8b_baseline.yaml` - Baseline
- `configs/llama3_8b_ngram.yaml` - N-gram self-drafting
- `configs/llama3_8b_eagle.yaml` - EAGLE self-drafting (requires model conversion)

### Llama3-70B-Instruct
- `configs/llama3_70b_baseline.yaml` - Baseline
- `configs/llama3_70b_ngram.yaml` - N-gram self-drafting

## Quick Start Commands

### Run All Experiments
```bash
python run_all_experiments.py \
    --benchmark data/your_benchmark.jsonl \
    --output-base results/study
```

### Run Single Model Group
```bash
# Qwen2-7B only
python run_all_experiments.py \
    --benchmark data/your_benchmark.jsonl \
    --group qwen2_7b
```

### Run Only Speculative Decoding (Skip Baselines)
```bash
python run_all_experiments.py \
    --benchmark data/your_benchmark.jsonl \
    --skip-baseline
```

### Run Single Configuration
```bash
python -m specdec_study.run_experiment \
    --benchmark data/your_benchmark.jsonl \
    --config configs/qwen2_7b_ngram.yaml \
    --output-dir results/qwen2_7b_ngram_test
```

## Model Compatibility

### Self-Drafting Methods Available

| Model | Baseline | N-gram | EAGLE |
|-------|----------|--------|-------|
| Qwen2-7B-Instruct | ✅ | ✅ | ✅* |
| Qwen2-72B-Instruct | ✅ | ✅ | ✅* |
| Qwen3-32B | ✅ | ✅ | ❌ |
| Llama3-8B-Instruct | ✅ | ✅ | ✅* |
| Llama3-70B-Instruct | ✅ | ✅ | ❌ |

*EAGLE requires model conversion - see EXPERIMENTS.md

## Expected Output

Each experiment creates:
- `metrics.json` - Aggregated metrics
- `responses.jsonl` - Individual responses
- `config.yaml` - Configuration used

Key metrics reported:
- Average tokens/second
- Average walltime per request
- Acceptance rate (for speculative decoding)
