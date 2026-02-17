# Experiment Configurations

This document describes the experiment configurations for evaluating speculative decoding algorithms on Qwen2, Qwen3, and Llama3 models.

## Models Tested

### Qwen2 Series
- **Qwen2-7B-Instruct**: `Qwen/Qwen2-7B-Instruct`
- **Qwen2-72B-Instruct**: `Qwen/Qwen2-72B-Instruct`

### Qwen3 Series
- **Qwen3-32B**: `Qwen/Qwen3-32B`

### Llama3 Series
- **Llama3-8B-Instruct**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Llama3-70B-Instruct**: `meta-llama/Meta-Llama-3-70B-Instruct`

## Self-Drafting Methods

### 1. Baseline (No Speculative Decoding)
- Standard autoregressive generation
- Config files: `*_baseline.yaml`

### 2. N-gram Self-Drafting
- Uses n-gram matching from the prompt to generate draft tokens
- No separate draft model required
- Config files: `*_ngram.yaml`
- Parameters:
  - `num_tokens`: 5 (number of speculative tokens)
  - `ngram_prompt_lookup_max`: 4 (maximum n-gram length)

### 3. EAGLE Self-Drafting (Optional)
- Uses EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)
- Requires converting EAGLE models from HuggingFace
- Config files: `*_eagle.yaml` (commented out by default)
- Available EAGLE models:
  - `yuhuili/EAGLE-Qwen2-7B-Instruct`
  - `yuhuili/EAGLE-Qwen2-72B-Instruct`
  - `yuhuili/EAGLE-LLaMA3-Instruct-8B`

## Running Experiments

### Run All Experiments
```bash
python run_all_experiments.py \
    --benchmark data/your_benchmark.jsonl \
    --group all \
    --output-base results/comprehensive_study
```

### Run Specific Model Group
```bash
# Qwen2-7B only
python run_all_experiments.py \
    --benchmark data/your_benchmark.jsonl \
    --group qwen2_7b \
    --output-base results/qwen2_7b_study
```

### Run with Filters
```bash
# Skip baselines, only run speculative decoding methods
python run_all_experiments.py \
    --benchmark data/your_benchmark.jsonl \
    --skip-baseline \
    --output-base results/speculative_only
```

### Run Individual Configurations
```bash
# Single configuration
python -m specdec_study.run_experiment \
    --benchmark data/your_benchmark.jsonl \
    --config configs/qwen2_7b_ngram.yaml \
    --output-dir results/qwen2_7b_ngram
```

## EAGLE Model Setup

To use EAGLE self-drafting:

1. Download the EAGLE model from HuggingFace (e.g., `yuhuili/EAGLE-Qwen2-7B-Instruct`)

2. Convert the model using the conversion script:
   ```bash
   # Download conversion script
   wget https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d
   
   # Convert model
   python convert_eagle_model.py \
       --input_model yuhuili/EAGLE-Qwen2-7B-Instruct \
       --output_model path/to/converted/EAGLE-Qwen2-7B-Instruct
   ```

3. Update the EAGLE config files with the converted model path

4. Uncomment EAGLE configs in `run_all_experiments.py`

## Expected Metrics

Each experiment will report:
- **Walltime**: Time per request (seconds)
- **Tokens/Second**: Generation throughput
- **Acceptance Rate**: For speculative decoding, percentage of tokens accepted
- **Total Tokens**: Prompt + generated tokens

## Output Structure

Results are saved in timestamped directories:
```
results/
├── qwen2_7b_baseline_20240216_120000/
│   ├── metrics.json
│   ├── responses.jsonl
│   └── config.yaml
├── qwen2_7b_ngram_20240216_120000/
│   └── ...
└── ...
```

## Notes

- **GPU Requirements**: 
  - 7B-8B models: 1 GPU recommended
  - 32B models: 2-4 GPUs recommended
  - 70B-72B models: 4+ GPUs required
  
- **Model Access**: Some Meta Llama models require accepting the license agreement on HuggingFace

- **Tensor Parallelism**: Adjust `tensor_parallel_size` in configs based on your GPU setup

- **Memory**: Adjust `gpu_memory_utilization` if you encounter OOM errors
