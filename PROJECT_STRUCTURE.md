# Project Structure

```
spec-dec-study/
├── specdec_study/              # Main package
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # Module entry point
│   ├── benchmark.py           # JSONL benchmark loader
│   ├── config.py              # vLLM configuration management
│   ├── metrics.py             # Metrics collection and reporting
│   ├── runner.py              # Experiment runner
│   ├── run_experiment.py      # CLI script for single experiments
│   └── utils.py               # Utility functions
│
├── configs/                    # Configuration files
│   ├── baseline.yaml          # Baseline config (no speculative decoding)
│   └── speculative_example.yaml # Example with speculative decoding
│
├── data/                       # Benchmark data
│   └── example_benchmark.jsonl # Example benchmark file
│
├── results/                    # Experiment results (gitignored)
│
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package configuration
├── README.md                   # Main documentation
├── QUICKSTART.md              # Quick start guide
├── run.sh                      # Convenience script
└── run_batch.py                # Batch experiment runner
```

## Key Components

### Benchmark (`benchmark.py`)
- Loads JSONL benchmark files
- Validates required fields (`prompt`, optional `id`)
- Provides iterator interface

### Configuration (`config.py`)
- `VLLMConfig` dataclass for vLLM settings
- Supports YAML and JSON config files
- Handles nested and flat config structures
- Converts to vLLM kwargs

### Metrics (`metrics.py`)
- `RequestMetrics`: Per-request metrics
- `ExperimentMetrics`: Aggregated experiment metrics
- Tracks: walltime, tokens/sec, acceptance rates
- Saves to JSON/JSONL

### Runner (`runner.py`)
- `ExperimentRunner`: Main experiment execution
- Initializes vLLM engine
- Runs benchmarks and collects metrics
- Handles cleanup

## Usage Patterns

### Single Experiment
```bash
python -m specdec_study.run_experiment \
    --benchmark data/benchmark.jsonl \
    --config configs/baseline.yaml \
    --output-dir results/exp1
```

### Batch Experiments
```bash
python run_batch.py \
    --benchmark data/benchmark.jsonl \
    --configs configs/*.yaml \
    --output-base results/batch
```

### Programmatic Usage
```python
from specdec_study import Benchmark, VLLMConfig, ExperimentRunner

benchmark = Benchmark("data/benchmark.jsonl")
config = VLLMConfig.from_file("configs/baseline.yaml")
runner = ExperimentRunner(config)
metrics = runner.run_benchmark(benchmark, output_dir="results/exp")
```
