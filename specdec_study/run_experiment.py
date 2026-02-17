#!/usr/bin/env python3
"""Main script to run experiments."""

import argparse
import sys
from pathlib import Path

from .benchmark import Benchmark
from .config import VLLMConfig
from .runner import ExperimentRunner


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run speculative decoding experiments with vLLM"
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Path to JSONL benchmark file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model name/path (required if --config not provided)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML/JSON config file (optional)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated from config and dataset names)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save responses to disk'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = VLLMConfig.from_file(args.config)
        print(f"Loaded config from {args.config}")
    elif args.model:
        config = VLLMConfig(model=args.model)
        print(f"Using model: {args.model}")
    else:
        parser.error("Either --model or --config must be provided")
    
    # Load benchmark
    try:
        benchmark = Benchmark(args.benchmark)
        print(f"Loaded benchmark with {len(benchmark)} entries")
    except Exception as e:
        print(f"Error loading benchmark: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        # Extract config name (without path and extension)
        if args.config:
            config_name = Path(args.config).stem
        else:
            # Use model name, sanitized for filesystem
            config_name = args.model.replace('/', '_').replace('\\', '_')
        
        # Extract dataset name (without path and extension)
        dataset_name = Path(args.benchmark).stem
        
        # Create output directory name
        args.output_dir = f"results/{config_name}_{dataset_name}"
        print(f"Auto-generated output directory: {args.output_dir}")
    
    # Run experiment
    runner = ExperimentRunner(config)
    try:
        metrics = runner.run_benchmark(
            benchmark=benchmark,
            output_dir=args.output_dir,
            save_responses=not args.no_save,
        )
        
        # Print summary
        metrics.print_summary()
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
