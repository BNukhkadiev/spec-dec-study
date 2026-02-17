#!/usr/bin/env python3
"""Run multiple experiments with different configurations."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from specdec_study.benchmark import Benchmark
from specdec_study.config import VLLMConfig
from specdec_study.runner import ExperimentRunner


def main():
    """Run batch experiments."""
    parser = argparse.ArgumentParser(
        description="Run multiple experiments with different configurations"
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Path to JSONL benchmark file'
    )
    
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to config files (YAML/JSON)'
    )
    
    parser.add_argument(
        '--output-base',
        type=str,
        default='results',
        help='Base output directory (default: results)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save responses to disk'
    )
    
    args = parser.parse_args()
    
    # Load benchmark
    try:
        benchmark = Benchmark(args.benchmark)
        print(f"Loaded benchmark with {len(benchmark)} entries")
    except Exception as e:
        print(f"Error loading benchmark: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run each config
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_path in args.configs:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}", file=sys.stderr)
            continue
        
        print(f"\n{'='*60}")
        print(f"Running experiment with config: {config_path.name}")
        print(f"{'='*60}")
        
        try:
            # Load config
            config = VLLMConfig.from_file(config_path)
            
            # Create output directory
            config_name = config_path.stem
            output_dir = Path(args.output_base) / f"{config_name}_{timestamp}"
            
            # Run experiment
            runner = ExperimentRunner(config)
            try:
                metrics = runner.run_benchmark(
                    benchmark=benchmark,
                    output_dir=output_dir,
                    save_responses=not args.no_save,
                )
                
                results.append({
                    'config': config_name,
                    'output_dir': str(output_dir),
                    'metrics': metrics,
                })
                
                metrics.print_summary()
                
            finally:
                runner.cleanup()
                
        except KeyboardInterrupt:
            print("\nBatch experiments interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error running experiment with {config_path}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    if results:
        print(f"\n{'='*60}")
        print("BATCH EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for result in results:
            m = result['metrics']
            print(f"\n{result['config']}:")
            print(f"  Avg Tokens/Second: {m.avg_tokens_per_second:.2f}")
            print(f"  Avg Walltime: {m.avg_walltime:.2f}s")
            if m.avg_acceptance_rate is not None:
                print(f"  Avg Acceptance Rate: {m.avg_acceptance_rate:.2%}")
            print(f"  Output: {result['output_dir']}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
