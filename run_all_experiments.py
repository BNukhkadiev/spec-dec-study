#!/usr/bin/env python3
"""Run all experiment configurations for Qwen2, Qwen3, and Llama3 models."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from specdec_study.benchmark import Benchmark
from specdec_study.config import VLLMConfig
from specdec_study.runner import ExperimentRunner


# Define experiment groups
EXPERIMENT_GROUPS = {
    'qwen2_7b': [
        'configs/qwen2_7b_baseline.yaml',
        'configs/qwen2_7b_ngram.yaml',
        # 'configs/qwen2_7b_eagle.yaml',  # Uncomment when EAGLE model is converted
    ],
    'qwen2_72b': [
        'configs/qwen2_72b_baseline.yaml',
        'configs/qwen2_72b_ngram.yaml',
        # 'configs/qwen2_72b_eagle.yaml',  # Uncomment when EAGLE model is converted
    ],
    'qwen3_32b': [
        'configs/qwen3_32b_baseline.yaml',
        'configs/qwen3_32b_ngram.yaml',
    ],
    'llama3_8b': [
        'configs/llama3_8b_baseline.yaml',
        'configs/llama3_8b_ngram.yaml',
        # 'configs/llama3_8b_eagle.yaml',  # Uncomment when EAGLE model is converted
    ],
    'llama3_70b': [
        'configs/llama3_70b_baseline.yaml',
        'configs/llama3_70b_ngram.yaml',
    ],
    'all': None,  # Will be populated with all configs
}


def get_all_configs():
    """Get all config files."""
    configs = []
    for group_name, group_configs in EXPERIMENT_GROUPS.items():
        if group_name != 'all' and group_configs:
            configs.extend(group_configs)
    return configs


def main():
    """Run batch experiments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments for Qwen2, Qwen3, and Llama3 models"
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Path to JSONL benchmark file'
    )
    
    parser.add_argument(
        '--group',
        type=str,
        choices=list(EXPERIMENT_GROUPS.keys()),
        default='all',
        help='Experiment group to run (default: all)'
    )
    
    parser.add_argument(
        '--output-base',
        type=str,
        default='results',
        help='Base output directory (default: results)'
    )
    
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline (no speculative decoding) experiments'
    )
    
    parser.add_argument(
        '--skip-ngram',
        action='store_true',
        help='Skip n-gram self-drafting experiments'
    )
    
    parser.add_argument(
        '--skip-eagle',
        action='store_true',
        help='Skip EAGLE self-drafting experiments'
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
    
    # Get configs to run
    if args.group == 'all':
        configs_to_run = get_all_configs()
    else:
        configs_to_run = EXPERIMENT_GROUPS[args.group]
    
    # Filter configs based on skip flags
    filtered_configs = []
    for config_path in configs_to_run:
        config_path_str = str(config_path)
        if args.skip_baseline and 'baseline' in config_path_str:
            continue
        if args.skip_ngram and 'ngram' in config_path_str:
            continue
        if args.skip_eagle and 'eagle' in config_path_str:
            continue
        filtered_configs.append(config_path)
    
    if not filtered_configs:
        print("No configurations to run after filtering.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Running {len(filtered_configs)} experiment configurations")
    print(f"{'='*60}\n")
    
    # Run each config
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_path in filtered_configs:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}", file=sys.stderr)
            continue
        
        print(f"\n{'='*60}")
        print(f"Running: {config_path.name}")
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
            print("\nExperiments interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error running experiment with {config_path}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    
    # Print comprehensive summary
    if results:
        print(f"\n{'='*60}")
        print("COMPREHENSIVE EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        # Group by model
        by_model = {}
        for result in results:
            config_name = result['config']
            # Extract model name (e.g., 'qwen2_7b' from 'qwen2_7b_baseline')
            model_name = '_'.join(config_name.split('_')[:-1])
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(result)
        
        for model_name, model_results in sorted(by_model.items()):
            print(f"\n{model_name.upper()}:")
            print("-" * 60)
            for result in sorted(model_results, key=lambda x: x['config']):
                m = result['metrics']
                config_type = result['config'].split('_')[-1]
                print(f"  {config_type:15s} | "
                      f"Tokens/s: {m.avg_tokens_per_second:7.2f} | "
                      f"Walltime: {m.avg_walltime:6.2f}s", end="")
                if m.avg_acceptance_rate is not None:
                    print(f" | Acceptance: {m.avg_acceptance_rate:6.2%}")
                else:
                    print()
                print(f"    Output: {result['output_dir']}")
        
        print(f"\n{'='*60}")
        print(f"All results saved to: {Path(args.output_base).absolute()}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
