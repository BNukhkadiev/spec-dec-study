"""Experiment runner for vLLM speculative decoding evaluation."""

import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from .benchmark import Benchmark
from .config import VLLMConfig
from .metrics import ExperimentMetrics, RequestMetrics
from .utils import count_tokens, extract_acceptance_metrics


class ExperimentRunner:
    """Run experiments with vLLM and collect metrics."""
    
    def __init__(self, config: VLLMConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: vLLM configuration
        """
        self.config = config
        self.llm: Optional[LLM] = None
        self.tokenizer = None
    
    def initialize(self):
        """Initialize vLLM engine."""
        print(f"Initializing vLLM with model: {self.config.model}")
        vllm_kwargs = self.config.to_vllm_kwargs()
        self.llm = LLM(**vllm_kwargs)
        # Get tokenizer for counting tokens
        # vLLM tokenizer structure varies by version - try multiple access patterns
        self.tokenizer = None
        try:
            # Try accessing tokenizer through various paths
            tokenizer_obj = self.llm.llm_engine.tokenizer
            # Check if it's a wrapper that needs unwrapping
            if hasattr(tokenizer_obj, '_tokenizer'):
                self.tokenizer = tokenizer_obj._tokenizer
            elif hasattr(tokenizer_obj, 'tokenizer'):
                self.tokenizer = tokenizer_obj.tokenizer
            elif hasattr(tokenizer_obj, 'encode'):
                # Direct tokenizer with encode method
                self.tokenizer = tokenizer_obj
            elif hasattr(self.llm.llm_engine, 'get_tokenizer'):
                self.tokenizer = self.llm.llm_engine.get_tokenizer()
            else:
                # Last resort: try to use it directly
                self.tokenizer = tokenizer_obj
        except Exception as e:
            print(f"Warning: Could not access tokenizer: {e}. Token counting will use estimates.")
            self.tokenizer = None

        # Print GPU memory state after model load
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("\n--- GPU state after model load (nvidia-smi) ---")
                print(result.stdout)
            else:
                print(f"\nWarning: nvidia-smi failed (exit {result.returncode}): {result.stderr}")
        except FileNotFoundError:
            print("\nWarning: nvidia-smi not found (non-CUDA environment?)")
        except subprocess.TimeoutExpired:
            print("\nWarning: nvidia-smi timed out")
        except Exception as e:
            print(f"\nWarning: Could not run nvidia-smi: {e}")
    
    def run_benchmark(
        self,
        benchmark: Benchmark,
        output_dir: Optional[str | Path] = None,
        save_responses: bool = True,
    ) -> ExperimentMetrics:
        """
        Run benchmark and collect metrics.
        
        Args:
            benchmark: Benchmark to run
            output_dir: Directory to save results
            save_responses: Whether to save individual responses
            
        Returns:
            ExperimentMetrics object
        """
        if self.llm is None:
            self.initialize()
        
        # Initialize metrics
        metrics = ExperimentMetrics(
            config_name=self.config.model,
            benchmark_name=Path(benchmark.file_path).stem,
            total_requests=0,
            total_walltime=0.0,
            avg_walltime=0.0,
            min_walltime=0.0,
            max_walltime=0.0,
            total_prompt_tokens=0,
            total_generated_tokens=0,
            avg_tokens_per_second=0.0,
            avg_decode_tokens_per_second=0.0,
        )
        
        # Prepare generation parameters
        gen_kwargs = self.config.to_generation_kwargs()
        sampling_params = SamplingParams(**gen_kwargs)
        
        # Get prompts
        prompts = benchmark.get_prompts()
        ids = benchmark.get_ids()
        
        print(f"\nRunning benchmark with {len(prompts)} prompts...")
        
        # Run inference with per-request timing
        outputs: list[RequestOutput] = []
        
        # Measure total time for the batch
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        # Process outputs and collect metrics
        for idx, (entry_id, prompt, output) in enumerate(zip(ids, prompts, outputs)):
            # Extract response
            generated_text = output.outputs[0].text
            
            # Count tokens - prefer using token_ids from output if available
            if hasattr(output.outputs[0], 'token_ids') and output.outputs[0].token_ids:
                generated_tokens = len(output.outputs[0].token_ids)
            elif self.tokenizer:
                generated_tokens = count_tokens(generated_text, self.tokenizer)
            else:
                # Fallback: rough estimate (1 token ≈ 4 chars)
                generated_tokens = len(generated_text) // 4
            
            # Count prompt tokens
            if self.tokenizer:
                prompt_tokens = count_tokens(prompt, self.tokenizer)
            else:
                # Fallback: rough estimate
                prompt_tokens = len(prompt) // 4
            
            # Try to get actual timing from vLLM output metrics
            walltime = None
            decode_time = None
            if hasattr(output, 'metrics'):
                output_metrics = output.metrics
                
                # Try to extract decode time (pure generation speed)
                # Decode time = time from first token to last token
                decode_time_source = None
                if hasattr(output_metrics, 'decode_time'):
                    # Direct decode_time if available
                    decode_time = output_metrics.decode_time
                    decode_time_source = 'vllm_metrics'
                elif hasattr(output_metrics, 'time_to_first_token') and hasattr(output_metrics, 'time_to_last_token'):
                    # Calculate decode_time = time_to_last_token - time_to_first_token
                    ttft = output_metrics.time_to_first_token
                    tlt = output_metrics.time_to_last_token
                    if ttft is not None and tlt is not None:
                        decode_time = tlt - ttft
                        decode_time_source = 'vllm_metrics'
                elif hasattr(output_metrics, 'time_to_first_token') and hasattr(output_metrics, 'inter_token_latency'):
                    # Approximate decode_time: (num_tokens - 1) * ITL
                    # Note: first token comes from prefill, so decode is (num_tokens - 1) iterations
                    itl = getattr(output_metrics, 'inter_token_latency', 0.0)
                    if itl > 0 and generated_tokens > 1:
                        decode_time = (generated_tokens - 1) * itl
                        decode_time_source = 'vllm_metrics_estimated'
                
                # Try to get end-to-end latency (prefill + decode)
                if hasattr(output_metrics, 'time_to_last_token'):
                    # time_to_last_token is relative to request start (includes prefill)
                    walltime = output_metrics.time_to_last_token
                elif hasattr(output_metrics, 'latency'):
                    walltime = output_metrics.latency
                elif hasattr(output_metrics, 'time_to_first_token') and decode_time is not None:
                    # Approximate: TTFT (prefill) + decode_time
                    ttft = output_metrics.time_to_first_token
                    if ttft is not None:
                        walltime = ttft + decode_time
                elif hasattr(output_metrics, 'time_to_first_token') and hasattr(output_metrics, 'inter_token_latency'):
                    # Approximate: TTFT + (num_tokens - 1) * ITL
                    ttft = output_metrics.time_to_first_token
                    itl = getattr(output_metrics, 'inter_token_latency', 0.0)
                    if ttft is not None:
                        walltime = ttft + (generated_tokens - 1) * itl if generated_tokens > 0 else ttft
            
            # If no metrics available, approximate per-request timing
            # For batched requests, distribute time proportionally by token count
            if walltime is None:
                # Calculate total tokens for all requests to distribute time proportionally
                total_generated_tokens_all = sum(
                    len(o.outputs[0].token_ids) if hasattr(o.outputs[0], 'token_ids') and o.outputs[0].token_ids 
                    else count_tokens(o.outputs[0].text, self.tokenizer) if self.tokenizer 
                    else len(o.outputs[0].text) // 4
                    for o in outputs
                )
                total_prompt_tokens_all = sum(
                    count_tokens(p, self.tokenizer) if self.tokenizer else len(p) // 4
                    for p in prompts
                )
                total_all_tokens = total_generated_tokens_all + total_prompt_tokens_all
                
                if total_all_tokens > 0:
                    # Distribute time proportionally to token count (prompt + generated)
                    request_tokens = generated_tokens + prompt_tokens
                    walltime = total_time * (request_tokens / total_all_tokens)
                else:
                    # Fallback: equal distribution
                    walltime = total_time / len(prompts) if len(prompts) > 0 else 0
            
            # If decode_time is still None, estimate it from walltime
            # Prefill is fast (parallel processing), typically 5-20% of total time for long generations
            # Strategy: Estimate prefill time and subtract from walltime
            if decode_time is None and walltime is not None and walltime > 0:
                # Estimate prefill time based on prompt tokens
                # Prefill processes tokens in parallel, so it's much faster than decode
                # Typical prefill speeds: 50-200 tokens/ms for modern GPUs
                # For Qwen2-7B on modern GPU: ~100-150 tokens/ms is reasonable
                
                if prompt_tokens > 0:
                    # Estimate prefill time: prompt_tokens / prefill_speed
                    # Use conservative estimate: 100 tokens/ms = 0.01s per 1000 tokens
                    # For small prompts, minimum prefill is ~5-10ms
                    estimated_prefill_time = max(
                        prompt_tokens * 0.00001,  # ~100 tokens/ms = 0.01ms per token
                        0.005  # Minimum 5ms for any prefill
                    )
                    # Cap prefill at reasonable fraction (max 20% of total time)
                    estimated_prefill_time = min(estimated_prefill_time, walltime * 0.20)
                    
                    # Decode time = walltime - prefill_time
                    decode_time = max(walltime - estimated_prefill_time, walltime * 0.80)
                    decode_time_source = 'estimated_from_walltime'
                else:
                    # No prompt tokens, all time is decode
                    decode_time = walltime
                    decode_time_source = 'estimated_from_walltime'
            
            # Extract acceptance metrics if available
            accepted_tokens, rejected_tokens, acceptance_rate = extract_acceptance_metrics(output)
            
            # Create request metrics
            request_metrics = RequestMetrics(
                request_id=entry_id,
                prompt=prompt,
                response=generated_text,
                walltime=walltime,
                decode_time=decode_time,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                accepted_tokens=accepted_tokens,
                rejected_tokens=rejected_tokens,
                acceptance_rate=acceptance_rate,
                metadata={
                    'output_id': idx,
                    'finish_reason': output.outputs[0].finish_reason,
                    'decode_time_source': decode_time_source if 'decode_time_source' in locals() else 'estimated_from_walltime',
                }
            )
            
            metrics.add_request(request_metrics)
        
        # Compute aggregates
        # Pass total_time for accurate aggregate throughput calculations
        metrics.compute_aggregates(total_elapsed_time=total_time)
        
        # Save results if requested
        if output_dir and save_responses:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            metrics.save(output_path)
            
            # Also save config for reproducibility
            config_path = output_path / 'config.yaml'
            self.config.save(config_path)
        
        return metrics
    
    def run_single_request(
        self,
        prompt: str,
        request_id: str = "single_request",
    ) -> RequestMetrics:
        """
        Run a single request and return metrics.
        
        Args:
            prompt: Input prompt
            request_id: Identifier for the request
            
        Returns:
            RequestMetrics object
        """
        if self.llm is None:
            self.initialize()
        
        gen_kwargs = self.config.to_generation_kwargs()
        sampling_params = SamplingParams(**gen_kwargs)
        
        # Run inference
        start_time = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        walltime = time.time() - start_time
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        # Count tokens - prefer using token_ids from output if available
        if hasattr(output.outputs[0], 'token_ids') and output.outputs[0].token_ids:
            generated_tokens = len(output.outputs[0].token_ids)
        elif self.tokenizer:
            generated_tokens = count_tokens(generated_text, self.tokenizer)
        else:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            generated_tokens = len(generated_text) // 4
        
        # Count prompt tokens
        if self.tokenizer:
            prompt_tokens = count_tokens(prompt, self.tokenizer)
        else:
            # Fallback: rough estimate
            prompt_tokens = len(prompt) // 4
        
        # Extract acceptance metrics
        accepted_tokens, rejected_tokens, acceptance_rate = extract_acceptance_metrics(output)
        
        return RequestMetrics(
            request_id=request_id,
            prompt=prompt,
            response=generated_text,
            walltime=walltime,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            accepted_tokens=accepted_tokens,
            rejected_tokens=rejected_tokens,
            acceptance_rate=acceptance_rate,
        )
    
    def cleanup(self):
        """Cleanup resources."""
        if self.llm is not None:
            # vLLM cleanup if needed
            pass
