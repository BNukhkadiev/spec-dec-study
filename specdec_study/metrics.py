"""Metrics collection and reporting."""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    
    request_id: str
    prompt: str
    response: str
    
    # Timing metrics (required fields first)
    prompt_tokens: int
    generated_tokens: int
    walltime: Optional[float] = None  # End-to-end time (prefill + decode) in seconds
    decode_time: Optional[float] = None  # Pure decode time (first token to last token) in seconds
    
    # Speculative decoding metrics (optional fields)
    accepted_tokens: Optional[int] = None
    rejected_tokens: Optional[int] = None
    acceptance_rate: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate end-to-end tokens per second (includes prefill)."""
        if self.walltime and self.walltime > 0:
            return self.generated_tokens / self.walltime
        return 0.0
    
    @property
    def decode_tokens_per_second(self) -> float:
        """Calculate pure generation speed (decode only, excludes prefill)."""
        if self.decode_time and self.decode_time > 0:
            return self.generated_tokens / self.decode_time
        # Fallback to end-to-end if decode_time not available
        return self.tokens_per_second
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'prompt': self.prompt,
            'response': self.response,
            'walltime': self.walltime,
            'decode_time': self.decode_time,
            'prompt_tokens': self.prompt_tokens,
            'generated_tokens': self.generated_tokens,
            'accepted_tokens': self.accepted_tokens,
            'rejected_tokens': self.rejected_tokens,
            'acceptance_rate': self.acceptance_rate,
            'tokens_per_second': self.tokens_per_second,
            'decode_tokens_per_second': self.decode_tokens_per_second,
            'metadata': self.metadata,
        }


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment."""
    
    config_name: str
    benchmark_name: str
    total_requests: int
    
    # Timing metrics
    total_walltime: float
    avg_walltime: float
    min_walltime: float
    max_walltime: float
    
    # Token metrics
    total_prompt_tokens: int
    total_generated_tokens: int
    avg_tokens_per_second: float  # End-to-end throughput (includes prefill)
    avg_decode_tokens_per_second: float  # Pure generation speed (decode only)
    input_tokens_per_second: float = 0.0  # Prefill throughput (input tokens/sec)
    output_tokens_per_second: float = 0.0  # Overall output throughput (output tokens/sec)
    
    # Speculative decoding metrics
    avg_acceptance_rate: Optional[float] = None
    total_accepted_tokens: Optional[int] = None
    total_rejected_tokens: Optional[int] = None
    
    # Per-request metrics
    request_metrics: List[RequestMetrics] = field(default_factory=list)
    
    def add_request(self, metrics: RequestMetrics):
        """Add request metrics."""
        self.request_metrics.append(metrics)
    
    def compute_aggregates(self, total_elapsed_time: Optional[float] = None):
        """Compute aggregated metrics from request metrics.
        
        Args:
            total_elapsed_time: Total elapsed time for the batch (for accurate aggregate throughput).
                                If None, uses sum of individual walltimes.
        """
        if not self.request_metrics:
            return
        
        self.total_requests = len(self.request_metrics)
        
        # Timing
        walltimes = [m.walltime for m in self.request_metrics if m.walltime is not None and m.walltime > 0]
        if walltimes:
            self.total_walltime = sum(walltimes)
            self.avg_walltime = self.total_walltime / len(walltimes)
            self.min_walltime = min(walltimes)
            self.max_walltime = max(walltimes)
        else:
            # No timing data available
            self.total_walltime = 0.0
            self.avg_walltime = 0.0
            self.min_walltime = 0.0
            self.max_walltime = 0.0
        
        # Tokens
        self.total_prompt_tokens = sum(m.prompt_tokens for m in self.request_metrics)
        self.total_generated_tokens = sum(m.generated_tokens for m in self.request_metrics)
        
        # Tokens per second - calculate aggregate throughput correctly
        # IMPORTANT: Use total tokens / total time for accurate aggregate throughput
        # Averaging per-request rates is incorrect for batched inference
        
        # End-to-end throughput (includes prefill)
        if self.total_walltime > 0:
            # Aggregate throughput: total generated tokens / total walltime
            # This gives the true system throughput, not inflated per-request averages
            self.avg_tokens_per_second = self.total_generated_tokens / self.total_walltime
        else:
            # Fallback: average per-request rates if no aggregate timing available
            tokens_per_second = [m.tokens_per_second for m in self.request_metrics if m.walltime and m.walltime > 0]
            if tokens_per_second:
                self.avg_tokens_per_second = sum(tokens_per_second) / len(tokens_per_second)
            else:
                self.avg_tokens_per_second = 0.0
        
        # Pure decode throughput (excludes prefill) - preferred metric for generation speed
        decode_times = [m.decode_time for m in self.request_metrics if m.decode_time is not None and m.decode_time > 0]
        if decode_times:
            total_decode_time = sum(decode_times)
            # Aggregate decode throughput: total generated tokens / total decode time
            self.avg_decode_tokens_per_second = self.total_generated_tokens / total_decode_time
        else:
            # Fallback: average per-request decode rates
            decode_tokens_per_second = [m.decode_tokens_per_second for m in self.request_metrics if m.decode_time and m.decode_time > 0]
            if decode_tokens_per_second:
                self.avg_decode_tokens_per_second = sum(decode_tokens_per_second) / len(decode_tokens_per_second)
            else:
                # If no decode_time available, use end-to-end as fallback
                self.avg_decode_tokens_per_second = self.avg_tokens_per_second
        
        # Calculate input/output throughput (matching vLLM's "est. speed" metrics)
        # These are aggregate metrics over total elapsed time (like vLLM's progress bar)
        # Use total_elapsed_time if provided (more accurate for batched inference),
        # otherwise fall back to total_walltime
        elapsed_time = total_elapsed_time if total_elapsed_time is not None and total_elapsed_time > 0 else self.total_walltime
        if elapsed_time > 0:
            # Input tokens/second (prefill throughput)
            self.input_tokens_per_second = self.total_prompt_tokens / elapsed_time
            # Output tokens/second (overall output throughput)
            self.output_tokens_per_second = self.total_generated_tokens / elapsed_time
        else:
            # Fallback: zero if no timing data
            self.input_tokens_per_second = 0.0
            self.output_tokens_per_second = 0.0
        
        # Speculative decoding metrics
        acceptance_rates = [m.acceptance_rate for m in self.request_metrics if m.acceptance_rate is not None]
        if acceptance_rates:
            self.avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates)
            self.total_accepted_tokens = sum(m.accepted_tokens for m in self.request_metrics if m.accepted_tokens is not None)
            self.total_rejected_tokens = sum(m.rejected_tokens for m in self.request_metrics if m.rejected_tokens is not None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config_name': self.config_name,
            'benchmark_name': self.benchmark_name,
            'total_requests': self.total_requests,
            'total_walltime': self.total_walltime,
            'avg_walltime': self.avg_walltime,
            'min_walltime': self.min_walltime,
            'max_walltime': self.max_walltime,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_generated_tokens': self.total_generated_tokens,
            'avg_tokens_per_second': self.avg_tokens_per_second,
            'avg_decode_tokens_per_second': self.avg_decode_tokens_per_second,
            'input_tokens_per_second': self.input_tokens_per_second,
            'output_tokens_per_second': self.output_tokens_per_second,
            'avg_acceptance_rate': self.avg_acceptance_rate,
            'total_accepted_tokens': self.total_accepted_tokens,
            'total_rejected_tokens': self.total_rejected_tokens,
            'request_metrics': [m.to_dict() for m in self.request_metrics],
        }
    
    def save(self, output_dir: str | Path):
        """Save metrics to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated metrics
        metrics_file = output_dir / 'metrics.json'
        if HAS_ORJSON:
            with open(metrics_file, 'wb') as f:
                f.write(orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2))
        else:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save individual responses
        responses_file = output_dir / 'responses.jsonl'
        if HAS_ORJSON:
            with open(responses_file, 'wb') as f:
                for metrics in self.request_metrics:
                    response_data = {
                        'request_id': metrics.request_id,
                        'prompt': metrics.prompt,
                        'response': metrics.response,
                        'walltime': metrics.walltime,
                        'tokens_per_second': metrics.tokens_per_second,
                        'acceptance_rate': metrics.acceptance_rate,
                    }
                    f.write(orjson.dumps(response_data))
                    f.write(b'\n')
        else:
            with open(responses_file, 'w', encoding='utf-8') as f:
                for metrics in self.request_metrics:
                    response_data = {
                        'request_id': metrics.request_id,
                        'prompt': metrics.prompt,
                        'response': metrics.response,
                        'walltime': metrics.walltime,
                        'tokens_per_second': metrics.tokens_per_second,
                        'acceptance_rate': metrics.acceptance_rate,
                    }
                    f.write(json.dumps(response_data, ensure_ascii=False))
                    f.write('\n')
        
        print(f"Metrics saved to {output_dir}")
    
    def print_summary(self):
        """Print summary of metrics."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Config: {self.config_name}")
        print(f"Benchmark: {self.benchmark_name}")
        print(f"Total Requests: {self.total_requests}")
        print(f"\nTiming:")
        print(f"  Total Walltime: {self.total_walltime:.2f}s")
        print(f"  Avg Walltime: {self.avg_walltime:.2f}s")
        print(f"  Min Walltime: {self.min_walltime:.2f}s")
        print(f"  Max Walltime: {self.max_walltime:.2f}s")
        print(f"\nTokens:")
        print(f"  Total Prompt Tokens: {self.total_prompt_tokens}")
        print(f"  Total Generated Tokens: {self.total_generated_tokens}")
        print(f"\nThroughput:")
        print(f"  Input Tokens/Second (Prefill): {self.input_tokens_per_second:.2f}")
        print(f"  Output Tokens/Second (Overall): {self.output_tokens_per_second:.2f}")
        print(f"  Avg Tokens/Second (E2E): {self.avg_tokens_per_second:.2f}")
        print(f"  Avg Tokens/Second (Decode, estimated): {self.avg_decode_tokens_per_second:.2f}")
        
        # Check if decode_time was estimated (not from vLLM metrics)
        decode_times_from_metrics = sum(
            1 for m in self.request_metrics 
            if m.decode_time is not None 
            and m.decode_time > 0
            and m.metadata.get('decode_time_source') == 'vllm_metrics'
        )
        if decode_times_from_metrics < self.total_requests:
            print(f"  Note: Decode throughput estimated by subtracting prefill time from walltime")
        if self.avg_acceptance_rate is not None:
            print(f"\nSpeculative Decoding:")
            print(f"  Avg Acceptance Rate: {self.avg_acceptance_rate:.2%}")
            print(f"  Total Accepted Tokens: {self.total_accepted_tokens}")
            print(f"  Total Rejected Tokens: {self.total_rejected_tokens}")
        print("="*60 + "\n")
