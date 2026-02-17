"""Utility functions."""

import time
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using tokenizer."""
    if tokenizer is None:
        # Fallback: rough estimate (1 token ≈ 4 chars for English)
        return len(text) // 4
    
    try:
        # Try encode method (most common)
        if hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text))
        # Try __call__ method
        elif hasattr(tokenizer, '__call__'):
            result = tokenizer(text)
            if isinstance(result, dict) and 'input_ids' in result:
                return len(result['input_ids'])
            elif isinstance(result, list):
                return len(result)
        # Fallback
        return len(text) // 4
    except Exception:
        # If anything fails, use rough estimate
        return len(text) // 4


def extract_acceptance_metrics(output: RequestOutput) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Extract acceptance metrics from vLLM output.
    
    vLLM's speculative decoding metrics may be available in different places
    depending on the version. This function tries multiple approaches.
    
    Returns:
        Tuple of (accepted_tokens, rejected_tokens, acceptance_rate)
    """
    # Try to get from output metadata
    metadata = getattr(output, 'metadata', {})
    if isinstance(metadata, dict):
        accepted = metadata.get('accepted_tokens')
        rejected = metadata.get('rejected_tokens')
        if accepted is not None and rejected is not None:
            total = accepted + rejected
            rate = accepted / total if total > 0 else 0.0
            return accepted, rejected, rate
    
    # Try to get from output object attributes
    if hasattr(output, 'accepted_tokens') and hasattr(output, 'rejected_tokens'):
        accepted = output.accepted_tokens
        rejected = output.rejected_tokens
        if accepted is not None and rejected is not None:
            total = accepted + rejected
            rate = accepted / total if total > 0 else 0.0
            return accepted, rejected, rate
    
    # Try to get from metrics if available
    if hasattr(output, 'metrics'):
        metrics = output.metrics
        if hasattr(metrics, 'accepted_tokens') and hasattr(metrics, 'rejected_tokens'):
            accepted = metrics.accepted_tokens
            rejected = metrics.rejected_tokens
            if accepted is not None and rejected is not None:
                total = accepted + rejected
                rate = accepted / total if total > 0 else 0.0
                return accepted, rejected, rate
    
    # Not available - return None
    return None, None, None
