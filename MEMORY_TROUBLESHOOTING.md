# Memory Troubleshooting Guide

## Common CUDA OOM Issues with Speculative Decoding

Speculative decoding requires additional GPU memory beyond the base model due to:
- Rejection sampler overhead
- Additional CUDA graphs for speculative paths
- KV cache for draft tokens
- Temporary buffers during token verification

## Solutions

### 1. Reduce GPU Memory Utilization

Lower `gpu_memory_utilization` in your config:

```yaml
engine:
  gpu_memory_utilization: 0.75  # Instead of 0.9
```

**Recommended values:**
- Baseline (no speculative decoding): 0.85-0.9
- With speculative decoding: 0.70-0.75
- Large models (70B+): 0.65-0.70

### 2. Reduce Maximum Model Length

The default `max_model_len` of 32768 requires significant KV cache memory:

```yaml
engine:
  max_model_len: 8192  # Or 4096 if you don't need long context
```

**Memory impact:**
- 32768 tokens: ~5-6 GB KV cache
- 8192 tokens: ~1.5 GB KV cache
- 4096 tokens: ~0.75 GB KV cache

### 3. Reduce Number of Speculative Tokens

Fewer speculative tokens = less memory overhead:

```yaml
speculative_decoding:
  num_tokens: 3  # Instead of 5
```

### 4. Use Quantization (Advanced)

For very large models, consider quantization:

```yaml
engine:
  quantization: awq  # or gptq
```

### 5. Set PyTorch Memory Allocation

If you see fragmentation warnings, try:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 6. Disable CUDA Graphs (Last Resort)

If still OOM, you can disable CUDA graphs (slower but uses less memory):

```python
# In your code, add to vLLM kwargs:
kwargs['enforce_eager'] = True
```

## Memory Usage Estimates

For a 7B model on RTX 3090 (24GB):

| Configuration | Model Weights | KV Cache | Overhead | Total |
|--------------|---------------|----------|----------|-------|
| Baseline, max_len=8192 | ~14 GB | ~1.5 GB | ~1 GB | ~16.5 GB |
| Baseline, max_len=32768 | ~14 GB | ~5 GB | ~1 GB | ~20 GB |
| SpecDec, max_len=8192 | ~14 GB | ~1.5 GB | ~3 GB | ~18.5 GB |
| SpecDec, max_len=32768 | ~14 GB | ~5 GB | ~3 GB | ~22 GB |

## Quick Fix for Your Current Issue

Update your config to:

```yaml
engine:
  gpu_memory_utilization: 0.75
  max_model_len: 8192
```

This should reduce memory usage from ~23.6 GB to ~18-19 GB, leaving headroom for speculative decoding overhead.
