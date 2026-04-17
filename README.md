# SunShape MLX

**Apple Silicon KV-cache compression via block vector quantization.**

SunShape MLX ports the [SunShape](https://github.com/gauravsaini/sunshape) PyTorch library to Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

## Why SunShape over TurboQuant?

| Feature | TurboQuant (scalar) | SunShape (block VQ) |
|---------|--------------------|--------------------|
| Quantization | Lloyd-Max scalar (1-4 bits per coordinate) | k-means block VQ (8-dim blocks) |
| Spatial structure | None (random rotation only) | ProfilePerm covariance-aware permutation |
| Attention kernel | Dequantize all keys → matmul | Precomputed query-centroid dots → gather |
| Memory traffic | O(T_kv × D) | O(T_kv × n_blocks) — up to 16× less |
| Codebook size | 2-16 scalars | 256 × 8-dim vectors per block |

The key structural advantage: SunShape's block codebook allows **precomputing** all query-centroid dot products once per decode step. The attention score then reduces to a pure **gather** operation over the index tensor, which is dramatically faster than materializing the full dequantized key matrix.

## Installation

```bash
pip install -e .
```

Requires: `mlx>=0.22.0`, `mlx-lm>=0.21.0`, `numpy`

## Quick Start

### Standalone codec usage

```python
import mlx.core as mx
from sunshape_mlx import SunShapeBlockCodec, SunShapeKVCache

# 1. Generate calibration data (in practice: extract from model forward pass)
q_cal = mx.random.normal((1024, 128))  # queries, shape (N, head_dim)
k_cal = mx.random.normal((2048, 128))  # keys

# 2. Fit the codec
codec = SunShapeBlockCodec(
    head_dim=128,
    block_dim=8,
    n_centroids=256,       # 1.0 bit/dim
    mode="profileperm_baseline",
)
codec.fit(q_cal, k_cal, kmeans_iters=15, seed=42)

# 3. Quantize and dequantize
k_hat = codec(k_cal)  # round-trip

# 4. Compute attention scores (no full dequantization!)
quantized = codec.quantize(k_cal[:128])
scores = codec.attention_scores(q_cal[:8], quantized)  # (8, 128)
```

### With mlx-lm inference

```python
from sunshape_mlx import SunShapeBlockCodec, create_kv_cache
from sunshape_mlx.patch import apply as apply_patch

# Apply the monkey-patch to mlx-lm's attention
apply_patch()

# 1. Fit the codec
codec = SunShapeBlockCodec(head_dim=128, block_dim=8, n_centroids=256, mode="profileperm_baseline")
codec.fit(q_cal, k_cal)

# 2. Create per-layer caches — using the unified factory function
# Option A: SunShape Keys + FP16 Values
caches = [create_kv_cache(codec, n_kv_heads=8) for _ in range(n_layers)]

# Option B: Hybrid (SunShape Keys + TurboQuant 2-bit Values)
# caches = [create_kv_cache(codec, n_kv_heads=8, value_backend="turboquant", value_bits=2) 
#           for _ in range(n_layers)]

# Run inference — the patched SDPA will use SunShape's fast path
output = model(input_ids, cache=caches)
```

## Modes

| Mode | Description |
|------|-------------|
| `profileperm_baseline` | ProfilePerm permutation + plain block VQ (recommended) |
| `profileperm_localmetric_dsq` | ProfilePerm + local covariance metric + DSQ refinement |
| `profileperm_mixed_precision` | ProfilePerm + FP heavy blocks + VQ on light blocks |
| `rotated` | Dense random rotation + block VQ (legacy ablation) |

## Architecture

```
sunshape_mlx/
├── __init__.py        # Package exports
├── rotation.py        # ProfilePerm permutation + random rotation generation
├── codec.py           # SunShapeBlockCodec — block VQ with k-means
├── cache.py           # SunShapeKVCache — drop-in for mlx-lm KVCache
├── attention.py       # Custom SDPA with precomputed query-centroid dots
├── kernels.py         # Metal kernel templates + index packing utilities
├── value_codecs.py    # Value backends: grouped / TurboQuant-style scalar quant
└── patch.py           # Monkey-patch for mlx-lm integration
```

## Benchmarks

Run benchmark matrix:

```bash
uv run --with mlx --with mlx-lm --with transformers python examples_mlx/benchmark.py \
  --model Qwen/Qwen3.5-0.8B
```

Current configs:

- `2bit_tq_kv`
- `1bit_sunshape_k__2bit_tq_v`
- `1bit_sunshape_kv`
- `2bit_sunshape_kv`

Runtime benchmark with generation + perplexity + compression:

```bash
uv run --with mlx --with mlx-lm python examples_mlx/runtime_benchmark.py \
  --model mlx-community/Qwen3.5-4B-OptiQ-4bit
```

## Testing

```bash
pytest tests_mlx/ -v
```