"""SunShape MLX — Apple Silicon KV-cache compression via block vector quantization.

SunShape uses ProfilePerm (covariance-aware permutation) + block VQ (k-means)
to compress KV caches with higher fidelity than scalar quantization at the same
bitrate.  This package ports the PyTorch SunShape library to MLX for
Apple Silicon inference.

Quick start
-----------
>>> from sunshape_mlx import SunShapeBlockCodec, create_kv_cache
>>> from sunshape_mlx.patch import apply as apply_patch
>>> codec = SunShapeBlockCodec(head_dim=128)
>>> codec.fit(q_cal, k_cal)
>>>
>>> # SunShape keys + FP16 values
>>> cache = create_kv_cache(codec, n_kv_heads=8)
>>>
>>> # SunShape keys + TurboQuant 2-bit values
>>> cache = create_kv_cache(codec, n_kv_heads=8,
...     value_backend="turboquant", value_bits=2)
>>>
>>> apply_patch()   # monkey-patch mlx-lm SDPA
"""

from sunshape_mlx.rotation import (
    generate_rotation_matrix,
    covariance_block_permutation,
    invert_permutation,
    apply_permutation,
    block_local_cov_metric,
    block_affinity_gate,
    mixed_precision_block_mask,
    positive_excess_kurtosis,
    safe_normalize,
    generate_wht_signs,
    apply_wht,
)
from sunshape_mlx.codec import SunShapeBlockCodec, SunShapeQuantized
from sunshape_mlx.cache import SunShapeKVCache, HybridSunShapeKVCache, create_kv_cache
from sunshape_mlx.attention import sunshape_sdpa
from sunshape_mlx.kernels import (
    pack_indices,
    unpack_indices,
    quantize_values,
    dequantize_values,
    fused_block_vq_attention,
    fused_attention_metal,
    fused_attention_causal_metal,
    block_vq_quantize_metal,
    get_kernel_stats,
    reset_kernel_stats,
)
from sunshape_mlx.value_codecs import GroupedValueCodec, TurboQuantValueCodec
from sunshape_mlx.turboquant_runtime import TurboQuantKVCache, turboquant_sdpa

__version__ = "0.3.0"

__all__ = [
    "SunShapeBlockCodec",
    "SunShapeQuantized",
    "SunShapeKVCache",
    "HybridSunShapeKVCache",
    "sunshape_sdpa",
    "generate_rotation_matrix",
    "generate_wht_signs",
    "apply_wht",
    "covariance_block_permutation",
    "invert_permutation",
    "apply_permutation",
    "block_local_cov_metric",
    "block_affinity_gate",
    "mixed_precision_block_mask",
    "positive_excess_kurtosis",
    "safe_normalize",
    "pack_indices",
    "unpack_indices",
    "quantize_values",
    "dequantize_values",
    "fused_block_vq_attention",
    "fused_attention_metal",
    "fused_attention_causal_metal",
    "block_vq_quantize_metal",
    "GroupedValueCodec",
    "TurboQuantValueCodec",
    "TurboQuantKVCache",
    "turboquant_sdpa",
    "create_kv_cache",
    "get_kernel_stats",
    "reset_kernel_stats",
]
