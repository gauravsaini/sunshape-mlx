"""SunShape SDPA — scaled dot-product attention with block-VQ keys.

The key insight: since SunShape uses a fixed block codebook, we can
precompute dot(query_block_b, centroid[b, c]) for ALL centroids outside
the kernel.  The attention score then reduces to a pure gather + reduce:

    score[q, t] = sum_b  qdots[q, b, indices[t, b]]

This is the structural advantage of block VQ over scalar quantization:
TurboQuant's scalar codebook has only 2 entries per coordinate, so
precomputation doesn't amortize.  SunShape has 256 entries per 8-dim
block — the precompute table is small and the gather kernel is
dramatically faster than materializing the full dequantized key matrix.
"""

from __future__ import annotations

import math

import mlx.core as mx

from sunshape_mlx.codec import SunShapeBlockCodec, SunShapeQuantized
from sunshape_mlx.cache import HybridSunShapeKVCache, SunShapeKVCache
from sunshape_mlx.kernels import block_vq_score_metal


def _compressed_sunshape_scores(
    query_transformed: mx.array,
    codec: SunShapeBlockCodec,
    indices: mx.array,
) -> mx.array:
    """Score quantized keys with optional Metal gather kernel."""
    T_q = query_transformed.shape[0]
    T_kv = indices.shape[0]
    q_blocks = query_transformed.reshape(T_q, codec.n_blocks, codec.block_dim)
    qdots = mx.einsum("qbd,bcd->qbc", q_blocks, codec.centroids)

    scores = block_vq_score_metal(
        qdots,
        indices,
        n_qh=T_q,
        T_kv=T_kv,
        n_blocks=codec.n_blocks,
        n_centroids=codec.n_centroids,
    )
    if scores is not None:
        codec.last_score_backend = "metal"
        return scores

    indices_int = indices.astype(mx.int32)
    qdots_exp = qdots[:, None, :, :]
    indices_exp = indices_int[None, :, :, None]
    gathered = mx.take_along_axis(qdots_exp, indices_exp, axis=-1)
    gathered = gathered.squeeze(-1)
    codec.last_score_backend = "mlx"
    return mx.sum(gathered, axis=2)


def sunshape_sdpa(
    queries: mx.array,
    cache: SunShapeKVCache | HybridSunShapeKVCache,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    """SunShape Scaled Dot-Product Attention.

    Computes attention using precomputed query-centroid block dot products
    and gathered indices, avoiding full key dequantization.

    Parameters
    ----------
    queries : mx.array, shape (B, n_q_heads, T_q, D)
    cache : SunShapeKVCache
        Fitted cache with quantized key indices and FP values.
    scale : float
        Attention scale (typically 1/sqrt(head_dim)).
    mask : mx.array or str or None
        Causal mask specification.

    Returns
    -------
    output : mx.array, shape (B, n_q_heads, T_q, D)
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = cache.n_kv_heads
    n_repeats = n_q_heads // n_kv_heads
    T_kv = cache.offset

    if T_kv == 0:
        return mx.zeros_like(queries)

    # Scale queries
    q_scaled = queries * scale

    # --- Per-KV-head processing with GQA ---
    # Get key indices and values from cache
    key_indices = cache.key_indices   # (B, n_kv_heads, T_compressed, n_blocks)
    values = cache.values             # (B, n_kv_heads, T_kv, head_dim)
    buffer_keys = cache.buffer_keys if isinstance(cache, HybridSunShapeKVCache) else None
    compressed_offset = cache.compressed_offset if isinstance(cache, HybridSunShapeKVCache) else T_kv

    codec = cache.codec

    # Reshape queries for GQA: (B, n_kv_heads, n_repeats, T_q, D)
    q_grouped = q_scaled.reshape(B, n_kv_heads, n_repeats, T_q, D)

    # Transform queries into codec space
    if codec.mode in {
        "profileperm_baseline",
        "profileperm_localmetric_dsq",
        "profileperm_mixed_precision",
    }:
        q_transformed = q_grouped[..., codec.permutation]
    elif codec.use_rotation:
        q_transformed = q_grouped @ codec.rotation.T
    else:
        q_transformed = q_grouped

    # Flatten for precompute: (B * n_kv_heads * n_repeats, T_q * D) -> per-head
    # Process each (batch, kv_head) group
    output = mx.zeros((B, n_q_heads, T_q, D), dtype=queries.dtype)

    for b in range(B):
        for h in range(n_kv_heads):
            indices_h = key_indices[b, h]  # (T_compressed, n_blocks)
            values_h = values[b, h]       # (T_kv, head_dim)
            buffer_keys_h = buffer_keys[b, h].astype(mx.float32) if buffer_keys is not None else None

            for r in range(n_repeats):
                q_h = q_transformed[b, h, r]  # (T_q, D)
                score_parts = []

                if compressed_offset > 0:
                    score_parts.append(_compressed_sunshape_scores(q_h, codec, indices_h))

                if buffer_keys_h is not None and cache.buffer_offset > 0:
                    # Use q_grouped (not q_scaled) — q_grouped[b,h,r] is (T_q, D)
                    # in the original basis, matching the un-quantized buffer keys.
                    score_parts.append(q_grouped[b, h, r].astype(mx.float32) @ buffer_keys_h.T)

                if not score_parts:
                    scores = mx.zeros((T_q, 0), dtype=mx.float32)
                elif len(score_parts) == 1:
                    scores = score_parts[0]
                else:
                    scores = mx.concatenate(score_parts, axis=1)

                # Mask
                if mask is not None:
                    if isinstance(mask, str) and mask == "causal":
                        # Simple causal: query position >= key position
                        offset = T_kv - T_q
                        q_pos = mx.arange(offset, offset + T_q)
                        k_pos = mx.arange(T_kv)
                        causal = q_pos[:, None] >= k_pos[None, :]
                        scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
                    elif isinstance(mask, mx.array):
                        if mask.dtype == mx.bool_:
                            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                        else:
                            scores = scores + mask

                # Softmax
                weights = mx.softmax(scores.astype(mx.float32), axis=-1, precise=True)

                # Value output
                o = weights @ values_h.astype(mx.float32)  # (T_q, head_dim)
                q_head_idx = h * n_repeats + r
                output[b, q_head_idx] = o.astype(queries.dtype)

    mx.eval(output)
    return output
