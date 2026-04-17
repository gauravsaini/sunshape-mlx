"""SunShape Metal kernels — fused block-VQ attention for Apple Silicon.

Provides custom Metal kernels that fuse the precomputed-query-centroid
gather + score accumulation into a single GPU dispatch.  This is the
structural advantage of block VQ: the kernel reads only the [n_tokens, n_blocks]
index tensor + the small [n_blocks, n_centroids, block_dim] codebook,
instead of the full [n_tokens, head_dim] key matrix.

Kernel design
-------------
- Threadgroup: 32 simdgroups × 32 threads = 1024 threads per query head
- Each simdgroup processes a slice of the KV sequence
- Online softmax with cross-simdgroup reduction
- Output in the original (un-rotated) basis

Kernels provided
-----------------
1. ``sunshape_block_vq_score`` — Score-only kernel (legacy, for debug)
2. ``sunshape_fused_attention`` — **Primary**: Fused online-softmax gather+score+softmax+value
    accumulation in a single pass. Flash-Attention style. No intermediate scores tensor.
3. ``sunshape_scalar_quantize`` — Scalar quantization against sorted boundaries
4. ``sunshape_block_vq_quantize`` — Vectorized block-VQ assignment (replaces Python loop)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import mlx.core as mx


_KERNEL_STATS: dict[str, int] = defaultdict(int)


def _record_kernel_dispatch(name: str) -> None:
    _KERNEL_STATS[name] += 1


def reset_kernel_stats() -> None:
    """Reset MLX kernel dispatch counters."""
    _KERNEL_STATS.clear()


def get_kernel_stats() -> dict[str, int]:
    """Return MLX kernel dispatch counters."""
    return dict(_KERNEL_STATS)


# ================================================================== #
#  Metal kernel: Fused block-VQ attention score (legacy/debug)        #
# ================================================================== #
#
# Adapted from turboQuantPlayground's mse_score.py Metal kernel.
# SunShape-specific: each thread processes one (query_head, kv_token) pair,
# reads packed block indices, gathers precomputed query-centroid dots,
# and accumulates the attention score.

_BLOCK_VQ_SCORE_SOURCE = r"""
uint t = thread_position_in_grid.x;
uint qh = thread_position_in_grid.y;

uint T_kv = indices_shape[0];
uint n_blocks = indices_shape[1];
uint n_qh = qdots_shape[0];
uint n_centroids = qdots_shape[2];

if (t >= T_kv || qh >= n_qh) return;

float score = 0.0f;
for (uint b = 0; b < n_blocks; b++) {
    uint idx = (uint)indices[t * n_blocks + b];
    score += qdots[(qh * n_blocks + b) * n_centroids + idx];
}

out[qh * T_kv + t] = score;
"""

_block_vq_score_kernel = mx.fast.metal_kernel(
    name="sunshape_block_vq_score",
    input_names=["qdots", "indices"],
    output_names=["out"],
    source=_BLOCK_VQ_SCORE_SOURCE,
)


def block_vq_score_metal(
    qdots: mx.array,
    indices: mx.array,
    n_qh: int,
    T_kv: int,
    n_blocks: int,
    n_centroids: int,
) -> mx.array | None:
    """Try to run the Metal block-VQ score kernel.

    Returns None if Metal kernels are unavailable (falls back to pure-MLX).
    """
    try:
        mx.eval(qdots, indices)

        # Launch
        grid = (T_kv, n_qh, 1)
        output = _block_vq_score_kernel(
            inputs=[qdots, indices],
            output_shapes=[(n_qh, T_kv)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=(32, 1, 1),
        )
        out = output[0] if isinstance(output, (list, tuple)) else output
        mx.eval(out)
        _record_kernel_dispatch("sunshape_block_vq_score_metal")
        return out
    except Exception:
        return None


# ================================================================== #
#  Metal kernel: Fused online-softmax block-VQ attention              #
#  (Flash-Attention style — single pass, no intermediate scores)      #
# ================================================================== #
#
# Optimized design:
# - Threadgroup: head_dim threads (e.g., 128) per query head
# - qdots table cached in threadgroup shared memory (16KB for 16×256)
# - Thread-per-dimension layout: each thread owns exactly 1 output dim
# - No SIMD reduction needed — all threads see the same score via
#   shared tg_qdots, so online softmax state is implicitly synchronized
# - Value loop eliminated: 1 load + 1 FMA per token per thread
#    (was 128 loads + 128 FMA in the old design)
#
# Grid: (n_qh * head_dim, 1, 1)
# Threadgroup: (head_dim, 1, 1)
#
# This eliminates the O(n_qh × T_kv) intermediate scores tensor.

# Maximum qdots table size in floats: n_blocks * n_centroids
# Typical: 16 * 256 = 4096  (16 KB).  Metal threadgroup limit is 32 KB.
_MAX_QDOTS_TG = 16 * 256  # compile-time upper bound for threadgroup array

_FUSED_ATTENTION_SOURCE = r"""
// -------------------------------------------------------------------
// Fused online-softmax block-VQ attention kernel (optimized)
// -------------------------------------------------------------------
// Inputs:
//   qdots:   (n_qh, n_blocks, n_centroids) float32
//   indices: (T_kv, n_blocks)              uint8/uint16
//   values:  (T_kv, head_dim)              float32
//   params:  (4,)                          float32
//            [0]=T_kv, [1]=n_blocks, [2]=n_centroids, [3]=head_dim
// Output:
//   out:     (n_qh, head_dim) float32
//
// head_dim threads per query head.  Thread-per-dimension layout.
// qdots cached in threadgroup memory.  No SIMD reduction.
// -------------------------------------------------------------------

uint head_dim = (uint)params[3];
uint qh = thread_position_in_grid.x / head_dim;
uint dim_id = thread_position_in_threadgroup.x;

uint T_kv    = (uint)params[0];
uint n_blocks   = (uint)params[1];
uint n_centroids = (uint)params[2];
uint n_qh = qdots_shape[0];

if (qh >= n_qh || dim_id >= head_dim) return;

// ---- Cache qdots[qh, :, :] in threadgroup memory ----
// Size: n_blocks * n_centroids floats (typically 16 * 256 = 4096 = 16 KB)
threadgroup float tg_qdots[4096];  // 16 * 256 max

uint qdots_size = n_blocks * n_centroids;
uint qdots_base = qh * qdots_size;
for (uint i = dim_id; i < qdots_size; i += head_dim) {
    tg_qdots[i] = qdots[qdots_base + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- Per-thread online softmax state (single dimension) ----
float m_local = -1e30f;
float d_local = 0.0f;
float acc = 0.0f;

// ---- Process ALL tokens (each thread handles its own dim_id) ----
for (uint t = 0; t < T_kv; t++) {
    // Gather score from shared threadgroup memory
    float score = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        uint idx = (uint)indices[t * n_blocks + b];
        score += tg_qdots[b * n_centroids + idx];
    }

    // Online softmax update
    float m_new = (score > m_local) ? score : m_local;
    float correction = exp(m_local - m_new);
    float exp_score = exp(score - m_new);

    d_local = d_local * correction + exp_score;
    acc = acc * correction + exp_score * values[t * head_dim + dim_id];
    m_local = m_new;
}

// ---- Write output (each thread writes its own dimension) ----
if (d_local > 0.0f) {
    out[qh * head_dim + dim_id] = acc / d_local;
} else {
    out[qh * head_dim + dim_id] = 0.0f;
}
"""

_fused_attention_kernel = mx.fast.metal_kernel(
    name="sunshape_fused_attention",
    input_names=["qdots", "indices", "values", "params"],
    output_names=["out"],
    source=_FUSED_ATTENTION_SOURCE,
)


def fused_attention_metal(
    qdots: mx.array,
    indices: mx.array,
    values: mx.array,
    n_qh: int,
    T_kv: int,
    n_blocks: int,
    n_centroids: int,
    head_dim: int,
) -> mx.array | None:
    """Fused online-softmax block-VQ attention — single Metal kernel.

    Fuses gather + reduce + softmax + value accumulation into a single pass.
    No intermediate scores tensor is materialized.

    Optimized design:
    - head_dim threads per query head (thread-per-dimension layout)
    - qdots cached in threadgroup shared memory
    - Each thread accumulates a single output dimension (no value loop)
    - No SIMD reduction needed

    Supports head_dim up to 256 and n_blocks * n_centroids <= 4096.

    Returns None if Metal kernels are unavailable or dimensions exceed limits.
    """
    if T_kv == 0:
        return mx.zeros((n_qh, head_dim), dtype=mx.float32)
    if head_dim > 256:
        return None
    if n_blocks * n_centroids > _MAX_QDOTS_TG:
        return None

    try:
        mx.eval(qdots, indices, values)

        params = mx.array(
            [float(T_kv), float(n_blocks), float(n_centroids), float(head_dim)],
            dtype=mx.float32,
        )

        # head_dim threads per threadgroup (1 thread per output dimension)
        grid = (n_qh * head_dim, 1, 1)
        threadgroup = (head_dim, 1, 1)

        output = _fused_attention_kernel(
            inputs=[qdots, indices, values, params],
            output_shapes=[(n_qh, head_dim)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup,
            init_value=0.0,
        )
        out = output[0] if isinstance(output, (list, tuple)) else output
        mx.eval(out)
        _record_kernel_dispatch("sunshape_fused_attention_metal")
        return out
    except Exception:
        return None


# ================================================================== #
#  Metal kernel: Fused online-softmax with causal mask                #
# ================================================================== #
# Same optimized design as above but respects a causal mask:
#   score is masked to -inf for positions where kv_pos > q_pos.
# q_offset = T_kv - T_q (so query i attends to keys 0..q_offset+i)
#
# Thread-per-dimension layout with qdots in threadgroup memory.

_FUSED_ATTENTION_CAUSAL_SOURCE = r"""
// Fused online-softmax block-VQ attention with causal masking (optimized)
// params: [T_kv, n_blocks, n_centroids, head_dim, q_offset]
// head_dim threads per query head.  Thread-per-dimension layout.
// qdots cached in threadgroup memory.  No SIMD reduction.

uint head_dim = (uint)params[3];
uint qh = thread_position_in_grid.x / head_dim;
uint dim_id = thread_position_in_threadgroup.x;

uint T_kv       = (uint)params[0];
uint n_blocks   = (uint)params[1];
uint n_centroids = (uint)params[2];
uint q_offset   = (uint)params[4];
uint n_qh = qdots_shape[0];

if (qh >= n_qh || dim_id >= head_dim) return;

// q_pos for this query head
uint q_pos = q_offset + qh;

// ---- Cache qdots[qh, :, :] in threadgroup memory ----
threadgroup float tg_qdots[4096];  // 16 * 256 max

uint qdots_size = n_blocks * n_centroids;
uint qdots_base = qh * qdots_size;
for (uint i = dim_id; i < qdots_size; i += head_dim) {
    tg_qdots[i] = qdots[qdots_base + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- Per-thread online softmax state (single dimension) ----
float m_local = -1e30f;
float d_local = 0.0f;
float acc = 0.0f;

// ---- Process tokens up to causal bound ----
for (uint t = 0; t < T_kv; t++) {
    // Causal mask: skip if key position > query position
    if (t > q_pos) continue;

    // Gather score from shared threadgroup memory
    float score = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        uint idx = (uint)indices[t * n_blocks + b];
        score += tg_qdots[b * n_centroids + idx];
    }

    // Online softmax update
    float m_new = (score > m_local) ? score : m_local;
    float correction = exp(m_local - m_new);
    float exp_score = exp(score - m_new);

    d_local = d_local * correction + exp_score;
    acc = acc * correction + exp_score * values[t * head_dim + dim_id];
    m_local = m_new;
}

// ---- Write output (each thread writes its own dimension) ----
if (d_local > 0.0f) {
    out[qh * head_dim + dim_id] = acc / d_local;
} else {
    out[qh * head_dim + dim_id] = 0.0f;
}
"""

_fused_attention_causal_kernel = mx.fast.metal_kernel(
    name="sunshape_fused_attention_causal",
    input_names=["qdots", "indices", "values", "params"],
    output_names=["out"],
    source=_FUSED_ATTENTION_CAUSAL_SOURCE,
)


def fused_attention_causal_metal(
    qdots: mx.array,
    indices: mx.array,
    values: mx.array,
    n_qh: int,
    T_kv: int,
    n_blocks: int,
    n_centroids: int,
    head_dim: int,
    q_offset: int = 0,
) -> mx.array | None:
    """Fused online-softmax block-VQ attention with causal mask.

    Same as fused_attention_metal but applies a causal mask:
    query at position q_offset + qh can only attend to keys at positions <= q_offset + qh.

    Optimized design:
    - head_dim threads per query head (thread-per-dimension layout)
    - qdots cached in threadgroup shared memory
    - Each thread accumulates a single output dimension
    - No SIMD reduction needed

    Returns None if Metal kernels are unavailable or dimensions exceed limits.
    """
    if T_kv == 0:
        return mx.zeros((n_qh, head_dim), dtype=mx.float32)
    if head_dim > 256:
        return None
    if n_blocks * n_centroids > _MAX_QDOTS_TG:
        return None

    try:
        mx.eval(qdots, indices, values)

        params = mx.array(
            [float(T_kv), float(n_blocks), float(n_centroids),
             float(head_dim), float(q_offset)],
            dtype=mx.float32,
        )

        # head_dim threads per threadgroup (1 thread per output dimension)
        grid = (n_qh * head_dim, 1, 1)
        threadgroup = (head_dim, 1, 1)

        output = _fused_attention_causal_kernel(
            inputs=[qdots, indices, values, params],
            output_shapes=[(n_qh, head_dim)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup,
            init_value=0.0,
        )
        out = output[0] if isinstance(output, (list, tuple)) else output
        mx.eval(out)
        _record_kernel_dispatch("sunshape_fused_attention_causal_metal")
        return out
    except Exception:
        return None


# ================================================================== #
#  Metal kernel: Scalar quantization                                  #
# ================================================================== #

_SCALAR_QUANTIZE_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint num_boundaries = boundaries_shape[0];

    float val = values[elem];
    uint idx = 0;
    for (uint b = 0; b < num_boundaries; b++) {
        idx += (val > boundaries[b]) ? 1 : 0;
    }
    indices[elem] = static_cast<uint8_t>(idx);
"""

_scalar_quantize_kernel = mx.fast.metal_kernel(
    name="sunshape_scalar_quantize",
    input_names=["values", "boundaries"],
    output_names=["indices"],
    source=_SCALAR_QUANTIZE_SOURCE,
)


def quantize_scalar_to_indices(
    values: mx.array,
    boundaries: mx.array,
    *,
    prefer_metal: bool = True,
) -> tuple[mx.array, str]:
    """Quantize scalar values against sorted boundaries.

    Returns a tuple of ``(indices, backend)`` where backend is ``metal`` or
    ``mlx`` depending on which path actually executed.
    """
    if prefer_metal:
        try:
            flat = values.reshape(-1).astype(mx.float32)
            mx.eval(flat, boundaries)
            outputs = _scalar_quantize_kernel(
                inputs=[flat, boundaries.astype(mx.float32)],
                grid=(flat.size, 1, 1),
                threadgroup=(min(256, flat.size), 1, 1),
                output_shapes=[flat.shape],
                output_dtypes=[mx.uint8],
            )
            out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            out = out.reshape(values.shape)
            mx.eval(out)
            _record_kernel_dispatch("sunshape_scalar_quantize_metal")
            return out, "metal"
        except Exception:
            pass

    result = mx.zeros(values.shape, dtype=mx.uint8)
    boundaries = boundaries.astype(mx.float32)
    for i in range(boundaries.size):
        result = result + (values > boundaries[i]).astype(mx.uint8)
    return result, "mlx"

# ================================================================== #
#  Pure-MLX fused block-VQ attention                                   #
# ================================================================== #


def fused_block_vq_attention(
    qdots: mx.array,
    indices: mx.array,
    values: mx.array,
    T_q: int,
    T_kv: int,
    n_blocks: int,
    n_centroids: int,
    head_dim: int,
) -> mx.array:
    """Fused block-VQ attention — pure-MLX gather path.

    Computes attention using precomputed query-centroid block dot products
    and gathered indices, with softmax and value accumulation.

    Parameters
    ----------
    qdots : mx.array, shape (T_q, n_blocks, n_centroids)
    indices : mx.array, shape (T_kv, n_blocks)
    values : mx.array, shape (T_kv, head_dim)
    T_q, T_kv, n_blocks, n_centroids, head_dim : int

    Returns
    -------
    output : mx.array, shape (T_q, head_dim)
    """
    indices_int = indices.astype(mx.int32)

    # Gather: score[q, t] = sum_b qdots[q, b, indices[t, b]]
    qdots_exp = qdots[None, :, :, :]           # (1, T_q, n_blocks, n_centroids)
    indices_exp = indices_int[:, None, :, None]  # (T_kv, 1, n_blocks, 1)

    gathered = mx.take_along_axis(qdots_exp, indices_exp, axis=-1)
    gathered = gathered.squeeze(-1)     # (T_kv, T_q, n_blocks)
    scores = mx.sum(gathered, axis=2)   # (T_kv, T_q)
    scores = scores.T                   # (T_q, T_kv)

    # Causal mask
    if T_kv >= T_q:
        q_pos = mx.arange(T_kv - T_q, T_kv - T_q + T_q)
        k_pos = mx.arange(T_kv)
        causal = q_pos[:, None] >= k_pos[None, :]
        scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)

    # Softmax
    weights = mx.softmax(scores, axis=-1, precise=True)

    # Value output
    output = weights @ values.astype(mx.float32)
    return output


# ================================================================== #
#  Metal kernel: Fused attention + in-register value dequantization   #
#  (Eliminates the dequantization wall)                               #
# ================================================================== #
#
# Instead of pre-dequantizing the entire value cache to float32 before
# kernel launch, this kernel reads:
#   - packed uint32 words (containing 2/4-bit quantized values)
#   - per-group scales and zeros
# and dequantizes each value on-the-fly in registers.
#
# Memory bandwidth savings (per token per threadgroup):
#   4-bit: 72 bytes vs 512 bytes (7.1× reduction)
#   2-bit: 40 bytes vs 512 bytes (12.8× reduction)
#
# Uses the same thread-per-dimension layout as the optimized attention
# kernel: each thread owns exactly one output dimension, so it knows
# exactly which packed word, sub-value, group scale and zero to read.
#
# Grid: (n_qh * head_dim, 1, 1)
# Threadgroup: (head_dim, 1, 1)

_FUSED_ATTENTION_DEQUANT_SOURCE = r"""
// -------------------------------------------------------------------
// Fused attention + in-register value dequantization kernel
// -------------------------------------------------------------------
// Inputs:
//   qdots:         (n_qh, n_blocks, n_centroids) float32
//   indices:       (T_kv, n_blocks)              uint8/uint16  — key VQ indices
//   packed_values: (T_kv, packed_dim)            uint32        — bit-packed values
//   scales:        (T_kv, n_groups)              float32
//   zeros:         (T_kv, n_groups)              float32
//   params:        (8,)                          float32
//            [0]=T_kv, [1]=n_blocks, [2]=n_centroids, [3]=head_dim,
//            [4]=bits, [5]=vals_per_word, [6]=bit_mask, [7]=group_size
// Output:
//   out:           (n_qh, head_dim) float32
// -------------------------------------------------------------------

uint head_dim      = (uint)params[3];
uint qh            = thread_position_in_grid.x / head_dim;
uint dim_id        = thread_position_in_threadgroup.x;

uint T_kv          = (uint)params[0];
uint n_blocks      = (uint)params[1];
uint n_centroids   = (uint)params[2];
uint bits          = (uint)params[4];
uint vals_per_word = (uint)params[5];
uint bit_mask      = (uint)params[6];
uint group_size    = (uint)params[7];
uint n_qh          = qdots_shape[0];

if (qh >= n_qh || dim_id >= head_dim) return;

// Precompute per-thread dequant addressing (constant across all tokens)
uint packed_dim = (head_dim + vals_per_word - 1) / vals_per_word;
uint n_groups   = head_dim / group_size;
uint word_idx   = dim_id / vals_per_word;
uint sub_idx    = dim_id % vals_per_word;
uint shift      = sub_idx * bits;
uint group_idx  = dim_id / group_size;

// ---- Cache qdots[qh, :, :] in threadgroup memory ----
threadgroup float tg_qdots[4096];  // 16 * 256 max

uint qdots_size = n_blocks * n_centroids;
uint qdots_base = qh * qdots_size;
for (uint i = dim_id; i < qdots_size; i += head_dim) {
    tg_qdots[i] = qdots[qdots_base + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- Per-thread online softmax state (single dimension) ----
float m_local = -1e30f;
float d_local = 0.0f;
float acc     = 0.0f;

// ---- Process ALL tokens ----
for (uint t = 0; t < T_kv; t++) {
    // Gather score from shared threadgroup memory
    float score = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        uint idx = (uint)indices[t * n_blocks + b];
        score += tg_qdots[b * n_centroids + idx];
    }

    // Online softmax update
    float m_new = (score > m_local) ? score : m_local;
    float correction = exp(m_local - m_new);
    float exp_score = exp(score - m_new);

    // ---- In-register value dequantization ----
    uint word = packed_values[t * packed_dim + word_idx];
    uint qval = (word >> shift) & bit_mask;
    float scale_val = scales[t * n_groups + group_idx];
    float zero_val  = zeros[t * n_groups + group_idx];
    float val = (float)qval * scale_val + zero_val;

    d_local = d_local * correction + exp_score;
    acc     = acc * correction + exp_score * val;
    m_local = m_new;
}

// ---- Write output ----
if (d_local > 0.0f) {
    out[qh * head_dim + dim_id] = acc / d_local;
} else {
    out[qh * head_dim + dim_id] = 0.0f;
}
"""

_fused_attention_dequant_kernel = mx.fast.metal_kernel(
    name="sunshape_fused_attention_dequant",
    input_names=["qdots", "indices", "packed_values", "scales", "zeros", "params"],
    output_names=["out"],
    source=_FUSED_ATTENTION_DEQUANT_SOURCE,
)


def fused_attention_dequant_metal(
    qdots: mx.array,
    indices: mx.array,
    packed_values: mx.array,
    scales: mx.array,
    zeros: mx.array,
    n_qh: int,
    T_kv: int,
    n_blocks: int,
    n_centroids: int,
    head_dim: int,
    bits: int,
    group_size: int,
) -> mx.array | None:
    """Fused attention + in-register value dequantization — single Metal kernel.

    Reads packed quantized values directly and dequantizes per-thread in
    registers, completely bypassing the dequantization wall.  No intermediate
    float32 value tensor is materialized.

    Parameters
    ----------
    qdots : (n_qh, n_blocks, n_centroids) float32
    indices : (T_kv, n_blocks) uint8/uint16 — key VQ indices
    packed_values : (T_kv, packed_dim) uint32 — bit-packed quantized values
    scales : (T_kv, n_groups) float32 — per-group scale
    zeros : (T_kv, n_groups) float32 — per-group zero-point
    bits : 2 or 4
    group_size : int

    Returns None if Metal kernels are unavailable or dimensions exceed limits.
    """
    if T_kv == 0:
        return mx.zeros((n_qh, head_dim), dtype=mx.float32)
    if head_dim > 256:
        return None
    if n_blocks * n_centroids > _MAX_QDOTS_TG:
        return None

    vals_per_word = 32 // bits  # 16 for 2-bit, 8 for 4-bit
    bit_mask = (1 << bits) - 1  # 0x3 for 2-bit, 0xF for 4-bit

    try:
        mx.eval(qdots, indices, packed_values, scales, zeros)

        params = mx.array(
            [float(T_kv), float(n_blocks), float(n_centroids), float(head_dim),
             float(bits), float(vals_per_word), float(bit_mask), float(group_size)],
            dtype=mx.float32,
        )

        grid = (n_qh * head_dim, 1, 1)
        threadgroup = (head_dim, 1, 1)

        output = _fused_attention_dequant_kernel(
            inputs=[qdots, indices, packed_values, scales, zeros, params],
            output_shapes=[(n_qh, head_dim)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup,
            init_value=0.0,
        )
        out = output[0] if isinstance(output, (list, tuple)) else output
        mx.eval(out)
        _record_kernel_dispatch("sunshape_fused_attention_dequant_metal")
        return out
    except Exception:
        return None


# ================================================================== #
#  Metal kernel: Fused attention + value dequant with causal mask     #
# ================================================================== #

_FUSED_ATTENTION_DEQUANT_CAUSAL_SOURCE = r"""
// Fused attention + in-register value dequantization with causal masking
// params: [T_kv, n_blocks, n_centroids, head_dim, bits,
//          vals_per_word, bit_mask, group_size, q_offset]

uint head_dim      = (uint)params[3];
uint qh            = thread_position_in_grid.x / head_dim;
uint dim_id        = thread_position_in_threadgroup.x;

uint T_kv          = (uint)params[0];
uint n_blocks      = (uint)params[1];
uint n_centroids   = (uint)params[2];
uint bits          = (uint)params[4];
uint vals_per_word = (uint)params[5];
uint bit_mask      = (uint)params[6];
uint group_size    = (uint)params[7];
uint q_offset      = (uint)params[8];
uint n_qh          = qdots_shape[0];

if (qh >= n_qh || dim_id >= head_dim) return;

uint q_pos = q_offset + qh;

// Precompute per-thread dequant addressing
uint packed_dim = (head_dim + vals_per_word - 1) / vals_per_word;
uint n_groups   = head_dim / group_size;
uint word_idx   = dim_id / vals_per_word;
uint sub_idx    = dim_id % vals_per_word;
uint shift      = sub_idx * bits;
uint group_idx  = dim_id / group_size;

// ---- Cache qdots in threadgroup memory ----
threadgroup float tg_qdots[4096];

uint qdots_size = n_blocks * n_centroids;
uint qdots_base = qh * qdots_size;
for (uint i = dim_id; i < qdots_size; i += head_dim) {
    tg_qdots[i] = qdots[qdots_base + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- Per-thread online softmax state ----
float m_local = -1e30f;
float d_local = 0.0f;
float acc     = 0.0f;

// ---- Process tokens up to causal bound ----
for (uint t = 0; t < T_kv; t++) {
    if (t > q_pos) continue;

    float score = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        uint idx = (uint)indices[t * n_blocks + b];
        score += tg_qdots[b * n_centroids + idx];
    }

    float m_new = (score > m_local) ? score : m_local;
    float correction = exp(m_local - m_new);
    float exp_score = exp(score - m_new);

    // ---- In-register value dequantization ----
    uint word = packed_values[t * packed_dim + word_idx];
    uint qval = (word >> shift) & bit_mask;
    float scale_val = scales[t * n_groups + group_idx];
    float zero_val  = zeros[t * n_groups + group_idx];
    float val = (float)qval * scale_val + zero_val;

    d_local = d_local * correction + exp_score;
    acc     = acc * correction + exp_score * val;
    m_local = m_new;
}

if (d_local > 0.0f) {
    out[qh * head_dim + dim_id] = acc / d_local;
} else {
    out[qh * head_dim + dim_id] = 0.0f;
}
"""

_fused_attention_dequant_causal_kernel = mx.fast.metal_kernel(
    name="sunshape_fused_attention_dequant_causal",
    input_names=["qdots", "indices", "packed_values", "scales", "zeros", "params"],
    output_names=["out"],
    source=_FUSED_ATTENTION_DEQUANT_CAUSAL_SOURCE,
)


def fused_attention_dequant_causal_metal(
    qdots: mx.array,
    indices: mx.array,
    packed_values: mx.array,
    scales: mx.array,
    zeros: mx.array,
    n_qh: int,
    T_kv: int,
    n_blocks: int,
    n_centroids: int,
    head_dim: int,
    bits: int,
    group_size: int,
    q_offset: int = 0,
) -> mx.array | None:
    """Fused attention + value dequant with causal mask — single Metal kernel.

    Same as fused_attention_dequant_metal but with causal masking.

    Returns None if Metal kernels are unavailable or dimensions exceed limits.
    """
    if T_kv == 0:
        return mx.zeros((n_qh, head_dim), dtype=mx.float32)
    if head_dim > 256:
        return None
    if n_blocks * n_centroids > _MAX_QDOTS_TG:
        return None

    vals_per_word = 32 // bits
    bit_mask = (1 << bits) - 1

    try:
        mx.eval(qdots, indices, packed_values, scales, zeros)

        params = mx.array(
            [float(T_kv), float(n_blocks), float(n_centroids), float(head_dim),
             float(bits), float(vals_per_word), float(bit_mask), float(group_size),
             float(q_offset)],
            dtype=mx.float32,
        )

        grid = (n_qh * head_dim, 1, 1)
        threadgroup = (head_dim, 1, 1)

        output = _fused_attention_dequant_causal_kernel(
            inputs=[qdots, indices, packed_values, scales, zeros, params],
            output_shapes=[(n_qh, head_dim)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup,
            init_value=0.0,
        )
        out = output[0] if isinstance(output, (list, tuple)) else output
        mx.eval(out)
        _record_kernel_dispatch("sunshape_fused_attention_dequant_causal_metal")
        return out
    except Exception:
        return None


# ================================================================== #
#  Metal kernel: Fused value dequantization                           #
# ================================================================== #
# Adapted from turboQuantPlayground's value_dequant.py.
# Performs per-group asymmetric dequantization in a single pass.
#
# NOTE: This kernel source is compatible with mx.fast.metal_kernel()
# (body-only, no function declaration).  It is provided for future use
# when Metal-accelerated value dequantization is needed.  Currently,
# dequantize_values() uses the pure-MLX path below.

_VALUE_DEQUANT_SOURCE = r"""
// Fused value dequantization kernel.
//
// Grid: (D, N_BATCH, 1)  — one thread per (batch, coordinate) pair.
// Inputs: packed (uint8), scales (float32), zeros (float32)
// Params: D, N_BATCH, BITS, VALS_PER_BYTE, BIT_MASK, GROUP_SIZE, N_GROUPS, PACKED_D
// Output: out (float32)

uint coord = thread_position_in_grid.x;
uint batch_idx = thread_position_in_grid.y;

uint D_val = params_shape[0];         // D passed as a (D,) dummy param
uint N_BATCH = packed_shape[0];

if (batch_idx >= N_BATCH || coord >= D_val) return;

// Unpack constants from the params array
uint BITS = (uint)params[0];
uint VALS_PER_BYTE = (uint)params[1];
uint BIT_MASK = (uint)params[2];
uint GROUP_SIZE = (uint)params[3];
uint N_GROUPS = (uint)params[4];
uint PACKED_D = (uint)params[5];

uint byte_idx = coord / VALS_PER_BYTE;
uint sub = coord % VALS_PER_BYTE;

uint8_t packed_byte = packed[batch_idx * PACKED_D + byte_idx];
uint qval = ((uint)packed_byte >> (sub * BITS)) & BIT_MASK;

uint group_idx = coord / GROUP_SIZE;
float scale_val = (float)scales[batch_idx * N_GROUPS + group_idx];
float zero_val = (float)zeros[batch_idx * N_GROUPS + group_idx];
float result = (float)qval * scale_val + zero_val;

out[batch_idx * D_val + coord] = result;
"""


# ================================================================== #
#  Metal kernel: Vectorized block-VQ quantize                         #
# ================================================================== #
# Replaces the Python for-loop in codec.quantize().
# Each thread handles one (token, block) pair: finds the nearest centroid.
#
# Grid: (N_tokens, n_blocks, 1)
# Inputs:
#   keys_transformed: (N, head_dim)  — already in codec space
#   centroids:        (n_blocks, n_centroids, block_dim)
#   params:           (3,) = [block_dim, n_centroids, head_dim]
# Output:
#   out_indices: (N, n_blocks) uint8

_BLOCK_VQ_QUANTIZE_SOURCE = r"""
uint token = thread_position_in_grid.x;
uint blk = thread_position_in_grid.y;

uint N_tokens = keys_transformed_shape[0];
uint n_blocks_val = centroids_shape[0];
uint n_centroids_val = centroids_shape[1];
uint block_dim_val = centroids_shape[2];

if (token >= N_tokens || blk >= n_blocks_val) return;

// Compute offset into key vector for this block
uint key_offset = token * (uint)params[2] + blk * block_dim_val;

// Find nearest centroid (L2 distance)
float best_dist = 1e30f;
uint best_idx = 0;

for (uint c = 0; c < n_centroids_val; c++) {
    float dist = 0.0f;
    uint centroid_offset = (blk * n_centroids_val + c) * block_dim_val;
    for (uint d = 0; d < block_dim_val; d++) {
        float diff = keys_transformed[key_offset + d] - centroids[centroid_offset + d];
        dist += diff * diff;
    }
    if (dist < best_dist) {
        best_dist = dist;
        best_idx = c;
    }
}

out_indices[token * n_blocks_val + blk] = (uint8_t)best_idx;
"""

_block_vq_quantize_kernel = mx.fast.metal_kernel(
    name="sunshape_block_vq_quantize",
    input_names=["keys_transformed", "centroids", "params"],
    output_names=["out_indices"],
    source=_BLOCK_VQ_QUANTIZE_SOURCE,
)


def block_vq_quantize_metal(
    keys_transformed: mx.array,
    centroids: mx.array,
    n_blocks: int,
    n_centroids: int,
    block_dim: int,
    head_dim: int,
) -> mx.array | None:
    """Vectorized block-VQ quantize — single Metal kernel.

    Replaces the Python for-loop in codec.quantize() when no mixed-precision
    or local-metric transforms are needed (identity E).

    Parameters
    ----------
    keys_transformed : (N, head_dim) float32 — keys already in codec space
    centroids : (n_blocks, n_centroids, block_dim) float32
    n_blocks, n_centroids, block_dim, head_dim : int

    Returns None if Metal kernels are unavailable.
    """
    try:
        N = keys_transformed.shape[0]
        mx.eval(keys_transformed, centroids)

        params = mx.array(
            [float(block_dim), float(n_centroids), float(head_dim)],
            dtype=mx.float32,
        )

        grid = (N, n_blocks, 1)
        threadgroup = (min(256, N), 1, 1)

        outputs = _block_vq_quantize_kernel(
            inputs=[keys_transformed, centroids, params],
            output_shapes=[(N, n_blocks)],
            output_dtypes=[mx.uint8],
            grid=grid,
            threadgroup=threadgroup,
        )
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        mx.eval(out)
        _record_kernel_dispatch("sunshape_block_vq_quantize_metal")
        return out
    except Exception:
        return None


# ================================================================== #
#  Index packing utilities (for compact storage)                       #
# ================================================================== #
# Uses native MLX bitwise ops (mx.left_shift, mx.right_shift, mx.bitwise_and)
# as confirmed available in turboQuantPlayground's mlx_backend.py.

# Pre-computed shift constants as MLX uint32 arrays
_SHIFTS_2BIT = mx.array([i * 2 for i in range(16)], dtype=mx.uint32)
_SHIFTS_3BIT = mx.array([i * 3 for i in range(10)], dtype=mx.uint32)
_SHIFTS_4BIT = mx.array([i * 4 for i in range(8)], dtype=mx.uint32)

# Bit masks as MLX uint32
_MASK_2BIT = mx.array(0x3, dtype=mx.uint32)
_MASK_3BIT = mx.array(0x7, dtype=mx.uint32)
_MASK_4BIT = mx.array(0xF, dtype=mx.uint32)


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack low-bit indices into uint32 words using native MLX bitwise ops.

    Args:
        indices: (..., D) uint8 indices
        bits: 2, 3, or 4 bits per index

    Returns:
        packed: (..., D // els_per_word) uint32
    """
    if bits == 2:
        D = indices.shape[-1]
        if D % 16 != 0:
            raise ValueError(f"Last dim must be divisible by 16 for 2-bit, got {D}")
        reshaped = indices.reshape(*indices.shape[:-1], D // 16, 16).astype(mx.uint32)
        shifted = mx.left_shift(mx.bitwise_and(reshaped, _MASK_2BIT), _SHIFTS_2BIT)
        return mx.sum(shifted, axis=-1).astype(mx.uint32)
    elif bits == 3:
        D = indices.shape[-1]
        num_words = (D + 9) // 10
        pad_needed = num_words * 10 - D
        if pad_needed > 0:
            pads = [(0, 0)] * (indices.ndim - 1) + [(0, pad_needed)]
            indices = mx.pad(indices, pads, constant_values=0)
        reshaped = indices.reshape(*indices.shape[:-1], num_words, 10).astype(mx.uint32)
        shifted = mx.left_shift(mx.bitwise_and(reshaped, _MASK_3BIT), _SHIFTS_3BIT)
        return mx.sum(shifted, axis=-1).astype(mx.uint32)
    elif bits == 4:
        D = indices.shape[-1]
        if D % 8 != 0:
            raise ValueError(f"Last dim must be divisible by 8 for 4-bit, got {D}")
        reshaped = indices.reshape(*indices.shape[:-1], D // 8, 8).astype(mx.uint32)
        shifted = mx.left_shift(mx.bitwise_and(reshaped, _MASK_4BIT), _SHIFTS_4BIT)
        return mx.sum(shifted, axis=-1).astype(mx.uint32)
    else:
        raise ValueError(f"Unsupported bits: {bits}")


def unpack_indices(packed: mx.array, D: int, bits: int) -> mx.array:
    """Unpack uint32 words back to low-bit indices using native MLX bitwise ops.

    Args:
        packed: (..., D_packed) uint32
        D: original dimension
        bits: 2, 3, or 4

    Returns:
        indices: (..., D) uint32
    """
    if bits == 2:
        expanded = mx.bitwise_and(mx.right_shift(packed[..., None], _SHIFTS_2BIT), _MASK_2BIT)
        return expanded.reshape(*packed.shape[:-1], D)
    elif bits == 3:
        expanded = mx.bitwise_and(mx.right_shift(packed[..., None], _SHIFTS_3BIT), _MASK_3BIT)
        total = packed.shape[-1] * 10
        flat = expanded.reshape(*packed.shape[:-1], total)
        return flat[..., :D]
    elif bits == 4:
        expanded = mx.bitwise_and(mx.right_shift(packed[..., None], _SHIFTS_4BIT), _MASK_4BIT)
        return expanded.reshape(*packed.shape[:-1], D)
    else:
        raise ValueError(f"Unsupported bits: {bits}")


# ================================================================== #
#  Value quantization (asymmetric group quantization)                 #
# ================================================================== #
# Adapted from turboQuantPlayground's kv_cache.py value quantization.
# Supports 2-bit and 4-bit asymmetric per-group quantization.


def quantize_values(
    values: mx.array,
    bits: int = 4,
    group_size: int = 64,
) -> tuple[mx.array, mx.array, mx.array, int]:
    """Quantize values using asymmetric per-group min-max quantization.

    Parameters
    ----------
    values : mx.array, shape (T, head_dim)
        Value vectors to quantize.
    bits : int
        Bits per value (2 or 4).
    group_size : int
        Number of consecutive dimensions per quantization group.

    Returns
    -------
    packed : mx.array, uint32
        Bit-packed quantized values (packed via ``pack_indices``).
    scales : mx.array, float32, shape (T, n_groups)
        Per-group scale factors.
    zeros : mx.array, float32, shape (T, n_groups)
        Per-group zero-points.
    n_groups : int
        Number of groups.
    """
    T, D = values.shape
    n_groups = D // group_size
    max_val = 2**bits - 1

    # Reshape into groups
    v_grouped = values.reshape(T, n_groups, group_size).astype(mx.float32)

    # Per-group min and max
    g_min = mx.min(v_grouped, axis=2, keepdims=True)   # (T, n_groups, 1)
    g_max = mx.max(v_grouped, axis=2, keepdims=True)   # (T, n_groups, 1)

    # Scale and zero-point
    scales = (g_max - g_min) / max_val                  # (T, n_groups, 1)
    scales = mx.where(scales < 1e-8, mx.array(1e-8), scales)
    zeros = g_min                                       # (T, n_groups, 1)

    # Quantize: q = clamp(round((v - zero) / scale), 0, max_val)
    q_vals = mx.round((v_grouped - zeros) / scales).astype(mx.int32)
    q_vals = mx.clip(q_vals, 0, max_val).astype(mx.uint8)

    # Pack into bytes
    packed = pack_indices(q_vals.reshape(T, D), bits=bits)

    return (
        packed,
        scales.squeeze(2),   # (T, n_groups)
        zeros.squeeze(2),    # (T, n_groups)
        n_groups,
    )


def dequantize_values(
    packed: mx.array,
    scales: mx.array,
    zeros: mx.array,
    D: int,
    bits: int = 4,
    group_size: int = 64,
) -> mx.array:
    """Dequantize values from packed asymmetric group quantization.

    Parameters
    ----------
    packed : mx.array, uint8
        Bit-packed quantized values.
    scales : mx.array, float32, shape (..., n_groups)
    zeros : mx.array, float32, shape (..., n_groups)
    D : int
        Original head dimension.
    bits : int
        Bits per value.
    group_size : int
        Number of dimensions per group.

    Returns
    -------
    values : mx.array, float32, shape (..., D)
    """
    n_groups = D // group_size
    q_vals = unpack_indices(packed, D=D, bits=bits).astype(mx.float32)

    # Reshape to groups
    q_grouped = q_vals.reshape(*q_vals.shape[:-1], n_groups, group_size)

    # Dequantize: v = q * scale + zero
    scales_exp = scales[..., None]   # (..., n_groups, 1)
    zeros_exp = zeros[..., None]     # (..., n_groups, 1)
    v_grouped = q_grouped * scales_exp + zeros_exp

    return v_grouped.reshape(*q_vals.shape[:-1], D)
