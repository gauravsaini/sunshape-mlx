"""SunShape KV Cache — MLX drop-in for mlx-lm's KVCache.

Provides two cache variants:

- ``SunShapeKVCache``: Simple quantized KV cache (keys as block-VQ indices,
  values in FP16).
- ``HybridSunShapeKVCache``: Hybrid buffer/quantized cache (adapted from
  turboQuantPlayground).  Recent tokens stay in a full-precision buffer;
  older tokens are flushed to compressed storage.  This preserves quality
  for the most recent context while still achieving high compression.

Both caches pre-allocate buffers in steps of 256 tokens and support
the standard mlx-lm KVCache interface.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np

from sunshape_mlx.codec import SunShapeBlockCodec, SunShapeQuantized
from sunshape_mlx.value_codecs import (
    GroupedValueCodec,
    TurboQuantValueCodec,
    build_value_codec,
)


# Pre-allocation step size (matches mlx-lm and turboquant-mlx)
_STEP = 256

# Default buffer size for hybrid cache (tokens kept unquantized)
_DEFAULT_BUFFER_SIZE = 128


def _packed_dim(head_dim: int, bits: int) -> int:
    """Compute packed dimension (number of uint32 words) for given head_dim and bits.

    Matches the packing scheme in ``kernels.pack_indices``:
      2-bit → D // 16,  3-bit → (D+9)//10,  4-bit → D // 8
    """
    if bits == 2:
        if head_dim % 16 != 0:
            raise ValueError(f"head_dim must be divisible by 16 for 2-bit packing, got {head_dim}")
        return head_dim // 16
    elif bits == 3:
        return (head_dim + 9) // 10
    elif bits == 4:
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be divisible by 8 for 4-bit packing, got {head_dim}")
        return head_dim // 8
    else:
        raise ValueError(f"Unsupported bits: {bits} (must be 2, 3, or 4)")


class SunShapeKVCache:
    """Quantized KV cache for mlx-lm using SunShape block VQ.

    Drop-in replacement for ``mlx_lm.models.base.KVCache``.  When
    ``update_and_fetch`` is called, keys are quantized by the codec
    and stored as centroid indices.  Values are stored in FP16/BF16.

    The patched SDPA function in ``sunshape_mlx.attention`` reads the
    quantized state directly and computes scores via precomputed
    query-centroid dots, avoiding full key dequantization.
    """

    def __init__(
        self,
        codec: SunShapeBlockCodec,
        *,
        n_kv_heads: int = 1,
        head_dim: int | None = None,
        dtype: mx.Dtype = mx.float16,
    ):
        self.codec = codec
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or codec.head_dim
        self.block_dim = codec.block_dim
        self.n_blocks = codec.n_blocks
        self.n_centroids = codec.n_centroids
        self.dtype = dtype
        self.offset = 0

        # Pre-allocated buffers (will be resized on first update)
        # MLX arrays are immutable, so we maintain numpy backing arrays
        # and convert to MLX after each mutation.
        self._key_indices: mx.array | None = None     # (B, n_kv_heads, max_seq, n_blocks)
        self._key_indices_np: np.ndarray | None = None  # numpy backing for mutations
        self._key_passthrough: mx.array | None = None  # for mixed-precision mode
        self._values: mx.array | None = None           # (B, n_kv_heads, max_seq, head_dim)
        self._values_np: np.ndarray | None = None      # numpy backing for mutations
        self._capacity = 0

    # ------------------------------------------------------------------ #
    #  Capacity management                                                 #
    # ------------------------------------------------------------------ #

    def _ensure_capacity(self, B: int, new_tokens: int) -> None:
        """Grow pre-allocated buffers if needed.

        MLX arrays are immutable, so we rebuild in numpy and convert.
        """
        needed = self.offset + new_tokens
        if needed <= self._capacity:
            return

        new_capacity = ((needed + _STEP - 1) // _STEP) * _STEP
        idx_dtype = np.uint8 if self.n_centroids <= 256 else np.uint16
        mx_idx_dtype = mx.uint8 if self.n_centroids <= 256 else mx.uint16

        # Key indices — build in numpy, convert to MLX
        if self._key_indices is None:
            self._key_indices_np = np.zeros(
                (B, self.n_kv_heads, new_capacity, self.n_blocks), dtype=idx_dtype,
            )
        elif self._key_indices.shape[2] < new_capacity:
            old_np = self._key_indices_np  # already current from update_and_fetch
            new_np = np.zeros(
                (B, self.n_kv_heads, new_capacity, self.n_blocks), dtype=idx_dtype,
            )
            if self.offset > 0:
                new_np[:, :, : self.offset, :] = old_np[:, :, : self.offset, :]
            self._key_indices_np = new_np
        # else: capacity sufficient, _key_indices_np already current
        self._key_indices = mx.array(self._key_indices_np, dtype=mx_idx_dtype)

        # Values — build in numpy, convert to MLX
        if self._values is None:
            self._values_np = np.zeros(
                (B, self.n_kv_heads, new_capacity, self.head_dim),
                dtype=np.float16,
            )
        elif self._values.shape[2] < new_capacity:
            old_np = self._values_np  # already current from update_and_fetch
            new_np = np.zeros(
                (B, self.n_kv_heads, new_capacity, self.head_dim),
                dtype=np.float16,
            )
            if self.offset > 0:
                new_np[:, :, : self.offset, :] = old_np[:, :, : self.offset, :]
            self._values_np = new_np
        # else: capacity sufficient, _values_np already current
        self._values = mx.array(self._values_np, dtype=self.dtype)

        self._capacity = new_capacity
        mx.eval(self._key_indices, self._values)

    # ------------------------------------------------------------------ #
    #  Update and fetch (main API)                                         #
    # ------------------------------------------------------------------ #

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize new keys, store indices, append values.

        Parameters
        ----------
        keys : mx.array, shape (B, n_kv_heads, T, head_dim)
        values : mx.array, shape (B, n_kv_heads, T, head_dim)

        Returns
        -------
        keys_out : mx.array — the keys as stored (may be dequantized for compatibility)
        values_out : mx.array — the values as stored
        """
        B, n_kv_heads, T, head_dim = keys.shape
        assert n_kv_heads == self.n_kv_heads, (
            f"Expected {self.n_kv_heads} kv heads, got {n_kv_heads}"
        )
        assert head_dim == self.head_dim, (
            f"Expected head_dim {self.head_dim}, got {head_dim}"
        )

        self._ensure_capacity(B, T)

        # Quantize keys: process per-batch per-head
        keys_float = keys.astype(mx.float32)
        idx_dtype = np.uint8 if self.n_centroids <= 256 else np.uint16
        indices_np = np.zeros((B, n_kv_heads, T, self.n_blocks), dtype=idx_dtype)

        for b in range(B):
            for h in range(n_kv_heads):
                flat_keys = keys_float[b, h]  # (T, head_dim)
                quantized = self.codec.quantize(flat_keys)
                idx_np = np.array(quantized.indices)
                indices_np[b, h, :T] = idx_np

        # Write indices into the numpy backing array
        self._key_indices_np[:, :, self.offset : self.offset + T, :] = indices_np
        self._key_indices = mx.array(self._key_indices_np)
        mx.eval(self._key_indices)

        # Write values into the numpy backing array
        values_np = np.array(values.astype(self.dtype))
        self._values_np[:, :, self.offset : self.offset + T, :] = values_np
        self._values = mx.array(self._values_np, dtype=self.dtype)
        mx.eval(self._values)

        self.offset += T

        # Return dequantized keys for compatibility (the patched SDPA bypasses this)
        return self.keys, self.values

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def keys(self) -> mx.array:
        """Dequantized keys up to current offset (for fallback paths)."""
        if self._key_indices is None or self.offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=mx.float32)
        indices = self._key_indices[:, :, : self.offset, :]
        # Dequantize for compatibility
        B, n_kv_heads, T, n_blocks = indices.shape
        # Use codec to dequantize per-batch per-head
        k_hat = mx.zeros((B, n_kv_heads, T, self.head_dim), dtype=mx.float32)
        indices_np = np.array(indices.astype(mx.int32))
        centroids_np = np.array(self.codec.centroids)
        inv_perm_np = np.array(self.codec.inv_permutation) if self.codec.mode != "rotated" else None
        rotation_np = np.array(self.codec.rotation) if self.codec.use_rotation else None

        for b in range(B):
            for h in range(n_kv_heads):
                # Reconstruct per token
                k_np = np.zeros((T, self.head_dim), dtype=np.float32)
                for blk in range(n_blocks):
                    sl = slice(blk * self.block_dim, (blk + 1) * self.block_dim)
                    k_np[:, sl] = centroids_np[blk][indices_np[b, h, :, blk]]
                # Inverse transform
                if self.codec.mode in {"profileperm_baseline", "profileperm_localmetric_dsq", "profileperm_mixed_precision"}:
                    k_np = k_np[:, inv_perm_np]
                elif self.codec.use_rotation:
                    k_np = k_np @ rotation_np
                k_hat[b, h] = mx.array(k_np, dtype=mx.float32)
        mx.eval(k_hat)
        return k_hat

    @property
    def values(self) -> mx.array:
        """Values up to current offset."""
        if self._values is None or self.offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=self.dtype)
        return self._values[:, :, : self.offset, :]

    @property
    def key_indices(self) -> mx.array:
        """Raw centroid indices for the patched SDPA fast path."""
        if self._key_indices is None or self.offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.n_blocks), dtype=mx.uint8)
        return self._key_indices[:, :, : self.offset, :]

    @property
    def state(self) -> tuple[mx.array, ...]:
        """Cache state for serialization."""
        return (
            self._key_indices[:, :, : self.offset, :] if self._key_indices is not None else mx.zeros((1,)),
            self._values[:, :, : self.offset, :] if self._values is not None else mx.zeros((1,)),
        )

    # ------------------------------------------------------------------ #
    #  Memory estimation                                                   #
    # ------------------------------------------------------------------ #

    @property
    def nbytes(self) -> int:
        """Compressed storage size in bytes."""
        if self._key_indices is None:
            return 0
        T = self.offset
        B = self._key_indices.shape[0]
        idx_bytes = B * self.n_kv_heads * T * self.n_blocks * (1 if self.n_centroids <= 256 else 2)
        val_bytes = B * self.n_kv_heads * T * self.head_dim * 2  # fp16
        return idx_bytes + val_bytes

    @property
    def nbytes_equivalent_fp16(self) -> int:
        """Equivalent FP16 storage size for compression ratio."""
        T = self.offset
        if T == 0:
            return 0
        B = self._key_indices.shape[0] if self._key_indices is not None else 1
        return 2 * B * self.n_kv_heads * T * self.head_dim * 2  # both K and V in fp16

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP16."""
        fp16 = self.nbytes_equivalent_fp16
        if fp16 == 0:
            return 1.0
        return fp16 / max(1, self.nbytes)

    # ------------------------------------------------------------------ #
    #  mlx-lm KVCache compatibility                                        #
    # ------------------------------------------------------------------ #

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> None:
        """Remove n tokens from the end of the cache."""
        self.offset = max(0, self.offset - n)

    def empty(self) -> bool:
        return self.offset == 0

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **kwargs) -> mx.array | str:
        """Create a causal mask (matches mlx-lm API).

        T = self.offset is the current KV length.  N new query tokens are
        appended at positions [T, T+N).  A causal mask ensures query at
        position q can only attend to key at position k where k <= q.
        """
        T = self.offset
        if T == 0 and N == 0:
            return "none"

        # Query positions: [T, T + N) — the newly added tokens
        # Key positions:   [0, T + N) — all keys seen so far
        q_pos = mx.arange(T, T + N)
        k_pos = mx.arange(T + N)
        causal = q_pos[:, None] >= k_pos[None, :]

        if return_array:
            return mx.where(causal, mx.array(0.0), mx.array(mx.finfo(mx.float16).min)).astype(mx.float16)
        return "causal"

    def __repr__(self) -> str:
        return (
            f"SunShapeKVCache(offset={self.offset}, head_dim={self.head_dim}, "
            f"n_blocks={self.n_blocks}, n_centroids={self.n_centroids}, "
            f"compression={self.compression_ratio:.1f}x)"
        )


# ==================================================================== #
#  Hybrid buffer / quantized cache                                     #
# ==================================================================== #
# Adapted from turboQuantPlayground's kv_cache.py.
# Keeps the most recent tokens in a full-precision buffer for quality,
# and flushes older tokens to compressed storage (block-VQ keys +
# optional value quantization).


class HybridSunShapeKVCache:
    """Hybrid buffer/quantized KV cache for mlx-lm.

    Adapted from turboQuantPlayground's ``TurboQuantKVCache``.

    Recent tokens (up to ``buffer_size``) are stored in full precision
    in a sliding window buffer.  When the buffer overflows, the oldest
    tokens are flushed to compressed storage:

    - **Keys**: quantized via the SunShape block-VQ codec (centroid indices)
    - **Values**: configurable backend:
      `fp16`, `grouped`, `turboquant`, or `sunshape`

    This hybrid approach preserves quality for the most recent context
    (which dominates attention scores) while still achieving high
    overall compression.
    """

    def __init__(
        self,
        codec: SunShapeBlockCodec,
        *,
        n_kv_heads: int = 1,
        head_dim: int | None = None,
        dtype: mx.Dtype = mx.float16,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        value_backend: str | None = None,
        value_bits: int | None = None,
        value_group_size: int = 64,
        value_seed: int = 43,
        value_codec: SunShapeBlockCodec | None = None,
        prefer_metal_kernels: bool = True,
    ):
        self.codec = codec
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or codec.head_dim
        self.block_dim = codec.block_dim
        self.n_blocks = codec.n_blocks
        self.n_centroids = codec.n_centroids
        self.dtype = dtype
        self.buffer_size = buffer_size
        self.value_backend = value_backend or ("fp16" if value_bits is None else "grouped")
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.value_seed = value_seed
        self.prefer_metal_kernels = prefer_metal_kernels
        self.value_codec = value_codec
        self.last_value_quant_backend = "none"

        if self.value_backend not in {"fp16", "grouped", "turboquant", "sunshape"}:
            raise ValueError(f"Unsupported value backend: {self.value_backend}")
        if self.value_backend != "fp16" and self.value_bits is None:
            raise ValueError(f"value_bits is required for value backend {self.value_backend}")
        if self.value_backend == "grouped" and self.head_dim % self.value_group_size != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be divisible by "
                f"value_group_size ({self.value_group_size}) for value quantization"
            )
        if self.value_backend == "sunshape":
            self.value_codec = self.value_codec or codec
            if self.value_codec.head_dim != self.head_dim:
                raise ValueError("SunShape value codec head_dim must match cache head_dim")
            self.value_n_blocks = self.value_codec.n_blocks
            self.value_n_centroids = self.value_codec.n_centroids
        else:
            self.value_n_blocks = 0
            self.value_n_centroids = 0

        self._value_codec_impl = None if self.value_backend == "sunshape" else build_value_codec(
            self.value_backend,
            head_dim=self.head_dim,
            value_bits=self.value_bits or 0,
            value_group_size=self.value_group_size,
            value_seed=self.value_seed,
            prefer_metal=self.prefer_metal_kernels,
        )

        # Compressed storage offset (tokens flushed from buffer)
        self._compressed_offset = 0
        # Buffer offset (tokens currently in buffer)
        self._buffer_offset = 0

        # Full-precision buffer for recent tokens
        self._key_buffer: mx.array | None = None    # (B, n_kv_heads, buffer_size, head_dim)
        self._key_buffer_np: np.ndarray | None = None
        self._value_buffer: mx.array | None = None  # (B, n_kv_heads, buffer_size, head_dim)
        self._value_buffer_np: np.ndarray | None = None

        # Compressed storage for flushed tokens
        self._key_indices: mx.array | None = None
        self._key_indices_np: np.ndarray | None = None
        self._values_packed: mx.array | None = None
        self._values_packed_np: np.ndarray | None = None
        self._value_scales: mx.array | None = None
        self._value_scales_np: np.ndarray | None = None
        self._value_zeros: mx.array | None = None
        self._value_zeros_np: np.ndarray | None = None
        self._value_norms: mx.array | None = None
        self._value_norms_np: np.ndarray | None = None
        self._value_indices: mx.array | None = None
        self._value_indices_np: np.ndarray | None = None
        self._value_n_groups: int = 0
        self._values_fp16: mx.array | None = None
        self._values_fp16_np: np.ndarray | None = None
        self._compressed_capacity = 0

    @property
    def offset(self) -> int:
        """Total number of tokens in the cache (compressed + buffer)."""
        return self._compressed_offset + self._buffer_offset

    # ------------------------------------------------------------------ #
    #  Internal: flush buffer to compressed storage                       #
    # ------------------------------------------------------------------ #

    def _flush_buffer(self, B: int, n_to_flush: int) -> None:
        """Move the oldest n_to_flush tokens from buffer to compressed storage."""
        if n_to_flush <= 0 or self._key_buffer is None:
            return

        n_to_flush = min(n_to_flush, self._buffer_offset)
        if n_to_flush == 0:
            return

        # Extract oldest tokens from buffer
        keys_flush = self._key_buffer[:, :, :n_to_flush, :]
        values_flush = self._value_buffer[:, :, :n_to_flush, :]

        # Quantize keys
        keys_float = keys_flush.astype(mx.float32)
        idx_dtype = np.uint8 if self.n_centroids <= 256 else np.uint16
        mx_idx_dtype = mx.uint8 if self.n_centroids <= 256 else mx.uint16
        indices_np = np.zeros((B, self.n_kv_heads, n_to_flush, self.n_blocks), dtype=idx_dtype)

        for b in range(B):
            for h in range(self.n_kv_heads):
                flat_keys = keys_float[b, h]
                quantized = self.codec.quantize(flat_keys)
                indices_np[b, h] = np.array(quantized.indices)

        # Grow compressed storage if needed
        needed = self._compressed_offset + n_to_flush
        if needed > self._compressed_capacity:
            new_cap = ((needed + _STEP - 1) // _STEP) * _STEP

            # Key indices
            if self._key_indices is None:
                self._key_indices_np = np.zeros(
                    (B, self.n_kv_heads, new_cap, self.n_blocks), dtype=idx_dtype,
                )
            else:
                old = self._key_indices_np
                new = np.zeros(
                    (B, self.n_kv_heads, new_cap, self.n_blocks), dtype=idx_dtype,
                )
                if self._compressed_offset > 0:
                    new[:, :, :self._compressed_offset, :] = old[:, :, :self._compressed_offset, :]
                self._key_indices_np = new
            self._key_indices = mx.array(self._key_indices_np, dtype=mx_idx_dtype)

            # Values
            if self.value_backend == "grouped":
                grouped_codec = self._value_codec_impl
                assert isinstance(grouped_codec, GroupedValueCodec)
                packed_D = grouped_codec.packed_dim
                n_groups = grouped_codec.n_groups
                if self._values_packed is None:
                    self._values_packed_np = np.zeros((B, self.n_kv_heads, new_cap, packed_D), dtype=np.uint32)
                    self._value_scales_np = np.zeros((B, self.n_kv_heads, new_cap, n_groups), dtype=np.float32)
                    self._value_zeros_np = np.zeros((B, self.n_kv_heads, new_cap, n_groups), dtype=np.float32)
                    self._value_n_groups = n_groups
                else:
                    old_packed = self._values_packed_np
                    old_scales = self._value_scales_np
                    old_zeros = self._value_zeros_np
                    self._values_packed_np = np.zeros((B, self.n_kv_heads, new_cap, packed_D), dtype=np.uint32)
                    self._value_scales_np = np.zeros((B, self.n_kv_heads, new_cap, n_groups), dtype=np.float32)
                    self._value_zeros_np = np.zeros((B, self.n_kv_heads, new_cap, n_groups), dtype=np.float32)
                    if self._compressed_offset > 0:
                        self._values_packed_np[:, :, :self._compressed_offset, :] = old_packed[:, :, :self._compressed_offset, :]
                        self._value_scales_np[:, :, :self._compressed_offset, :] = old_scales[:, :, :self._compressed_offset, :]
                        self._value_zeros_np[:, :, :self._compressed_offset, :] = old_zeros[:, :, :self._compressed_offset, :]
                self._values_packed = mx.array(self._values_packed_np)
                self._value_scales = mx.array(self._value_scales_np)
                self._value_zeros = mx.array(self._value_zeros_np)
            elif self.value_backend == "turboquant":
                tq_codec = self._value_codec_impl
                assert isinstance(tq_codec, TurboQuantValueCodec)
                packed_D = tq_codec.packed_dim
                if self._values_packed is None:
                    self._values_packed_np = np.zeros((B, self.n_kv_heads, new_cap, packed_D), dtype=np.uint32)
                    self._value_norms_np = np.zeros((B, self.n_kv_heads, new_cap), dtype=np.float32)
                else:
                    old_packed = self._values_packed_np
                    old_norms = self._value_norms_np
                    self._values_packed_np = np.zeros((B, self.n_kv_heads, new_cap, packed_D), dtype=np.uint32)
                    self._value_norms_np = np.zeros((B, self.n_kv_heads, new_cap), dtype=np.float32)
                    if self._compressed_offset > 0:
                        self._values_packed_np[:, :, :self._compressed_offset, :] = old_packed[:, :, :self._compressed_offset, :]
                        self._value_norms_np[:, :, :self._compressed_offset] = old_norms[:, :, :self._compressed_offset]
                self._values_packed = mx.array(self._values_packed_np)
                self._value_norms = mx.array(self._value_norms_np)
            elif self.value_backend == "sunshape":
                val_idx_dtype = np.uint8 if self.value_n_centroids <= 256 else np.uint16
                if self._value_indices is None:
                    self._value_indices_np = np.zeros(
                        (B, self.n_kv_heads, new_cap, self.value_n_blocks),
                        dtype=val_idx_dtype,
                    )
                else:
                    old_idx = self._value_indices_np
                    self._value_indices_np = np.zeros(
                        (B, self.n_kv_heads, new_cap, self.value_n_blocks),
                        dtype=val_idx_dtype,
                    )
                    if self._compressed_offset > 0:
                        self._value_indices_np[:, :, :self._compressed_offset, :] = old_idx[:, :, :self._compressed_offset, :]
                self._value_indices = mx.array(self._value_indices_np)
            else:
                if self._values_fp16 is None:
                    self._values_fp16_np = np.zeros((B, self.n_kv_heads, new_cap, self.head_dim), dtype=np.float16)
                else:
                    old_v = self._values_fp16_np
                    self._values_fp16_np = np.zeros((B, self.n_kv_heads, new_cap, self.head_dim), dtype=np.float16)
                    if self._compressed_offset > 0:
                        self._values_fp16_np[:, :, :self._compressed_offset, :] = old_v[:, :, :self._compressed_offset, :]

            self._compressed_capacity = new_cap

        # Write key indices
        self._key_indices_np[:, :, self._compressed_offset : self._compressed_offset + n_to_flush, :] = indices_np
        self._key_indices = mx.array(self._key_indices_np, dtype=mx_idx_dtype)

        # Write values — batch quantize all tokens at once for efficiency
        values_flush_np = np.array(values_flush.astype(mx.float32))
        if self.value_backend == "grouped":
            grouped_codec = self._value_codec_impl
            assert isinstance(grouped_codec, GroupedValueCodec)
            for b in range(B):
                for h in range(self.n_kv_heads):
                    vals_mx = mx.array(values_flush_np[b, h], dtype=mx.float32)
                    packed, scales, zeros = grouped_codec.quantize(vals_mx)
                    packed_np = np.array(packed)
                    scales_np = np.array(scales)
                    zeros_np = np.array(zeros)
                    self._values_packed_np[b, h, self._compressed_offset : self._compressed_offset + n_to_flush, :packed_np.shape[-1]] = packed_np
                    self._value_scales_np[b, h, self._compressed_offset : self._compressed_offset + n_to_flush, :] = scales_np
                    self._value_zeros_np[b, h, self._compressed_offset : self._compressed_offset + n_to_flush, :] = zeros_np
            self._values_packed = mx.array(self._values_packed_np)
            self._value_scales = mx.array(self._value_scales_np)
            self._value_zeros = mx.array(self._value_zeros_np)
            self.last_value_quant_backend = grouped_codec.last_quantize_backend
        elif self.value_backend == "turboquant":
            tq_codec = self._value_codec_impl
            assert isinstance(tq_codec, TurboQuantValueCodec)
            for b in range(B):
                for h in range(self.n_kv_heads):
                    vals_mx = mx.array(values_flush_np[b, h], dtype=mx.float32)
                    packed, norms = tq_codec.quantize(vals_mx)
                    packed_np = np.array(packed)
                    norms_np = np.array(norms)
                    self._values_packed_np[b, h, self._compressed_offset : self._compressed_offset + n_to_flush, :packed_np.shape[-1]] = packed_np
                    self._value_norms_np[b, h, self._compressed_offset : self._compressed_offset + n_to_flush] = norms_np
            self._values_packed = mx.array(self._values_packed_np)
            self._value_norms = mx.array(self._value_norms_np)
            self.last_value_quant_backend = tq_codec.last_quantize_backend
        elif self.value_backend == "sunshape":
            assert self.value_codec is not None
            val_idx_dtype = mx.uint8 if self.value_n_centroids <= 256 else mx.uint16
            for b in range(B):
                for h in range(self.n_kv_heads):
                    vals_mx = mx.array(values_flush_np[b, h], dtype=mx.float32)
                    quantized = self.value_codec.quantize(vals_mx)
                    self._value_indices_np[b, h, self._compressed_offset : self._compressed_offset + n_to_flush, :] = np.array(quantized.indices)
            self._value_indices = mx.array(self._value_indices_np, dtype=val_idx_dtype)
            self.last_value_quant_backend = "sunshape"
        else:
            self._values_fp16_np[:, :, self._compressed_offset : self._compressed_offset + n_to_flush, :] = \
                np.array(values_flush.astype(self.dtype))
            self._values_fp16 = mx.array(self._values_fp16_np, dtype=self.dtype)
            self.last_value_quant_backend = "fp16"

        mx.eval(self._key_indices)

        # Shift buffer: slide remaining tokens to the front
        remaining = self._buffer_offset - n_to_flush
        if remaining > 0:
            self._key_buffer_np[:, :, :remaining, :] = self._key_buffer_np[:, :, n_to_flush : n_to_flush + remaining, :]
            self._value_buffer_np[:, :, :remaining, :] = self._value_buffer_np[:, :, n_to_flush : n_to_flush + remaining, :]
            # Zero out the rest
            self._key_buffer_np[:, :, remaining:, :] = 0
            self._value_buffer_np[:, :, remaining:, :] = 0
        else:
            self._key_buffer_np[:] = 0
            self._value_buffer_np[:] = 0

        self._key_buffer = mx.array(self._key_buffer_np, dtype=self.dtype)
        self._value_buffer = mx.array(self._value_buffer_np, dtype=self.dtype)

        self._compressed_offset += n_to_flush
        self._buffer_offset = remaining

        mx.eval(self._key_buffer, self._value_buffer)

    # ------------------------------------------------------------------ #
    #  Update and fetch (main API)                                        #
    # ------------------------------------------------------------------ #

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Add new tokens; flush to compressed storage when buffer overflows.

        Parameters
        ----------
        keys : mx.array, shape (B, n_kv_heads, T, head_dim)
        values : mx.array, shape (B, n_kv_heads, T, head_dim)

        Returns
        -------
        keys_out, values_out : mx.array
        """
        B, n_kv_heads, T, head_dim = keys.shape
        assert n_kv_heads == self.n_kv_heads
        assert head_dim == self.head_dim

        # Initialize buffer on first call
        if self._key_buffer is None:
            buf_size = max(self.buffer_size, T)
            self._key_buffer_np = np.zeros((B, n_kv_heads, buf_size, head_dim), dtype=np.float16)
            self._value_buffer_np = np.zeros((B, n_kv_heads, buf_size, head_dim), dtype=np.float16)
            self._key_buffer = mx.array(self._key_buffer_np, dtype=self.dtype)
            self._value_buffer = mx.array(self._value_buffer_np, dtype=self.dtype)

        # Grow physical buffer if needed (temporarily hold prefill tokens)
        if self._buffer_offset + T > self._key_buffer.shape[2]:
            new_size = max(self._buffer_offset + T, self._key_buffer.shape[2] * 2)
            new_k = np.zeros((B, n_kv_heads, new_size, head_dim), dtype=np.float16)
            new_v = np.zeros((B, n_kv_heads, new_size, head_dim), dtype=np.float16)
            if self._buffer_offset > 0:
                new_k[:, :, :self._buffer_offset, :] = self._key_buffer_np[:, :, :self._buffer_offset, :]
                new_v[:, :, :self._buffer_offset, :] = self._value_buffer_np[:, :, :self._buffer_offset, :]
            self._key_buffer_np = new_k
            self._value_buffer_np = new_v

        # Append ALL new tokens to buffer memory arrays
        keys_fp = np.array(keys.astype(self.dtype))
        values_fp = np.array(values.astype(self.dtype))
        self._key_buffer_np[:, :, self._buffer_offset : self._buffer_offset + T, :] = keys_fp
        self._value_buffer_np[:, :, self._buffer_offset : self._buffer_offset + T, :] = values_fp

        self._buffer_offset += T

        # Ensure MLX backing is fresh before passing to compression
        self._key_buffer = mx.array(self._key_buffer_np, dtype=self.dtype)
        self._value_buffer = mx.array(self._value_buffer_np, dtype=self.dtype)

        # Flush oldest tokens if the LOGICAL buffer size is exceeded
        if self._buffer_offset > self.buffer_size:
            overflow = self._buffer_offset - self.buffer_size
            self._flush_buffer(B, overflow)

        mx.eval(self._key_buffer, self._value_buffer)

        return self.keys, self.values

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def keys(self) -> mx.array:
        """Concatenated keys: compressed (dequantized) + buffer."""
        parts = []
        if self._compressed_offset > 0 and self._key_indices is not None:
            # Dequantize compressed keys
            # Reuse the same logic as SunShapeKVCache.keys but simpler
            codec = self.codec
            T_c = self._compressed_offset
            indices = self._key_indices[:, :, :T_c, :]
            indices_np = np.array(indices.astype(mx.int32))
            centroids_np = np.array(codec.centroids)
            B = indices.shape[0]
            n_kv = indices.shape[1]

            k_hat_np = np.zeros((B, n_kv, T_c, self.head_dim), dtype=np.float32)

            for b in range(B):
                for h in range(n_kv):
                    k_np = np.zeros((T_c, self.head_dim), dtype=np.float32)
                    for blk in range(self.n_blocks):
                        sl = slice(blk * self.block_dim, (blk + 1) * self.block_dim)
                        k_np[:, sl] = centroids_np[blk][indices_np[b, h, :, blk]]
                    if codec.mode in {"profileperm_baseline", "profileperm_localmetric_dsq", "profileperm_mixed_precision"}:
                        k_np = k_np[:, np.array(codec.inv_permutation)]
                    elif codec.use_rotation:
                        k_np = k_np @ np.array(codec.rotation)
                    k_hat_np[b, h] = k_np
            parts.append(mx.array(k_hat_np, dtype=mx.float32))

        if self._buffer_offset > 0 and self._key_buffer is not None:
            parts.append(self._key_buffer[:, :, :self._buffer_offset, :].astype(mx.float32))

        if not parts:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=mx.float32)
        return mx.concatenate(parts, axis=2)

    @property
    def values(self) -> mx.array:
        """Concatenated values: compressed (dequantized) + buffer."""
        parts = []
        if self._compressed_offset > 0:
            if self.value_backend == "grouped" and self._values_packed is not None:
                grouped_codec = self._value_codec_impl
                assert isinstance(grouped_codec, GroupedValueCodec)
                packed = self._values_packed[:, :, :self._compressed_offset, :]
                packed_flat = packed.reshape(-1, packed.shape[-1])
                scales_flat = self._value_scales[:, :, :self._compressed_offset].reshape(-1, self._value_n_groups)
                zeros_flat = self._value_zeros[:, :, :self._compressed_offset].reshape(-1, self._value_n_groups)
                v_deq = grouped_codec.dequantize(
                    packed_flat,
                    scales_flat,
                    zeros_flat,
                )
                T_c = self._compressed_offset
                B = packed.shape[0]
                n_kv = packed.shape[1]
                parts.append(v_deq.reshape(B, n_kv, T_c, self.head_dim).astype(self.dtype))
            elif self.value_backend == "turboquant" and self._values_packed is not None:
                tq_codec = self._value_codec_impl
                assert isinstance(tq_codec, TurboQuantValueCodec)
                packed = self._values_packed[:, :, :self._compressed_offset, :]
                packed_flat = packed.reshape(-1, packed.shape[-1])
                norms_flat = self._value_norms[:, :, :self._compressed_offset].reshape(-1)
                v_deq = tq_codec.dequantize(packed_flat, norms_flat)
                T_c = self._compressed_offset
                B = packed.shape[0]
                n_kv = packed.shape[1]
                parts.append(v_deq.reshape(B, n_kv, T_c, self.head_dim).astype(self.dtype))
            elif self.value_backend == "sunshape" and self._value_indices is not None:
                assert self.value_codec is not None
                T_c = self._compressed_offset
                indices = self._value_indices[:, :, :T_c, :]
                indices_np = np.array(indices.astype(mx.int32))
                centroids_np = np.array(self.value_codec.centroids)
                inv_perm_np = np.array(self.value_codec.inv_permutation)
                rotation_np = np.array(self.value_codec.rotation) if self.value_codec.use_rotation else None
                B = indices.shape[0]
                n_kv = indices.shape[1]
                v_hat_np = np.zeros((B, n_kv, T_c, self.head_dim), dtype=np.float32)

                for b in range(B):
                    for h in range(n_kv):
                        v_np = np.zeros((T_c, self.head_dim), dtype=np.float32)
                        for blk in range(self.value_n_blocks):
                            sl = slice(blk * self.value_codec.block_dim, (blk + 1) * self.value_codec.block_dim)
                            v_np[:, sl] = centroids_np[blk][indices_np[b, h, :, blk]]
                        if self.value_codec.mode in {
                            "profileperm_baseline",
                            "profileperm_localmetric_dsq",
                            "profileperm_mixed_precision",
                        }:
                            v_np = v_np[:, inv_perm_np]
                        elif self.value_codec.use_rotation and rotation_np is not None:
                            v_np = v_np @ rotation_np
                        v_hat_np[b, h] = v_np
                parts.append(mx.array(v_hat_np, dtype=self.dtype))
            elif self._values_fp16 is not None:
                parts.append(self._values_fp16[:, :, :self._compressed_offset, :])

        if self._buffer_offset > 0 and self._value_buffer is not None:
            parts.append(self._value_buffer[:, :, :self._buffer_offset, :])

        if not parts:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=self.dtype)
        return mx.concatenate(parts, axis=2)

    @property
    def key_indices(self) -> mx.array:
        """Compressed key indices (buffer keys are not quantized)."""
        if self._key_indices is None or self._compressed_offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.n_blocks), dtype=mx.uint8)
        return self._key_indices[:, :, : self._compressed_offset, :]

    @property
    def buffer_keys(self) -> mx.array:
        """Full-precision keys currently in the buffer."""
        if self._key_buffer is None or self._buffer_offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=self.dtype)
        return self._key_buffer[:, :, :self._buffer_offset, :]

    @property
    def buffer_values(self) -> mx.array:
        """Full-precision values currently in the buffer."""
        if self._value_buffer is None or self._buffer_offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=self.dtype)
        return self._value_buffer[:, :, :self._buffer_offset, :]

    @property
    def compressed_offset(self) -> int:
        """Number of tokens in compressed storage."""
        return self._compressed_offset

    @property
    def buffer_offset(self) -> int:
        """Number of tokens in the full-precision buffer."""
        return self._buffer_offset

    # ------------------------------------------------------------------ #
    #  Memory estimation                                                   #
    # ------------------------------------------------------------------ #

    @property
    def nbytes(self) -> int:
        """Total storage size in bytes (compressed + buffer)."""
        B = 1  # approximate
        # Compressed keys
        idx_bytes = B * self.n_kv_heads * self._compressed_offset * self.n_blocks * (
            1 if self.n_centroids <= 256 else 2
        )
        # Compressed values
        if self.value_backend == "grouped" and self._compressed_offset > 0:
            packed_D = _packed_dim(self.head_dim, self.value_bits)
            val_bytes = B * self.n_kv_heads * self._compressed_offset * packed_D * 4  # uint32 = 4 bytes
            val_bytes += B * self.n_kv_heads * self._compressed_offset * self._value_n_groups * 4 * 2  # scales + zeros
        elif self.value_backend == "turboquant" and self._compressed_offset > 0:
            packed_D = _packed_dim(self.head_dim, self.value_bits)
            val_bytes = B * self.n_kv_heads * self._compressed_offset * packed_D * 4
            val_bytes += B * self.n_kv_heads * self._compressed_offset * 4
        elif self.value_backend == "sunshape" and self._compressed_offset > 0:
            bytes_per_index = 1 if self.value_n_centroids <= 256 else 2
            val_bytes = B * self.n_kv_heads * self._compressed_offset * self.value_n_blocks * bytes_per_index
        else:
            val_bytes = B * self.n_kv_heads * self._compressed_offset * self.head_dim * 2
        # Buffer (full precision)
        buf_bytes = B * self.n_kv_heads * self._buffer_offset * self.head_dim * 2 * 2  # K + V in fp16
        return idx_bytes + val_bytes + buf_bytes

    @property
    def nbytes_equivalent_fp16(self) -> int:
        """Equivalent FP16 storage for both K and V."""
        T = self.offset
        if T == 0:
            return 0
        B = 1
        return 2 * B * self.n_kv_heads * T * self.head_dim * 2

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP16."""
        fp16 = self.nbytes_equivalent_fp16
        if fp16 == 0:
            return 1.0
        return fp16 / max(1, self.nbytes)

    # ------------------------------------------------------------------ #
    #  mlx-lm KVCache compatibility                                        #
    # ------------------------------------------------------------------ #

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> None:
        """Remove n tokens from the end of the cache."""
        if n <= 0:
            return
        # Trim from buffer first, then compressed
        buf_trim = min(n, self._buffer_offset)
        self._buffer_offset -= buf_trim
        remaining = n - buf_trim
        if remaining > 0:
            self._compressed_offset = max(0, self._compressed_offset - remaining)

    def empty(self) -> bool:
        return self.offset == 0

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **kwargs) -> mx.array | str:
        """Create a causal mask (matches mlx-lm API)."""
        T = self.offset
        if T == 0 and N == 0:
            return "none"
        q_pos = mx.arange(T, T + N)
        k_pos = mx.arange(T + N)
        causal = q_pos[:, None] >= k_pos[None, :]
        if return_array:
            return mx.where(causal, mx.array(0.0), mx.array(mx.finfo(mx.float16).min)).astype(mx.float16)
        return "causal"

    def __repr__(self) -> str:
        return (
            f"HybridSunShapeKVCache(offset={self.offset}, "
            f"compressed={self._compressed_offset}, buffer={self._buffer_offset}, "
            f"value_backend={self.value_backend}, "
            f"compression={self.compression_ratio:.1f}x)"
        )

    @property
    def state(self) -> tuple[mx.array, ...]:
        """Cache state for serialization."""
        parts = []
        if self._key_indices is not None and self._compressed_offset > 0:
            parts.append(self._key_indices[:, :, :self._compressed_offset, :])
        if self._key_buffer is not None and self._buffer_offset > 0:
            parts.append(self._key_buffer[:, :, :self._buffer_offset, :])
        if not parts:
            parts.append(mx.zeros((1,)))
        return tuple(parts)


# ==================================================================== #
#  Factory function — unified cache creation API                       #
# ==================================================================== #


def create_kv_cache(
    codec: SunShapeBlockCodec,
    *,
    n_kv_heads: int = 1,
    head_dim: int | None = None,
    dtype: mx.Dtype = mx.float16,
    value_backend: str = "fp16",
    value_bits: int | None = None,
    value_group_size: int = 64,
    value_seed: int = 43,
    value_codec: SunShapeBlockCodec | None = None,
    buffer_size: int = _DEFAULT_BUFFER_SIZE,
    prefer_metal_kernels: bool = True,
) -> SunShapeKVCache | HybridSunShapeKVCache:
    """Create a SunShape KV cache with configurable key and value compression.

    This is the recommended entry point for creating SunShape caches.
    Keys are always compressed via the SunShape block-VQ codec.
    Values can use a different compression backend.

    Parameters
    ----------
    codec : SunShapeBlockCodec
        Fitted SunShape codec for key compression.
    n_kv_heads : int
        Number of KV attention heads.
    head_dim : int or None
        Head dimension (defaults to ``codec.head_dim``).
    dtype : mx.Dtype
        Storage dtype for full-precision values.
    value_backend : str
        Value compression backend:

        - ``"fp16"`` — no value compression (default)
        - ``"grouped"`` — asymmetric per-group min-max quantization
        - ``"turboquant"`` — TurboQuant scalar rotated quantization
        - ``"sunshape"`` — SunShape block VQ (uses ``value_codec``)
    value_bits : int or None
        Bits per value element (required for non-fp16 backends).
        Supported: 2, 3, or 4.
    value_group_size : int
        Group size for ``"grouped"`` backend (default 64).
    value_seed : int
        Random seed for ``"turboquant"`` rotation matrix.
    value_codec : SunShapeBlockCodec or None
        Separate codec for value compression (``"sunshape"`` backend only).
        If None, reuses the key codec.
    buffer_size : int
        Number of recent tokens to keep in full precision before
        flushing to compressed storage.  Only used when value
        compression is enabled.  Set to 0 to compress immediately.
    prefer_metal_kernels : bool
        Whether to prefer Metal kernels when available.

    Returns
    -------
    cache : SunShapeKVCache or HybridSunShapeKVCache

    Examples
    --------
    SunShape keys + FP16 values (simplest):

    >>> cache = create_kv_cache(codec, n_kv_heads=8)

    SunShape keys + TurboQuant 2-bit values:

    >>> cache = create_kv_cache(
    ...     codec, n_kv_heads=8,
    ...     value_backend="turboquant", value_bits=2,
    ... )

    SunShape keys + grouped 4-bit values (no buffer):

    >>> cache = create_kv_cache(
    ...     codec, n_kv_heads=8,
    ...     value_backend="grouped", value_bits=4,
    ...     buffer_size=0,
    ... )
    """
    if value_backend == "fp16" and buffer_size == _DEFAULT_BUFFER_SIZE:
        # Simple path: no value compression, no buffer needed
        return SunShapeKVCache(
            codec,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    # Any value compression (or explicit buffer) → use hybrid cache
    return HybridSunShapeKVCache(
        codec,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        buffer_size=buffer_size,
        value_backend=value_backend,
        value_bits=value_bits,
        value_group_size=value_group_size,
        value_seed=value_seed,
        value_codec=value_codec,
        prefer_metal_kernels=prefer_metal_kernels,
    )

