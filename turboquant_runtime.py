"""TurboQuant-style scalar KV cache for MLX runtime benchmarks."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from sunshape_mlx.value_codecs import TurboQuantValueCodec

_STEP = 256


class TurboQuantKVCache:
    """Simple TurboQuant-style KV cache using scalar rotated quantization."""

    def __init__(
        self,
        *,
        head_dim: int,
        n_kv_heads: int,
        bits: int = 2,
        dtype: mx.Dtype = mx.float16,
        key_seed: int = 42,
        value_seed: int = 43,
    ):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.bits = bits
        self.dtype = dtype
        self.offset = 0
        self.key_codec = TurboQuantValueCodec(head_dim=head_dim, bits=bits, seed=key_seed)
        self.value_codec = TurboQuantValueCodec(head_dim=head_dim, bits=bits, seed=value_seed)

        self._key_packed: mx.array | None = None
        self._key_packed_np: np.ndarray | None = None
        self._key_norms: mx.array | None = None
        self._key_norms_np: np.ndarray | None = None
        self._value_packed: mx.array | None = None
        self._value_packed_np: np.ndarray | None = None
        self._value_norms: mx.array | None = None
        self._value_norms_np: np.ndarray | None = None
        self._capacity = 0

    def _ensure_capacity(self, B: int, new_tokens: int) -> None:
        needed = self.offset + new_tokens
        if needed <= self._capacity:
            return

        new_capacity = ((needed + _STEP - 1) // _STEP) * _STEP
        packed_dim = self.key_codec.packed_dim

        def _grow(old_np: np.ndarray | None, shape: tuple[int, ...], dtype) -> np.ndarray:
            new_np = np.zeros(shape, dtype=dtype)
            if old_np is not None and self.offset > 0:
                if old_np.ndim == 4:
                    new_np[:, :, : self.offset, :] = old_np[:, :, : self.offset, :]
                else:
                    new_np[:, :, : self.offset] = old_np[:, :, : self.offset]
            return new_np

        self._key_packed_np = _grow(self._key_packed_np, (B, self.n_kv_heads, new_capacity, packed_dim), np.uint32)
        self._value_packed_np = _grow(self._value_packed_np, (B, self.n_kv_heads, new_capacity, packed_dim), np.uint32)
        self._key_norms_np = _grow(self._key_norms_np, (B, self.n_kv_heads, new_capacity), np.float32)
        self._value_norms_np = _grow(self._value_norms_np, (B, self.n_kv_heads, new_capacity), np.float32)

        self._key_packed = mx.array(self._key_packed_np)
        self._value_packed = mx.array(self._value_packed_np)
        self._key_norms = mx.array(self._key_norms_np)
        self._value_norms = mx.array(self._value_norms_np)
        self._capacity = new_capacity

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        B, n_kv_heads, T, head_dim = keys.shape
        assert n_kv_heads == self.n_kv_heads
        assert head_dim == self.head_dim
        self._ensure_capacity(B, T)

        for b in range(B):
            for h in range(n_kv_heads):
                k_packed, k_norms = self.key_codec.quantize(keys[b, h].astype(mx.float32))
                v_packed, v_norms = self.value_codec.quantize(values[b, h].astype(mx.float32))
                self._key_packed_np[b, h, self.offset : self.offset + T, :] = np.array(k_packed)
                self._value_packed_np[b, h, self.offset : self.offset + T, :] = np.array(v_packed)
                self._key_norms_np[b, h, self.offset : self.offset + T] = np.array(k_norms)
                self._value_norms_np[b, h, self.offset : self.offset + T] = np.array(v_norms)

        self._key_packed = mx.array(self._key_packed_np)
        self._value_packed = mx.array(self._value_packed_np)
        self._key_norms = mx.array(self._key_norms_np)
        self._value_norms = mx.array(self._value_norms_np)
        self.offset += T
        return self.keys, self.values

    @property
    def keys(self) -> mx.array:
        if self._key_packed is None or self.offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=mx.float32)
        packed = self._key_packed[:, :, : self.offset, :]
        norms = self._key_norms[:, :, : self.offset]
        flat = packed.reshape(-1, packed.shape[-1])
        flat_norms = norms.reshape(-1)
        deq = self.key_codec.dequantize(flat, flat_norms)
        return deq.reshape(packed.shape[0], packed.shape[1], self.offset, self.head_dim)

    @property
    def values(self) -> mx.array:
        if self._value_packed is None or self.offset == 0:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=self.dtype)
        packed = self._value_packed[:, :, : self.offset, :]
        norms = self._value_norms[:, :, : self.offset]
        flat = packed.reshape(-1, packed.shape[-1])
        flat_norms = norms.reshape(-1)
        deq = self.value_codec.dequantize(flat, flat_norms)
        return deq.reshape(packed.shape[0], packed.shape[1], self.offset, self.head_dim).astype(self.dtype)

    @property
    def nbytes(self) -> int:
        if self._key_packed is None:
            return 0
        B = self._key_packed.shape[0]
        T = self.offset
        packed_bytes = B * self.n_kv_heads * T * self.key_codec.packed_dim * 4
        norm_bytes = B * self.n_kv_heads * T * 4 * 2
        return 2 * packed_bytes + norm_bytes

    @property
    def nbytes_equivalent_fp16(self) -> int:
        if self._key_packed is None:
            return 0
        B = self._key_packed.shape[0]
        T = self.offset
        return 2 * B * self.n_kv_heads * T * self.head_dim * 2

    @property
    def compression_ratio(self) -> float:
        fp16 = self.nbytes_equivalent_fp16
        return 1.0 if fp16 == 0 else fp16 / max(1, self.nbytes)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> None:
        self.offset = max(0, self.offset - n)

    def empty(self) -> bool:
        return self.offset == 0

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **kwargs):
        T = self.offset
        if T == 0 and N == 0:
            return "none"
        q_pos = mx.arange(T, T + N)
        k_pos = mx.arange(T + N)
        causal = q_pos[:, None] >= k_pos[None, :]
        if return_array:
            return mx.where(causal, mx.array(0.0), mx.array(mx.finfo(mx.float16).min)).astype(mx.float16)
        return "causal"

    @property
    def state(self):
        if self._key_packed is None:
            return []
        return [
            self._key_packed[:, :, : self.offset, :],
            self._key_norms[:, :, : self.offset],
            self._value_packed[:, :, : self.offset, :],
            self._value_norms[:, :, : self.offset],
        ]


def turboquant_sdpa(
    queries: mx.array,
    cache: TurboQuantKVCache,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attention over a dequantized TurboQuant-style cache."""
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = cache.n_kv_heads
    n_repeats = n_q_heads // n_kv_heads
    T_kv = cache.offset

    if T_kv == 0:
        return mx.zeros_like(queries)

    q_scaled = queries.astype(mx.float32) * scale
    keys = cache.keys.astype(mx.float32)
    values = cache.values.astype(mx.float32)

    q_grouped = q_scaled.reshape(B, n_kv_heads, n_repeats, T_q, D)
    output = mx.zeros((B, n_q_heads, T_q, D), dtype=mx.float32)

    for b in range(B):
        for h in range(n_kv_heads):
            k_h = keys[b, h]
            v_h = values[b, h]
            for r in range(n_repeats):
                q_h = q_grouped[b, h, r]
                scores = q_h @ k_h.T
                if mask is not None:
                    if isinstance(mask, str) and mask == "causal":
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
                weights = mx.softmax(scores, axis=-1, precise=True)
                output[b, h * n_repeats + r] = weights @ v_h

    return output.astype(queries.dtype)
