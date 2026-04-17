"""Tests for sunshape_mlx — rotation, codec, cache, and attention."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

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
from sunshape_mlx.codec import SunShapeBlockCodec, SunShapeQuantized, _kmeans
from sunshape_mlx.cache import SunShapeKVCache, HybridSunShapeKVCache
from sunshape_mlx.turboquant_runtime import TurboQuantKVCache, turboquant_sdpa
from sunshape_mlx.kernels import (
    pack_indices,
    unpack_indices,
    quantize_values,
    dequantize_values,
    fused_block_vq_attention,
    get_kernel_stats,
    reset_kernel_stats,
)
from sunshape_mlx.value_codecs import TurboQuantValueCodec


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

HEAD_DIM = 64
BLOCK_DIM = 8
N_CENTROIDS = 256  # 8 bits/block when block_dim=8, 1 bit/dim


@pytest.fixture
def random_keys():
    mx.random.seed(42)
    return mx.random.normal((512, HEAD_DIM))


@pytest.fixture
def random_queries():
    mx.random.seed(137)
    return mx.random.normal((256, HEAD_DIM))


@pytest.fixture
def fitted_codec(random_queries, random_keys):
    codec = SunShapeBlockCodec(
        head_dim=HEAD_DIM,
        block_dim=BLOCK_DIM,
        n_centroids=N_CENTROIDS,
        mode="profileperm_baseline",
    )
    codec.fit(random_queries, random_keys, kmeans_iters=10, seed=42)
    return codec


# ------------------------------------------------------------------ #
#  Rotation tests                                                      #
# ------------------------------------------------------------------ #


class TestRotation:
    def test_rotation_orthogonal(self):
        R = generate_rotation_matrix(HEAD_DIM, seed=0)
        # R @ R.T ≈ I
        I = R @ R.T
        mx.eval(I)
        I_np = np.array(I)
        np.testing.assert_allclose(I_np, np.eye(HEAD_DIM), atol=1e-5)

    def test_rotation_det_plus_one(self):
        R = generate_rotation_matrix(HEAD_DIM, seed=0)
        # det ≈ 1 (proper rotation)
        # Use numpy for det
        R_np = np.array(R)
        det = np.linalg.det(R_np)
        assert abs(det - 1.0) < 1e-4, f"det(R) = {det}, expected ≈ 1"

    def test_rotation_reproducible(self):
        R1 = generate_rotation_matrix(HEAD_DIM, seed=42)
        R2 = generate_rotation_matrix(HEAD_DIM, seed=42)
        np.testing.assert_allclose(np.array(R1), np.array(R2), atol=1e-6)


class TestPermutation:
    def test_permutation_is_valid(self, random_queries):
        perm = covariance_block_permutation(random_queries, BLOCK_DIM)
        perm_np = np.array(perm)
        assert perm_np.shape == (HEAD_DIM,)
        # Should be a valid permutation of [0, d)
        assert set(perm_np.tolist()) == set(range(HEAD_DIM))

    def test_invert_permutation(self, random_queries):
        perm = covariance_block_permutation(random_queries, BLOCK_DIM)
        inv = invert_permutation(perm)
        # perm[inv] should be identity
        perm_np = np.array(perm)
        inv_np = np.array(inv)
        identity = perm_np[inv_np]
        np.testing.assert_array_equal(identity, np.arange(HEAD_DIM))

    def test_apply_permutation_roundtrip(self, random_queries):
        x = random_queries
        perm = covariance_block_permutation(x, BLOCK_DIM)
        inv = invert_permutation(perm)
        x_perm = apply_permutation(x, perm)
        x_back = apply_permutation(x_perm, inv)
        np.testing.assert_allclose(np.array(x_back), np.array(x), atol=1e-5)


# ------------------------------------------------------------------ #
#  Codec tests                                                         #
# ------------------------------------------------------------------ #


class TestKMeans:
    def test_kmeans_shape(self):
        mx.random.seed(0)
        data = mx.random.normal((200, BLOCK_DIM))
        centroids = _kmeans(data, 16, n_iters=5, seed=0)
        assert centroids.shape == (16, BLOCK_DIM)

    def test_kmeans_convergence(self):
        """K-means should reduce distortion over iterations."""
        mx.random.seed(0)
        data = mx.random.normal((500, BLOCK_DIM))
        c1 = _kmeans(data, 16, n_iters=1, seed=0)
        c10 = _kmeans(data, 16, n_iters=10, seed=0)
        data_np = np.array(data)
        c1_np, c10_np = np.array(c1), np.array(c10)
        d1 = np.min(np.sum((data_np[:, None] - c1_np[None]) ** 2, axis=-1), axis=1).mean()
        d10 = np.min(np.sum((data_np[:, None] - c10_np[None]) ** 2, axis=-1), axis=1).mean()
        assert d10 < d1, "More k-means iterations should reduce distortion"


class TestCodec:
    def test_codec_fit(self, fitted_codec):
        c = fitted_codec
        assert c.centroids.shape == (c.n_blocks, c.n_centroids, c.block_dim)
        assert c.permutation.shape == (HEAD_DIM,)

    def test_codec_quantize_dequantize(self, fitted_codec, random_keys):
        quantized = fitted_codec.quantize(random_keys)
        assert quantized.indices.shape == (random_keys.shape[0], fitted_codec.n_blocks)
        k_hat = fitted_codec.dequantize(quantized)
        assert k_hat.shape == random_keys.shape

    def test_codec_roundtrip_mse(self, fitted_codec, random_keys):
        """Round-trip MSE should be reasonable."""
        k_hat = fitted_codec(random_keys)
        mse = float(mx.mean(mx.sum((random_keys.astype(mx.float32) - k_hat) ** 2, axis=-1)).item())
        # With 256 centroids per 8-dim block, MSE should be small relative to ||k||^2
        k_norm = float(mx.mean(mx.sum(random_keys.astype(mx.float32) ** 2, axis=-1)).item())
        relative_mse = mse / max(k_norm, 1e-8)
        assert relative_mse < 0.5, f"Relative MSE too high: {relative_mse}"

    def test_codec_attention_scores(self, fitted_codec, random_queries, random_keys):
        """Attention scores via precomputed dots should match dequantize+matmul."""
        quantized = fitted_codec.quantize(random_keys)
        # Fast path
        scores_fast = fitted_codec.attention_scores(random_queries[:8], quantized)
        # Reference path
        k_hat = fitted_codec.dequantize(quantized)
        scores_ref = (random_queries[:8].astype(mx.float32) @ k_hat.T)
        np.testing.assert_allclose(
            np.array(scores_fast), np.array(scores_ref), atol=1e-3, rtol=1e-2
        )

    def test_codec_rotated_mode(self, random_queries, random_keys):
        codec = SunShapeBlockCodec(
            head_dim=HEAD_DIM,
            block_dim=BLOCK_DIM,
            n_centroids=N_CENTROIDS,
            mode="rotated",
            use_rotation=True,
            rotation_seed=42,
        )
        codec.fit(random_queries, random_keys, kmeans_iters=5, seed=42)
        k_hat = codec(random_keys)
        assert k_hat.shape == random_keys.shape


# ------------------------------------------------------------------ #
#  Cache tests                                                         #
# ------------------------------------------------------------------ #


class TestCache:
    def test_cache_create(self, fitted_codec):
        cache = SunShapeKVCache(fitted_codec, n_kv_heads=4, head_dim=HEAD_DIM)
        assert cache.offset == 0
        assert cache.empty()

    def test_cache_update_and_fetch(self, fitted_codec):
        cache = SunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM)
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert cache.offset == 16
        assert not cache.empty()
        assert v_out.shape == (1, 2, 16, HEAD_DIM)

    def test_cache_compression_ratio(self, fitted_codec):
        cache = SunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM)
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        # Keys: 8 bits/block * 8 blocks = 64 bits/key vs 128*16=2048 bits/key fp16
        # Compression should be > 2x (keys compressed, values fp16)
        ratio = cache.compression_ratio
        assert ratio > 1.5, f"Compression ratio too low: {ratio}"

    def test_cache_trim(self, fitted_codec):
        cache = SunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM)
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 32, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 32, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 32
        cache.trim(8)
        assert cache.offset == 24


# ------------------------------------------------------------------ #
#  Kernel / packing tests                                              #
# ------------------------------------------------------------------ #


class TestPacking:
    def test_pack_unpack_2bit(self):
        indices = mx.array(np.random.randint(0, 4, (2, 32)), dtype=mx.uint8)
        packed = pack_indices(indices, bits=2)
        unpacked = unpack_indices(packed, D=32, bits=2)
        np.testing.assert_array_equal(np.array(unpacked), np.array(indices))

    def test_pack_unpack_4bit(self):
        indices = mx.array(np.random.randint(0, 16, (2, 32)), dtype=mx.uint8)
        packed = pack_indices(indices, bits=4)
        unpacked = unpack_indices(packed, D=32, bits=4)
        np.testing.assert_array_equal(np.array(unpacked), np.array(indices))

    def test_pack_unpack_3bit(self):
        indices = mx.array(np.random.randint(0, 8, (2, 30)), dtype=mx.uint8)
        packed = pack_indices(indices, bits=3)
        unpacked = unpack_indices(packed, D=30, bits=3)
        np.testing.assert_array_equal(np.array(unpacked), np.array(indices))


# ------------------------------------------------------------------ #
#  Block metric tests                                                  #
# ------------------------------------------------------------------ #


class TestMetrics:
    def test_block_affinity_gate(self, random_queries):
        affinity, active = block_affinity_gate(random_queries, BLOCK_DIM)
        n_blocks = HEAD_DIM // BLOCK_DIM
        assert affinity.shape == (n_blocks,)
        assert active.shape == (n_blocks,)
        # At least one block should be active
        assert np.array(active).any()

    def test_mixed_precision_mask(self, random_queries):
        heavy, mass = mixed_precision_block_mask(random_queries, BLOCK_DIM)
        n_blocks = HEAD_DIM // BLOCK_DIM
        assert heavy.shape == (n_blocks,)
        assert mass.shape == (n_blocks,)

    def test_safe_normalize(self):
        x = mx.array([[3.0, 4.0], [0.0, 0.0]])
        x_norm, norms = safe_normalize(x)
        x_norm_np = np.array(x_norm)
        # First vector should be unit
        np.testing.assert_allclose(np.linalg.norm(x_norm_np[0]), 1.0, atol=1e-5)
        # Zero vector should not produce NaN
        assert not np.any(np.isnan(x_norm_np[1]))


# ------------------------------------------------------------------ #
#  WHT rotation tests                                                  #
# ------------------------------------------------------------------ #


class TestWHT:
    def test_wht_signs_shape(self):
        signs = generate_wht_signs(HEAD_DIM, seed=0)
        assert signs.shape == (HEAD_DIM,)
        assert signs.dtype == np.float32
        # All values should be ±1
        unique = np.unique(signs)
        assert set(unique.tolist()).issubset({-1.0, 1.0})

    def test_wht_forward_inverse(self):
        """WHT forward then inverse should recover the original."""
        signs = generate_wht_signs(HEAD_DIM, seed=42)
        mx.random.seed(0)
        x = mx.random.normal((16, HEAD_DIM))
        y = apply_wht(x, signs, inverse=False)
        x_back = apply_wht(y, signs, inverse=True)
        np.testing.assert_allclose(np.array(x_back), np.array(x), atol=1e-4)

    def test_wht_orthogonal(self):
        """WHT rotation should preserve vector norms."""
        signs = generate_wht_signs(HEAD_DIM, seed=42)
        mx.random.seed(0)
        x = mx.random.normal((32, HEAD_DIM))
        y = apply_wht(x, signs, inverse=False)
        norms_x = np.linalg.norm(np.array(x), axis=1)
        norms_y = np.linalg.norm(np.array(y), axis=1)
        np.testing.assert_allclose(norms_y, norms_x, atol=1e-4)

    def test_wht_power_of_2_required(self):
        with pytest.raises(ValueError, match="power of 2"):
            generate_wht_signs(48, seed=0)  # 48 is not a power of 2


# ------------------------------------------------------------------ #
#  Value quantization tests                                            #
# ------------------------------------------------------------------ #


class TestValueQuantization:
    def test_quantize_dequantize_4bit(self):
        mx.random.seed(0)
        values = mx.random.normal((64, HEAD_DIM))
        packed, scales, zeros, n_groups = quantize_values(values, bits=4, group_size=16)
        values_hat = dequantize_values(packed, scales, zeros, D=HEAD_DIM, bits=4, group_size=16)
        assert values_hat.shape == values.shape
        # 4-bit quantization should be decent
        mse = float(mx.mean((values.astype(mx.float32) - values_hat) ** 2).item())
        energy = float(mx.mean(values.astype(mx.float32) ** 2).item())
        assert mse / max(energy, 1e-8) < 0.1, f"4-bit relative MSE too high: {mse/energy}"

    def test_quantize_dequantize_2bit(self):
        mx.random.seed(0)
        values = mx.random.normal((64, HEAD_DIM))
        packed, scales, zeros, n_groups = quantize_values(values, bits=2, group_size=16)
        values_hat = dequantize_values(packed, scales, zeros, D=HEAD_DIM, bits=2, group_size=16)
        assert values_hat.shape == values.shape
        # 2-bit will be worse but should still work
        mse = float(mx.mean((values.astype(mx.float32) - values_hat) ** 2).item())
        assert mse < 1.0, f"2-bit MSE too high: {mse}"


# ------------------------------------------------------------------ #
#  Hybrid cache tests                                                  #
# ------------------------------------------------------------------ #


class TestHybridCache:
    def test_hybrid_create(self, fitted_codec):
        cache = HybridSunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM, buffer_size=32)
        assert cache.offset == 0
        assert cache.empty()

    def test_hybrid_update_buffer_only(self, fitted_codec):
        cache = HybridSunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM, buffer_size=64)
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 16
        assert cache.buffer_offset == 16
        assert cache.compressed_offset == 0

    def test_hybrid_flush_on_overflow(self, fitted_codec):
        cache = HybridSunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM, buffer_size=32)
        mx.random.seed(0)
        # First batch fits in buffer
        keys1 = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        values1 = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys1, values1)
        assert cache.buffer_offset == 16

        # Second batch triggers flush
        keys2 = mx.random.normal((1, 2, 24, HEAD_DIM)).astype(mx.float16)
        values2 = mx.random.normal((1, 2, 24, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys2, values2)
        # Should have flushed some tokens
        assert cache.offset == 40
        assert cache.compressed_offset > 0

    def test_hybrid_compression_ratio(self, fitted_codec):
        cache = HybridSunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM, buffer_size=32)
        mx.random.seed(0)
        # Add enough tokens to trigger flush
        keys = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        ratio = cache.compression_ratio
        assert ratio > 1.0, f"Compression ratio too low: {ratio}"

    def test_hybrid_with_value_quantization(self, fitted_codec):
        cache = HybridSunShapeKVCache(
            fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM,
            buffer_size=32, value_backend="grouped", value_bits=4, value_group_size=16,
        )
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 64
        assert cache.compressed_offset > 0

    def test_hybrid_with_turboquant_values(self, fitted_codec):
        cache = HybridSunShapeKVCache(
            fitted_codec,
            n_kv_heads=2,
            head_dim=HEAD_DIM,
            buffer_size=32,
            value_backend="turboquant",
            value_bits=2,
        )
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 64
        assert cache.compressed_offset > 0
        assert cache.last_value_quant_backend in {"metal", "mlx"}

    def test_hybrid_with_sunshape_values(self, random_queries, random_keys):
        value_codec = SunShapeBlockCodec(
            head_dim=HEAD_DIM,
            block_dim=BLOCK_DIM,
            n_centroids=N_CENTROIDS,
            mode="profileperm_baseline",
        )
        value_codec.fit(random_queries, random_keys, kmeans_iters=8, seed=7)
        cache = HybridSunShapeKVCache(
            value_codec,
            n_kv_heads=2,
            head_dim=HEAD_DIM,
            buffer_size=32,
            value_backend="sunshape",
            value_bits=1,
            value_codec=value_codec,
        )
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 64, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 64
        assert cache.compressed_offset > 0
        assert cache.values.shape == (1, 2, 64, HEAD_DIM)

    def test_hybrid_trim(self, fitted_codec):
        cache = HybridSunShapeKVCache(fitted_codec, n_kv_heads=2, head_dim=HEAD_DIM, buffer_size=32)
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 48, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 48, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        offset_before = cache.offset
        cache.trim(8)
        assert cache.offset == offset_before - 8


# ------------------------------------------------------------------ #
#  Fused attention test                                                #
# ------------------------------------------------------------------ #


class TestFusedAttention:
    def test_fused_attention_output_shape(self, fitted_codec, random_keys):
        quantized = fitted_codec.quantize(random_keys[:32])
        q = random_keys[:4].astype(mx.float32)
        q_t = fitted_codec._forward_transform(q)
        q_blocks = q_t.reshape(4, fitted_codec.n_blocks, fitted_codec.block_dim)
        qdots = mx.einsum("qbd,bcd->qbc", q_blocks, fitted_codec.centroids)
        output = fused_block_vq_attention(
            qdots, quantized.indices, random_keys[:32].astype(mx.float32),
            T_q=4, T_kv=32, n_blocks=fitted_codec.n_blocks,
            n_centroids=fitted_codec.n_centroids, head_dim=HEAD_DIM,
        )
        assert output.shape == (4, HEAD_DIM)


class TestTurboQuantValueCodec:
    def test_roundtrip_shape_and_quality(self):
        codec = TurboQuantValueCodec(head_dim=HEAD_DIM, bits=2)
        mx.random.seed(0)
        values = mx.random.normal((64, HEAD_DIM)).astype(mx.float32)
        packed, norms = codec.quantize(values)
        values_hat = codec.dequantize(packed, norms)
        assert values_hat.shape == values.shape
        mse = float(mx.mean((values - values_hat) ** 2).item())
        assert mse < 1.5, f"TurboQuant 2-bit value MSE too high: {mse}"

    def test_kernel_stats_visible(self):
        codec = TurboQuantValueCodec(head_dim=HEAD_DIM, bits=2)
        reset_kernel_stats()
        mx.random.seed(1)
        values = mx.random.normal((8, HEAD_DIM)).astype(mx.float32)
        codec.quantize(values)
        stats = get_kernel_stats()
        if codec.last_quantize_backend == "metal":
            assert stats.get("sunshape_scalar_quantize_metal", 0) >= 1


class TestTurboQuantRuntimeCache:
    def test_update_and_fetch(self):
        cache = TurboQuantKVCache(head_dim=HEAD_DIM, n_kv_heads=2, bits=2)
        mx.random.seed(0)
        keys = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert cache.offset == 16
        assert k_out.shape == (1, 2, 16, HEAD_DIM)
        assert v_out.shape == (1, 2, 16, HEAD_DIM)
        assert cache.compression_ratio > 1.0

    def test_turboquant_sdpa_shape(self):
        cache = TurboQuantKVCache(head_dim=HEAD_DIM, n_kv_heads=2, bits=2)
        mx.random.seed(1)
        keys = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal((1, 2, 16, HEAD_DIM)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        queries = mx.random.normal((1, 2, 1, HEAD_DIM)).astype(mx.float16)
        out = turboquant_sdpa(queries, cache, scale=1.0 / math.sqrt(HEAD_DIM), mask="causal")
        assert out.shape == queries.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
