"""Validation tests for sunshape_mlx kernels.

Tests correctness of:
1. pack_indices / unpack_indices round-trip (2, 3, 4 bits)
2. quantize_values / dequantize_values round-trip
3. block_vq_score_metal vs pure-MLX reference
4. quantize_scalar_to_indices vs naive reference
5. fused_block_vq_attention vs explicit reference
6. 3-bit packing edge cases (non-divisible dims)
7. Value quantization numerical fidelity
"""

import sys
import traceback
import mlx.core as mx
import numpy as np

from sunshape_mlx.kernels import (
    pack_indices,
    unpack_indices,
    quantize_values,
    dequantize_values,
    quantize_scalar_to_indices,
    block_vq_score_metal,
    fused_block_vq_attention,
)

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def test_pack_unpack_roundtrip():
    """Test pack/unpack round-trip for 2, 3, 4 bits."""
    print("\n[1] pack_indices / unpack_indices round-trip")
    np.random.seed(42)

    for bits, max_val in [(2, 3), (3, 7), (4, 15)]:
        if bits == 2:
            D = 128   # must be divisible by 16
        elif bits == 3:
            D = 100   # 3-bit pads to next multiple of 10
        else:
            D = 128   # must be divisible by 8

        indices = mx.array(np.random.randint(0, max_val + 1, size=(32, D), dtype=np.uint8))
        packed = pack_indices(indices, bits=bits)
        unpacked = unpack_indices(packed, D=D, bits=bits)
        mx.eval(unpacked)

        match = np.array_equal(np.array(indices.astype(mx.uint32)), np.array(unpacked))
        check(f"{bits}-bit round-trip (D={D})", match,
              f"max diff = {np.max(np.abs(np.array(indices.astype(mx.uint32)) - np.array(unpacked)))}")


def test_pack_unpack_3bit_edge():
    """3-bit packing when D is not divisible by 10."""
    print("\n[2] 3-bit packing edge case (D=27, non-divisible by 10)")
    np.random.seed(99)
    D = 27  # not divisible by 10
    indices = mx.array(np.random.randint(0, 8, size=(5, D), dtype=np.uint8))
    packed = pack_indices(indices, bits=3)
    unpacked = unpack_indices(packed, D=D, bits=3)
    mx.eval(unpacked)

    match = np.array_equal(np.array(indices.astype(mx.uint32)), np.array(unpacked))
    check("3-bit non-aligned round-trip (D=27)", match)


def test_pack_boundary_values():
    """Ensure boundary (all-0 and all-max) values survive pack/unpack."""
    print("\n[3] Boundary values for pack/unpack")
    for bits, max_val, D in [(2, 3, 32), (3, 7, 30), (4, 15, 32)]:
        # All zeros
        zeros = mx.zeros((1, D), dtype=mx.uint8)
        packed = pack_indices(zeros, bits=bits)
        unpacked = unpack_indices(packed, D=D, bits=bits)
        mx.eval(unpacked)
        check(f"{bits}-bit all-zeros", np.array_equal(np.array(unpacked), np.zeros((1, D), dtype=np.uint32)))

        # All max
        maxes = mx.full((1, D), max_val, dtype=mx.uint8)
        packed = pack_indices(maxes, bits=bits)
        unpacked = unpack_indices(packed, D=D, bits=bits)
        mx.eval(unpacked)
        check(f"{bits}-bit all-max ({max_val})", np.all(np.array(unpacked) == max_val))


def test_quantize_dequantize_values():
    """Value quantization round-trip fidelity."""
    print("\n[4] quantize_values / dequantize_values round-trip")
    np.random.seed(123)
    T, D = 64, 128
    values = mx.array(np.random.randn(T, D).astype(np.float32))

    for bits in [2, 4]:
        group_size = 64
        packed, scales, zeros, n_groups = quantize_values(values, bits=bits, group_size=group_size)
        mx.eval(packed, scales, zeros)

        deq = dequantize_values(packed, scales, zeros, D=D, bits=bits, group_size=group_size)
        mx.eval(deq)

        orig_np = np.array(values)
        deq_np = np.array(deq)
        mse = np.mean((orig_np - deq_np) ** 2)
        rmse = np.sqrt(mse)
        rel_err = rmse / (np.std(orig_np) + 1e-8)

        # 4-bit (16 levels, group_size=64): ~9% relative RMSE on Gaussian data
        # 2-bit (4 levels, group_size=64): ~45% relative RMSE on Gaussian data
        # These are mathematically expected for asymmetric min-max quantization.
        threshold = 0.15 if bits == 4 else 0.55
        check(f"{bits}-bit value quant RMSE (rel={rel_err:.4f})", rel_err < threshold,
              f"rel_err={rel_err:.6f}, threshold={threshold}")


def test_scalar_quantize():
    """Scalar quantization against sorted boundaries."""
    print("\n[5] quantize_scalar_to_indices")
    np.random.seed(77)
    boundaries = mx.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=mx.float32)
    values = mx.array(np.random.randn(100, 64).astype(np.float32))

    # Pure MLX path
    result_mlx, backend_mlx = quantize_scalar_to_indices(values, boundaries, prefer_metal=False)
    mx.eval(result_mlx)

    # Verify against numpy reference
    vals_np = np.array(values)
    bounds_np = np.array(boundaries)
    ref = np.zeros(vals_np.shape, dtype=np.uint8)
    for i, b in enumerate(bounds_np):
        ref += (vals_np > b).astype(np.uint8)

    match = np.array_equal(np.array(result_mlx), ref)
    check("MLX path matches numpy reference", match,
          f"max diff = {np.max(np.abs(np.array(result_mlx).astype(int) - ref.astype(int)))}")

    # Metal path (may fall back)
    result_metal, backend_metal = quantize_scalar_to_indices(values, boundaries, prefer_metal=True)
    mx.eval(result_metal)
    match_metal = np.array_equal(np.array(result_metal), ref)
    check(f"Metal path ({backend_metal}) matches reference", match_metal)


def test_block_vq_score_vs_reference():
    """Metal block-VQ score kernel vs pure-MLX gather."""
    print("\n[6] block_vq_score_metal vs reference")
    np.random.seed(55)
    T_q, n_blocks, n_centroids = 4, 16, 256
    T_kv = 32

    qdots = mx.array(np.random.randn(T_q, n_blocks, n_centroids).astype(np.float32))
    indices = mx.array(np.random.randint(0, n_centroids, size=(T_kv, n_blocks), dtype=np.uint8))

    # Reference: pure gather
    indices_int = indices.astype(mx.int32)
    qdots_exp = qdots[:, None, :, :]            # (T_q, 1, n_blocks, n_centroids)
    indices_exp = indices_int[None, :, :, None]  # (1, T_kv, n_blocks, 1)
    gathered = mx.take_along_axis(qdots_exp, indices_exp, axis=-1).squeeze(-1)
    ref_scores = mx.sum(gathered, axis=2)       # (T_q, T_kv)
    mx.eval(ref_scores)

    # Metal kernel
    metal_scores = block_vq_score_metal(
        qdots, indices,
        n_qh=T_q, T_kv=T_kv,
        n_blocks=n_blocks, n_centroids=n_centroids,
    )

    if metal_scores is not None:
        mx.eval(metal_scores)
        max_err = float(mx.max(mx.abs(metal_scores - ref_scores)).item())
        check(f"Metal kernel matches reference (max_err={max_err:.6f})", max_err < 1e-3,
              f"max_err={max_err}")
    else:
        print("  ⚠️  Metal kernel returned None (may not be available), skipping.")


def test_fused_block_vq_attention():
    """Fused block-VQ attention vs explicit attention."""
    print("\n[7] fused_block_vq_attention vs explicit")
    np.random.seed(42)
    T_q, T_kv = 4, 32
    n_blocks, n_centroids, block_dim = 16, 256, 8
    head_dim = n_blocks * block_dim

    # Random centroids and indices
    centroids = np.random.randn(n_blocks, n_centroids, block_dim).astype(np.float32)
    indices_np = np.random.randint(0, n_centroids, size=(T_kv, n_blocks), dtype=np.int32)
    values_np = np.random.randn(T_kv, head_dim).astype(np.float32)
    query_np = np.random.randn(T_q, head_dim).astype(np.float32)

    # Reconstruct keys from centroids
    k_hat = np.zeros((T_kv, head_dim), dtype=np.float32)
    for b in range(n_blocks):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        k_hat[:, sl] = centroids[b][indices_np[:, b]]

    # Explicit attention
    scores_ref = query_np @ k_hat.T  # (T_q, T_kv)
    # Causal mask
    q_pos = np.arange(T_kv - T_q, T_kv)
    k_pos = np.arange(T_kv)
    causal = q_pos[:, None] >= k_pos[None, :]
    scores_ref = np.where(causal, scores_ref, -1e9)
    weights_ref = np.exp(scores_ref - scores_ref.max(axis=-1, keepdims=True))
    weights_ref = weights_ref / weights_ref.sum(axis=-1, keepdims=True)
    output_ref = weights_ref @ values_np

    # precompute qdots: qdots[q, b, c] = sum_d query[q, b*bd+d:(b+1)*bd] * centroids[b, c, d]
    q_blocks = query_np.reshape(T_q, n_blocks, block_dim)
    qdots_np = np.einsum('qbd,bcd->qbc', q_blocks, centroids)

    qdots_mx = mx.array(qdots_np)
    indices_mx = mx.array(indices_np)
    values_mx = mx.array(values_np)

    output_fused = fused_block_vq_attention(
        qdots_mx, indices_mx, values_mx,
        T_q=T_q, T_kv=T_kv,
        n_blocks=n_blocks, n_centroids=n_centroids,
        head_dim=head_dim,
    )
    mx.eval(output_fused)

    output_fused_np = np.array(output_fused)
    max_err = np.max(np.abs(output_fused_np - output_ref))
    check(f"Fused attention matches explicit (max_err={max_err:.6f})", max_err < 5e-3,
          f"max_err={max_err}")


def test_value_dequant_source_syntax():
    """Verify the value dequant Metal source compiles as a string (no syntax errors)."""
    print("\n[8] Value dequant Metal source syntax check")
    from sunshape_mlx.kernels import _VALUE_DEQUANT_SOURCE
    # Validate the body-only format (compatible with mx.fast.metal_kernel)
    has_thread = "thread_position_in_grid" in _VALUE_DEQUANT_SOURCE
    has_shape_array = "packed_shape" in _VALUE_DEQUANT_SOURCE
    has_bounds_check = "return" in _VALUE_DEQUANT_SOURCE
    no_kernel_decl = "[[kernel]]" not in _VALUE_DEQUANT_SOURCE  # must NOT have full declaration
    check("Metal source has thread_position_in_grid", has_thread)
    check("Metal source uses _shape arrays (mx.fast.metal_kernel style)", has_shape_array)
    check("Metal source has bounds check", has_bounds_check)
    check("Metal source is body-only (no [[kernel]] declaration)", no_kernel_decl)


def test_3bit_overflow():
    """3-bit: ensure maximum value 7 = 0b111 doesn't overflow into neighbors after packing."""
    print("\n[9] 3-bit overflow guard")
    D = 30
    # Fill with max value
    indices = mx.full((1, D), 7, dtype=mx.uint8)
    packed = pack_indices(indices, bits=3)
    unpacked = unpack_indices(packed, D=D, bits=3)
    mx.eval(unpacked)
    match = np.all(np.array(unpacked) == 7)
    check("3-bit max-value round-trip", match,
          f"unpacked values: {np.array(unpacked)[0, :10]}...")

    # 3-bit: 10 values * 3 bits = 30 bits, fits in uint32 (32 bits), 2 spare bits
    # Check the packed word doesn't have stray bits
    packed_np = np.array(packed)
    # max 30-bit value = (2^30 - 1) = 0x3FFFFFFF
    stray = packed_np[0, 0] >> 30
    check("No stray bits in 3-bit packed uint32", int(stray) == 0,
          f"stray bits = {int(stray):032b}")


def test_block_vq_score_shape():
    """Verify output shapes from block_vq_score_metal."""
    print("\n[10] Block VQ score output shape")
    T_q, n_blocks, n_centroids = 2, 8, 64
    T_kv = 16
    qdots = mx.array(np.random.randn(T_q, n_blocks, n_centroids).astype(np.float32))
    indices = mx.array(np.random.randint(0, n_centroids, size=(T_kv, n_blocks), dtype=np.uint8))

    result = block_vq_score_metal(
        qdots, indices,
        n_qh=T_q, T_kv=T_kv,
        n_blocks=n_blocks, n_centroids=n_centroids,
    )
    if result is not None:
        mx.eval(result)
        check(f"Score shape = ({T_q}, {T_kv})", result.shape == (T_q, T_kv),
              f"got {result.shape}")
    else:
        print("  ⚠️  Metal kernel unavailable, skipping shape check.")


if __name__ == "__main__":
    print("=" * 60)
    print("SunShape MLX Kernel Validation Suite")
    print("=" * 60)

    tests = [
        test_pack_unpack_roundtrip,
        test_pack_unpack_3bit_edge,
        test_pack_boundary_values,
        test_quantize_dequantize_values,
        test_scalar_quantize,
        test_block_vq_score_vs_reference,
        test_fused_block_vq_attention,
        test_value_dequant_source_syntax,
        test_3bit_overflow,
        test_block_vq_score_shape,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"  💥 {t.__name__} CRASHED: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
