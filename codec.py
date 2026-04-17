"""SunShape Block Codec — MLX implementation.

Core block vector quantization codec for SunShape on Apple Silicon.
Supports the same modes as the PyTorch version:

- ``profileperm_baseline``: ProfilePerm permutation + plain block VQ
- ``profileperm_localmetric_dsq``: ProfilePerm + local metric + DSQ refinement
- ``profileperm_mixed_precision``: ProfilePerm + high-precision heavy blocks + VQ on light blocks
- ``rotated``: Dense random rotation + block VQ (legacy ablation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from sunshape_mlx.kernels import block_vq_quantize_metal, block_vq_score_metal
from sunshape_mlx.rotation import (
    generate_rotation_matrix,
    covariance_block_permutation,
    invert_permutation,
    apply_permutation,
    block_local_cov_metric,
    block_affinity_gate,
    mixed_precision_block_mask,
)


# ------------------------------------------------------------------ #
#  Quantized representation                                           #
# ------------------------------------------------------------------ #


class SunShapeQuantized(NamedTuple):
    """Quantized key representation from SunShapeBlockCodec."""
    indices: mx.array          # (n_tokens, n_blocks) — centroid indices
    head_dim: int
    block_dim: int
    n_blocks: int
    passthrough_blocks: mx.array | None = None   # (n_blocks,) bool
    passthrough_values: mx.array | None = None    # (n_tokens, n_blocks, block_dim)


# ------------------------------------------------------------------ #
#  K-means in pure MLX                                                #
# ------------------------------------------------------------------ #


def _kmeans(
    data: mx.array,
    n_centroids: int,
    n_iters: int = 25,
    seed: int = 0,
) -> mx.array:
    """Lloyd's k-means over row vectors, pure MLX.

    Parameters
    ----------
    data : mx.array, shape (N, D)
    n_centroids : int
    n_iters : int
    seed : int

    Returns
    -------
    centroids : mx.array, shape (n_centroids, D)
    """
    N, D = data.shape
    if N <= n_centroids:
        out = mx.zeros((n_centroids, D), dtype=data.dtype)
        out[:N] = data
        return out

    # Initialize centroids via random selection (use numpy for permutation)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)[:n_centroids]
    data_np = np.array(data)
    centroids_np = data_np[perm].copy()

    for _ in range(n_iters):
        # Assign each point to nearest centroid
        # dists: (N, n_centroids)
        # Use broadcasting: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x·c
        x_sq = np.sum(data_np ** 2, axis=1, keepdims=True)  # (N, 1)
        c_sq = np.sum(centroids_np ** 2, axis=1, keepdims=True).T  # (1, K)
        dists = x_sq + c_sq - 2.0 * data_np @ centroids_np.T  # (N, K)
        assigns = np.argmin(dists, axis=1)  # (N,)

        # Recompute centroids
        new_centroids = np.zeros_like(centroids_np)
        counts = np.zeros(n_centroids, dtype=np.float64)
        np.add.at(new_centroids, assigns, data_np)
        np.add.at(counts, assigns, 1.0)
        mask = counts > 0
        new_centroids[mask] /= counts[mask, None]
        new_centroids[~mask] = centroids_np[~mask]
        centroids_np = new_centroids

    return mx.array(centroids_np, dtype=mx.float32)


# ------------------------------------------------------------------ #
#  DSQ refinement                                                     #
# ------------------------------------------------------------------ #


def _refine_centroids(
    q_cal: mx.array,
    k_cal: mx.array,
    centroids: mx.array,
    e_metric: mx.array,
    block_dim: int,
    n_steps: int = 3,
    step_size: float = 0.1,
    block_affinity: mx.array | None = None,
    active_blocks: mx.array | None = None,
) -> mx.array:
    """DSQ centroid refinement — attention-aware gradient step.

    Simplified port of sunshape.dsq.refine_centroids_strict.
    Operates in numpy for the iterative loop, returns MLX array.
    """
    head_dim = k_cal.shape[-1]
    n_blocks = head_dim // block_dim
    scale = math.sqrt(head_dim)

    q_sub = np.array(q_cal[:min(128, q_cal.shape[0])].astype(mx.float32))
    k_sub = np.array(k_cal[:min(1024, k_cal.shape[0])].astype(mx.float32))
    e_np = np.array(e_metric.astype(mx.float32))
    refined_np = np.array(centroids.astype(mx.float32)).copy()

    if block_affinity is not None:
        ba_np = np.array(block_affinity.astype(mx.float32))
    else:
        ba_np = np.ones(n_blocks, dtype=np.float32)

    if active_blocks is not None:
        ab_np = np.array(active_blocks.astype(mx.bool_))
    else:
        ab_np = np.ones(n_blocks, dtype=bool)

    # Build initial k_hat
    k_hat = np.zeros_like(k_sub)
    for b in range(n_blocks):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        e_blk = e_np[sl, sl]
        k_shaped = k_sub[:, sl] @ e_blk.T
        c_shaped = refined_np[b] @ e_blk.T
        diff = k_shaped[:, None, :] - c_shaped[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        a = np.argmin(d2, axis=1)
        k_hat[:, sl] = refined_np[b][a]

    for _ in range(n_steps):
        # Attention distributions
        logits_orig = (q_sub @ k_sub.T) / scale
        logits_hat = (q_sub @ k_hat.T) / scale
        p_orig = _softmax(logits_orig)
        p_hat = _softmax(logits_hat)
        err = p_hat - p_orig
        sens = p_hat * (1.0 - p_hat)
        grad_attn = ((err * sens).T @ q_sub) / scale
        total_grad = (k_hat - k_sub) + 0.5 * grad_attn

        for b in range(n_blocks):
            if not ab_np[b]:
                continue
            sl = slice(b * block_dim, (b + 1) * block_dim)
            e_blk = e_np[sl, sl]
            k_shaped = k_sub[:, sl] @ e_blk.T
            c_shaped = refined_np[b] @ e_blk.T
            diff = k_shaped[:, None, :] - c_shaped[None, :, :]
            d2 = np.sum(diff ** 2, axis=-1)
            a = np.argmin(d2, axis=1)

            new_cents = refined_np[b].copy()
            block_step = step_size * float(ba_np[b])
            for ci in range(refined_np.shape[1]):
                mask = a == ci
                if mask.any():
                    new_cents[ci] -= block_step * total_grad[mask, sl].mean(axis=0)
            refined_np[b] = new_cents
            k_hat[:, sl] = new_cents[a]

    return mx.array(refined_np, dtype=mx.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numpy softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ------------------------------------------------------------------ #
#  Compact index dtype                                                #
# ------------------------------------------------------------------ #


def _compact_index_dtype(n_centroids: int):
    """Return the smallest MLX dtype that can hold indices [0, n_centroids)."""
    if n_centroids <= 256:
        return mx.uint8
    if n_centroids <= 65535:
        return mx.uint16
    return mx.uint32


# ------------------------------------------------------------------ #
#  SunShapeBlockCodec                                                 #
# ------------------------------------------------------------------ #


class SunShapeBlockCodec:
    """SunShape block vector quantization codec (MLX).

    Modes
    -----
    - ``profileperm_baseline``: ProfilePerm permutation + plain block VQ
    - ``profileperm_localmetric_dsq``: ProfilePerm + local metric + DSQ
    - ``profileperm_mixed_precision``: ProfilePerm + FP heavy blocks + VQ light blocks
    - ``rotated``: Dense random rotation + block VQ (legacy ablation)
    """

    def __init__(
        self,
        head_dim: int,
        block_dim: int = 8,
        n_centroids: int = 256,
        c: float = 5.0,
        n_refine_dsq: int = 3,
        mode: str = "profileperm_baseline",
        use_rotation: bool = False,
        rotation_seed: int = 0,
    ):
        if head_dim % block_dim != 0:
            raise ValueError("head_dim must be divisible by block_dim")
        self.head_dim = head_dim
        self.block_dim = block_dim
        self.n_blocks = head_dim // block_dim
        self.n_centroids = n_centroids
        self.c = c
        self.mode = "rotated" if use_rotation else mode
        self.n_refine_dsq = n_refine_dsq
        self.use_rotation = self.mode == "rotated"

        # Codec state (MLX arrays — set during fit())
        self.E: mx.array = mx.eye(head_dim, dtype=mx.float32)
        self.E_inv: mx.array = mx.eye(head_dim, dtype=mx.float32)
        self.centroids: mx.array = mx.zeros(
            (self.n_blocks, n_centroids, block_dim), dtype=mx.float32
        )
        self.dsq_block_affinity: mx.array = mx.ones((self.n_blocks,), dtype=mx.float32)
        self.dsq_active_blocks: mx.array = mx.ones((self.n_blocks,), dtype=mx.bool_)
        self.mixed_high_precision_blocks: mx.array = mx.zeros((self.n_blocks,), dtype=mx.bool_)
        self.mixed_block_mass: mx.array = mx.zeros((self.n_blocks,), dtype=mx.float32)
        self.permutation: mx.array = mx.arange(head_dim, dtype=mx.int32)
        self.inv_permutation: mx.array = mx.arange(head_dim, dtype=mx.int32)
        self.last_score_backend = "none"

        if self.use_rotation:
            self.rotation: mx.array = generate_rotation_matrix(head_dim, seed=rotation_seed)
        else:
            self.rotation: mx.array = mx.eye(head_dim, dtype=mx.float32)

    # ------------------------------------------------------------------ #
    #  Transform helpers                                                   #
    # ------------------------------------------------------------------ #

    def _forward_transform(self, x: mx.array) -> mx.array:
        if self.mode in {
            "profileperm_baseline",
            "profileperm_localmetric_dsq",
            "profileperm_mixed_precision",
        }:
            return apply_permutation(x, self.permutation)
        if self.use_rotation:
            return x @ self.rotation.T
        return x

    def _inverse_transform(self, x: mx.array) -> mx.array:
        if self.mode in {
            "profileperm_baseline",
            "profileperm_localmetric_dsq",
            "profileperm_mixed_precision",
        }:
            return apply_permutation(x, self.inv_permutation)
        if self.use_rotation:
            return x @ self.rotation
        return x

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        q_cal: mx.array,
        k_cal: mx.array,
        kmeans_iters: int = 25,
        seed: int = 0,
    ) -> SunShapeBlockCodec:
        """Fit the codec from calibration query/key vectors.

        Parameters
        ----------
        q_cal : mx.array, shape (N, head_dim)
            Calibration queries (post-RoPE).
        k_cal : mx.array, shape (M, head_dim)
            Calibration keys (post-RoPE).
        kmeans_iters : int
            Number of Lloyd iterations per block.
        seed : int
            Random seed for k-means initialization.
        """
        q_cal = q_cal.astype(mx.float32)
        k_cal = k_cal.astype(mx.float32)

        if self.mode == "rotated":
            q_cal = q_cal @ self.rotation.T
            k_cal = k_cal @ self.rotation.T
            self.E = mx.eye(self.head_dim, dtype=mx.float32)
            self.E_inv = mx.eye(self.head_dim, dtype=mx.float32)
            self.permutation = mx.arange(self.head_dim, dtype=mx.int32)

        elif self.mode == "profileperm_baseline":
            perm = covariance_block_permutation(q_cal, self.block_dim)
            inv_perm = invert_permutation(perm)
            self.permutation = perm
            self.inv_permutation = inv_perm
            q_cal = apply_permutation(q_cal, perm)
            k_cal = apply_permutation(k_cal, perm)
            self.E = mx.eye(self.head_dim, dtype=mx.float32)
            self.E_inv = mx.eye(self.head_dim, dtype=mx.float32)

        elif self.mode == "profileperm_localmetric_dsq":
            perm = covariance_block_permutation(q_cal, self.block_dim)
            inv_perm = invert_permutation(perm)
            self.permutation = perm
            self.inv_permutation = inv_perm
            q_cal = apply_permutation(q_cal, perm)
            k_cal = apply_permutation(k_cal, perm)
            e = block_local_cov_metric(q_cal, mx.arange(self.head_dim, dtype=mx.int32), self.block_dim)
            self.E = e
            self.E_inv = mx.linalg.pinv(e)
            dsq_ba, dsq_ab = block_affinity_gate(q_cal, self.block_dim)
            self.dsq_block_affinity = dsq_ba
            self.dsq_active_blocks = dsq_ab

        elif self.mode == "profileperm_mixed_precision":
            perm = covariance_block_permutation(q_cal, self.block_dim)
            inv_perm = invert_permutation(perm)
            self.permutation = perm
            self.inv_permutation = inv_perm
            q_cal = apply_permutation(q_cal, perm)
            k_cal = apply_permutation(k_cal, perm)
            self.E = mx.eye(self.head_dim, dtype=mx.float32)
            self.E_inv = mx.eye(self.head_dim, dtype=mx.float32)
            heavy, bm = mixed_precision_block_mask(q_cal, self.block_dim, mass_threshold=0.5)
            self.mixed_high_precision_blocks = heavy
            self.mixed_block_mass = bm

        else:
            raise ValueError(f"Unknown SunShapeBlockCodec mode: {self.mode}")

        # Reset non-applicable fields
        if self.mode != "profileperm_localmetric_dsq":
            self.dsq_block_affinity = mx.ones((self.n_blocks,), dtype=mx.float32)
            self.dsq_active_blocks = mx.ones((self.n_blocks,), dtype=mx.bool_)
        if self.mode != "profileperm_mixed_precision":
            self.mixed_high_precision_blocks = mx.zeros((self.n_blocks,), dtype=mx.bool_)
            self.mixed_block_mass = mx.zeros((self.n_blocks,), dtype=mx.float32)

        # Per-block k-means
        working_centroids = []
        for b in range(self.n_blocks):
            sl = slice(b * self.block_dim, (b + 1) * self.block_dim)
            if (
                self.mode == "profileperm_mixed_precision"
                and bool(np.array(self.mixed_high_precision_blocks)[b])
            ):
                working_centroids.append(
                    mx.zeros((self.n_centroids, self.block_dim), dtype=mx.float32)
                )
                continue
            e_blk = self.E[sl, sl]
            k_shaped = k_cal[:, sl] @ e_blk.T
            c = _kmeans(k_shaped, self.n_centroids, n_iters=kmeans_iters, seed=seed + b)
            # Map centroids back to original space
            e_inv_blk = self.E_inv[sl, sl]
            c_orig = c @ e_inv_blk.T
            working_centroids.append(c_orig)

        working = mx.stack(working_centroids)

        # DSQ refinement
        if self.mode not in ("profileperm_baseline", "rotated") and self.n_refine_dsq > 0:
            working = _refine_centroids(
                q_cal=q_cal,
                k_cal=k_cal,
                centroids=working,
                e_metric=self.E,
                block_dim=self.block_dim,
                n_steps=self.n_refine_dsq,
                block_affinity=self.dsq_block_affinity,
                active_blocks=self.dsq_active_blocks,
            )

        self.centroids = working
        mx.eval(
            self.E, self.E_inv, self.centroids, self.permutation, self.inv_permutation,
            self.rotation, self.dsq_block_affinity, self.dsq_active_blocks,
            self.mixed_high_precision_blocks, self.mixed_block_mass,
        )
        return self

    # ------------------------------------------------------------------ #
    #  Quantize / Dequantize                                               #
    # ------------------------------------------------------------------ #

    def quantize(self, keys: mx.array) -> SunShapeQuantized:
        """Quantize keys into block centroid indices.

        Parameters
        ----------
        keys : mx.array, shape (N, head_dim)

        Returns
        -------
        SunShapeQuantized with indices shape (N, n_blocks)
        """
        keys = keys.astype(mx.float32)
        keys = self._forward_transform(keys)

        N = keys.shape[0]
        passthrough_values = None

        if (
            self.mode == "profileperm_mixed_precision"
            and bool(np.array(self.mixed_high_precision_blocks).any())
        ):
            passthrough_values = mx.zeros(
                (N, self.n_blocks, self.block_dim), dtype=mx.float32
            )

        # ---- Fast path: vectorized Metal kernel ----
        # Usable when E is identity and no mixed-precision blocks
        e_is_identity = self.mode in {
            "profileperm_baseline", "rotated",
        }
        if e_is_identity and passthrough_values is None:
            metal_indices = block_vq_quantize_metal(
                keys, self.centroids,
                n_blocks=self.n_blocks,
                n_centroids=self.n_centroids,
                block_dim=self.block_dim,
                head_dim=self.head_dim,
            )
            if metal_indices is not None:
                return SunShapeQuantized(
                    indices=metal_indices.astype(_compact_index_dtype(self.n_centroids)),
                    head_dim=self.head_dim,
                    block_dim=self.block_dim,
                    n_blocks=self.n_blocks,
                    passthrough_blocks=None,
                    passthrough_values=None,
                )

        # ---- Fallback: numpy loop (supports E metric + mixed precision) ----
        keys_np = np.array(keys)
        centroids_np = np.array(self.centroids)
        E_np = np.array(self.E)
        indices_np = np.zeros((N, self.n_blocks), dtype=np.int32)

        if passthrough_values is not None:
            ptv_np = np.zeros((N, self.n_blocks, self.block_dim), dtype=np.float32)
        else:
            ptv_np = None

        heavy_np = np.array(self.mixed_high_precision_blocks) if self.mode == "profileperm_mixed_precision" else None

        for b in range(self.n_blocks):
            sl = slice(b * self.block_dim, (b + 1) * self.block_dim)
            if heavy_np is not None and heavy_np[b]:
                if ptv_np is not None:
                    ptv_np[:, b, :] = keys_np[:, sl]
                continue

            e_blk = E_np[sl, sl]
            k_shaped = keys_np[:, sl] @ e_blk.T
            c_shaped = centroids_np[b] @ e_blk.T

            # Compute distances: (N, n_centroids)
            diff = k_shaped[:, None, :] - c_shaped[None, :, :]
            dists = np.sum(diff ** 2, axis=-1)
            indices_np[:, b] = np.argmin(dists, axis=1)

        indices = mx.array(indices_np, dtype=_compact_index_dtype(self.n_centroids))
        if ptv_np is not None:
            passthrough_values = mx.array(ptv_np, dtype=mx.float32)

        return SunShapeQuantized(
            indices=indices,
            head_dim=self.head_dim,
            block_dim=self.block_dim,
            n_blocks=self.n_blocks,
            passthrough_blocks=self.mixed_high_precision_blocks
            if self.mode == "profileperm_mixed_precision"
            else None,
            passthrough_values=passthrough_values,
        )

    def dequantize(self, quantized: SunShapeQuantized) -> mx.array:
        """Reconstruct keys from quantized representation.

        Returns
        -------
        k_hat : mx.array, shape (N, head_dim)
        """
        indices_np = np.array(quantized.indices.astype(mx.int32))
        centroids_np = np.array(self.centroids)
        N = indices_np.shape[0]

        k_hat = np.zeros((N, self.head_dim), dtype=np.float32)

        for b in range(quantized.n_blocks):
            sl = slice(b * self.block_dim, (b + 1) * self.block_dim)
            if (
                quantized.passthrough_blocks is not None
                and bool(np.array(quantized.passthrough_blocks)[b])
            ):
                if quantized.passthrough_values is None:
                    raise ValueError("Mixed-precision missing passthrough values.")
                ptv_np = np.array(quantized.passthrough_values)
                k_hat[:, sl] = ptv_np[:, b, :]
            else:
                k_hat[:, sl] = centroids_np[b][indices_np[:, b]]

        k_hat_mlx = mx.array(k_hat, dtype=mx.float32)
        return self._inverse_transform(k_hat_mlx)

    def __call__(self, keys: mx.array) -> mx.array:
        """Quantize then dequantize (round-trip)."""
        return self.dequantize(self.quantize(keys))

    # ------------------------------------------------------------------ #
    #  Attention scores (precomputed dots)                                 #
    # ------------------------------------------------------------------ #

    def precompute_query_centroid_dots(self, query: mx.array) -> mx.array:
        """Precompute dot(query_block_b, centroid[b, c, :]) for all (b, c).

        This is the key structural advantage of block VQ over scalar quant:
        the block dot products are small and can be precomputed, reducing
        the attention kernel to a pure gather operation.

        Parameters
        ----------
        query : mx.array, shape (n_qh, head_dim)
            Query vectors in the codec's transformed space.

        Returns
        -------
        qdots : mx.array, shape (n_qh, n_blocks, n_centroids)
        """
        n_qh = query.shape[0]
        q_blocks = query.reshape(n_qh, self.n_blocks, self.block_dim)
        # Batch matmul via einsum: qdots[q, b, c] = sum_d q[q, b, d] * centroids[b, c, d]
        qdots = mx.einsum("qbd,bcd->qbc", q_blocks, self.centroids)
        return qdots

    def attention_scores(
        self,
        query: mx.array,
        quantized: SunShapeQuantized,
        *,
        prefer_metal: bool = True,
    ) -> mx.array:
        """Compute attention scores using precomputed query-centroid dots.

        Parameters
        ----------
        query : mx.array, shape (n_qh, head_dim)
            In original model basis.
        quantized : SunShapeQuantized
            Quantized keys.

        Returns
        -------
        scores : mx.array, shape (n_qh, n_tokens)
        """
        query = query.astype(mx.float32)
        query = self._forward_transform(query)

        n_qh = query.shape[0]
        n_tokens = quantized.indices.shape[0]

        # Precompute dot products
        qdots = self.precompute_query_centroid_dots(query)

        if prefer_metal and quantized.passthrough_blocks is None:
            scores = block_vq_score_metal(
                qdots,
                quantized.indices,
                n_qh=n_qh,
                T_kv=n_tokens,
                n_blocks=self.n_blocks,
                n_centroids=self.n_centroids,
            )
            if scores is not None:
                self.last_score_backend = "metal"
                return scores

        # Gather: scores[q, t] = sum_b qdots[q, b, indices[t, b]]
        indices_int = quantized.indices.astype(mx.int32)

        # Use advanced indexing: for each (q, t, b), look up qdots[q, b, indices[t, b]]
        # Reshape for broadcasting
        # qdots: (n_qh, n_blocks, n_centroids)
        # indices: (n_tokens, n_blocks)
        # We want: result[q, t, b] = qdots[q, b, indices[t, b]]
        #          then sum over b -> (n_qh, n_tokens)

        # Expand dims for broadcasting
        qdots_exp = qdots[:, None, :, :]  # (n_qh, 1, n_blocks, n_centroids)
        indices_exp = indices_int[None, :, :, None]  # (1, n_tokens, n_blocks, 1)

        # Use take along axis on the last dimension
        gathered = mx.take_along_axis(qdots_exp, indices_exp, axis=-1)  # (n_qh, n_tokens, n_blocks, 1)
        gathered = gathered.squeeze(-1)  # (n_qh, n_tokens, n_blocks)
        scores = mx.sum(gathered, axis=2)  # (n_qh, n_tokens)

        self.last_score_backend = "mlx"
        return scores

    def heldout_logit_mse(
        self,
        q_test: mx.array,
        k_test: mx.array,
    ) -> float:
        """Compute held-out logit MSE: E[(q · (k - k_hat))²]."""
        q_test = q_test.astype(mx.float32)
        k_test = k_test.astype(mx.float32)
        k_hat = self(k_test)
        delta = k_test - k_hat
        logit_err = mx.sum(q_test * delta, axis=1) ** 2
        return float(mx.mean(logit_err).item())
