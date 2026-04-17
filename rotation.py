"""Permutation and rotation utilities for SunShape on MLX.

Provides:
- Random orthogonal rotation via QR decomposition (matching TurboQuant)
- Randomized Walsh-Hadamard Transform (WHT) — O(d log d) rotation
- Covariance-aware block permutation (ProfilePerm)
- Block-local covariance metric construction
- Block affinity gate and mixed-precision masks

The WHT rotation is adapted from turboQuantPlayground and provides an
O(d log d) alternative to the O(d²) QR rotation for high-dimensional
embeddings.
"""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import numpy as np


# ================================================================== #
#  Random rotation via QR decomposition                                #
# ================================================================== #


def generate_rotation_matrix(d: int, seed: int = 0) -> mx.array:
    """Generate a random orthogonal rotation matrix via QR decomposition.

    Same construction as TurboQuant: guarantees a uniform Haar-random
    rotation so that each coordinate of the rotated vector follows a
    Beta(d/2, d/2) distribution.

    The QR decomposition is computed on the CPU for numerical stability.
    """
    mx.random.seed(seed)
    G = mx.random.normal((d, d))
    mx.eval(G)

    # QR on CPU for precision
    Q, R = mx.linalg.qr(G, stream=mx.cpu)
    mx.eval(Q, R)

    # Ensure det(Q) = +1 (proper rotation, not reflection)
    diag_sign = mx.sign(mx.diag(R))
    Q = Q * diag_sign[None, :]
    mx.eval(Q)
    return Q


# ================================================================== #
#  Walsh-Hadamard Transform (WHT) rotation — O(d log d)              #
# ================================================================== #
# Adapted from turboQuantPlayground's rotation.py.
# The WHT provides a fast O(d log d) alternative to the O(d²) QR rotation.
# R = (1/√d) · H_d · diag(signs)  where H_d is the Hadamard matrix.


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def generate_wht_signs(d: int, seed: int = 0) -> np.ndarray:
    """Generate random ±1 signs for the randomized WHT.

    Parameters
    ----------
    d : int
        Dimension (must be a power of 2).
    seed : int
        Random seed.

    Returns
    -------
    signs : np.ndarray, shape (d,), float32
        Random signs in {-1, +1}.
    """
    if not _is_power_of_2(d):
        raise ValueError(f"WHT requires d to be a power of 2, got {d}")
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=d).astype(np.float32)


def apply_wht(
    x: mx.array,
    signs: np.ndarray | mx.array,
    inverse: bool = False,
) -> mx.array:
    """Apply the randomized Walsh-Hadamard Transform.

    Computes R · x  where R = (1/√d) · H_d · diag(signs).
    Uses the O(d log d) butterfly factorization instead of
    materializing the full d×d Hadamard matrix.

    Parameters
    ----------
    x : mx.array, shape (..., d)
        Input vectors.  Last dimension must be a power of 2.
    signs : np.ndarray or mx.array, shape (d,)
        Random ±1 signs.
    inverse : bool
        If True, compute x · R  (inverse transform).

    Returns
    -------
    y : mx.array, shape (..., d)
        Rotated vectors.
    """
    d = x.shape[-1]
    if not _is_power_of_2(d):
        raise ValueError(f"WHT requires last dim to be a power of 2, got {d}")

    # Convert signs to MLX if needed
    if isinstance(signs, np.ndarray):
        signs_mx = mx.array(signs, dtype=mx.float32)
    else:
        signs_mx = signs.astype(mx.float32)

    scale = 1.0 / math.sqrt(d)

    if inverse:
        # Inverse: apply butterfly first, then multiply by signs
        result = _wht_butterfly(x)
        result = result * signs_mx * scale
    else:
        # Forward: multiply by signs first, then butterfly
        result = x * signs_mx
        result = _wht_butterfly(result) * scale

    mx.eval(result)
    return result


def _wht_butterfly(x: mx.array) -> mx.array:
    """Apply the Walsh-Hadamard butterfly factorization.

    Implements the O(d log d) fast WHT using the iterative butterfly
    pattern.  Operates on the last dimension of x.

    H_d = (H_2 ⊗ I_{d/2}) · ... · (I_{d/2} ⊗ H_2)
    Each stage splits into pairs and applies [a+b, a-b].
    """
    d = x.shape[-1]
    # Work in numpy for the iterative butterfly (Python loop over log2(d) stages)
    # This is still fast because d is typically 64-256 and log2(d) <= 8.
    result = np.array(x)
    orig_shape = result.shape
    # Flatten all dims except last
    result = result.reshape(-1, d)

    h = 1
    while h < d:
        # Reshape into (N, d/(2h), 2, h) blocks
        result = result.reshape(-1, d // (2 * h), 2, h)
        a = result[:, :, 0, :].copy()
        b = result[:, :, 1, :].copy()
        result[:, :, 0, :] = a + b
        result[:, :, 1, :] = a - b
        result = result.reshape(-1, d)
        h *= 2

    result = result.reshape(*orig_shape)
    return mx.array(result, dtype=mx.float32)


def wht_rotation_mode() -> str:
    """Return the WHT rotation mode identifier for codec configuration."""
    return "wht"


# ------------------------------------------------------------------ #
#  ProfilePerm — covariance-aware block permutation                   #
# ------------------------------------------------------------------ #


def _covariance(x: mx.array) -> mx.array:
    """Empirical covariance of row vectors."""
    centered = x - mx.mean(x, axis=0, keepdims=True)
    n = max(1, x.shape[0] - 1)
    return (centered.T @ centered) / n


def covariance_block_permutation(
    q_cal: mx.array,
    block_dim: int,
) -> mx.array:
    """Greedy affinity-based block packing (ProfilePerm).

    Groups high-covariance dimensions into blocks so that block-diagonal
    VQ retains as much energy as possible within each block.

    Parameters
    ----------
    q_cal : mx.array, shape (N, d)
        Calibration query vectors.
    block_dim : int
        Target block size.

    Returns
    -------
    perm : mx.array, shape (d,), int32
        Permutation that packs correlated dims together.
    """
    cov = _covariance(q_cal)
    mx.eval(cov)
    d = cov.shape[0]
    affinity = mx.abs(cov)
    # Zero out diagonal so self-affinity doesn't dominate
    affinity = affinity - mx.diag(mx.diag(affinity)) * mx.eye(d)
    mx.eval(affinity)

    remaining: set[int] = set(range(d))
    ordered: list[int] = []

    # Eagerly compute numpy version of affinity for the greedy loop
    # (Python control-flow over MLX arrays is expensive in the loop)
    aff_np = np_from_mlx(affinity)

    while remaining:
        rem_list = sorted(remaining)
        if len(rem_list) <= block_dim:
            ordered.extend(rem_list)
            break

        # Pick seed: dimension with highest total affinity
        row_scores = aff_np[rem_list].sum(axis=1)
        seed_idx = int(row_scores.argmax())
        seed = rem_list[seed_idx]
        block = [seed]
        remaining.remove(seed)

        while len(block) < block_dim and remaining:
            cand_list = sorted(remaining)
            # Sum affinity from each candidate to all current block members
            block_np = np.array(block)
            cand_np = np.array(cand_list)
            gains = aff_np[cand_np][:, block_np].sum(axis=1)
            best_idx = int(gains.argmax())
            best = cand_list[best_idx]
            block.append(best)
            remaining.remove(best)

        ordered.extend(block)

    return mx.array(ordered, dtype=mx.int32)


def invert_permutation(perm: mx.array) -> mx.array:
    """Compute the inverse of a permutation."""
    d = perm.shape[0]
    inv = mx.zeros((d,), dtype=perm.dtype)
    # Use numpy for simplicity since this is offline
    perm_np = np_from_mlx(perm)
    inv_np = np.empty_like(perm_np)
    inv_np[perm_np] = np.arange(d, dtype=perm_np.dtype)
    return mx.array(inv_np, dtype=perm.dtype)


def apply_permutation(x: mx.array, perm: mx.array | None) -> mx.array:
    """Apply a permutation to the last dimension of x."""
    if perm is None or perm.size == 0:
        return x
    return x[..., perm]


# ------------------------------------------------------------------ #
#  Block-local covariance metric                                      #
# ------------------------------------------------------------------ #


def block_local_cov_metric(
    q_cal: mx.array,
    perm: mx.array,
    block_dim: int,
) -> mx.array:
    """Build a block-diagonal covariance metric from permuted queries.

    The metric E is block-diagonal with each block = covariance of
    the corresponding permuted dimensions.  This is the local metric
    used by ``profileperm_localmetric_dsq`` mode.

    Note: MLX arrays are immutable, so we build the result in numpy
    and convert back to MLX.
    """
    q_perm = apply_permutation(q_cal, perm)
    cov = _covariance(q_perm)
    mx.eval(cov)

    d = cov.shape[0]
    n_blocks = d // block_dim

    # Build block-diagonal metric in numpy (MLX arrays are immutable)
    cov_np = np_from_mlx(cov)
    local_metric_np = np.zeros((d, d), dtype=np.float32)

    for b in range(n_blocks):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        local_metric_np[sl, sl] = cov_np[sl, sl]

    return mx.array(local_metric_np, dtype=mx.float32)


# ------------------------------------------------------------------ #
#  Block affinity gate (for DSQ)                                      #
# ------------------------------------------------------------------ #


def block_affinity_gate(
    q_cal: mx.array,
    block_dim: int,
) -> tuple[mx.array, mx.array]:
    """Compute per-block affinity weights and active-block mask.

    Returns
    -------
    block_affinity : mx.array, shape (n_blocks,)
        Normalized affinity weight per block.
    active : mx.array, shape (n_blocks,), bool
        Which blocks carry >= 50% of total variance mass.
    """
    centered = q_cal.astype(mx.float32) - mx.mean(q_cal.astype(mx.float32), axis=0, keepdims=True)
    per_dim_var = mx.mean(mx.square(centered), axis=0)
    mx.eval(per_dim_var)

    d = per_dim_var.shape[0]
    n_blocks = d // block_dim
    block_mass = per_dim_var.reshape(n_blocks, block_dim).sum(axis=1)
    mx.eval(block_mass)

    total_mass = mx.sum(block_mass)
    block_affinity = block_mass / mx.maximum(total_mass, 1e-12)
    mx.eval(block_affinity)

    # Top-k blocks covering 50% of mass
    aff_np = np_from_mlx(block_affinity)
    ranked = np.argsort(aff_np)[::-1]
    cumulative = np.cumsum(aff_np[ranked])
    cutoff = int((cumulative < 0.5).sum()) + 1
    cutoff = max(1, min(cutoff, n_blocks))

    active_np = np.zeros(n_blocks, dtype=bool)
    active_np[ranked[:cutoff]] = True
    active = mx.array(active_np)

    # Normalize affinity so max = 1.0
    active_affinity = block_affinity * active.astype(mx.float32)
    max_aff = mx.max(active_affinity)
    active_affinity = active_affinity / mx.maximum(max_aff, 1e-12)
    mx.eval(active_affinity)
    return active_affinity, active


# ------------------------------------------------------------------ #
#  Mixed precision block mask                                         #
# ------------------------------------------------------------------ #


def positive_excess_kurtosis(q: mx.array) -> mx.array:
    """Per-dimension positive excess kurtosis (heavy-tail indicator)."""
    centered = q.astype(mx.float32) - mx.mean(q.astype(mx.float32), axis=0, keepdims=True)
    var = mx.mean(mx.square(centered), axis=0)
    var = mx.maximum(var, 1e-12)
    fourth = mx.mean(centered ** 4, axis=0)
    excess = fourth / mx.square(var) - 3.0
    mx.eval(excess)
    return mx.maximum(excess, mx.zeros_like(excess))


def mixed_precision_block_mask(
    q_cal: mx.array,
    block_dim: int,
    mass_threshold: float = 0.5,
) -> tuple[mx.array, mx.array]:
    """Identify heavy-tailed blocks for mixed-precision mode.

    Returns
    -------
    heavy : mx.array, shape (n_blocks,), bool
        Which blocks should be stored at high precision.
    block_mass : mx.array, shape (n_blocks,)
        Per-block kurtosis mass.
    """
    kurt = positive_excess_kurtosis(q_cal)
    d = kurt.shape[0]
    n_blocks = d // block_dim
    block_mass = kurt.reshape(n_blocks, block_dim).sum(axis=1)
    mx.eval(block_mass)

    total_mass = mx.sum(block_mass)
    block_mass_normed = block_mass / mx.maximum(total_mass, 1e-12)
    mx.eval(block_mass_normed)

    bm_np = np_from_mlx(block_mass_normed)
    ranked = np.argsort(bm_np)[::-1]
    cumulative = np.cumsum(bm_np[ranked])
    cutoff = int((cumulative < mass_threshold).sum()) + 1
    cutoff = max(1, min(cutoff, n_blocks))

    heavy_np = np.zeros(n_blocks, dtype=bool)
    heavy_np[ranked[:cutoff]] = True
    return mx.array(heavy_np), block_mass


# ------------------------------------------------------------------ #
#  Utilities                                                          #
# ------------------------------------------------------------------ #


def safe_normalize(
    x: mx.array,
    axis: int = -1,
    eps: float = 1e-8,
) -> tuple[mx.array, mx.array]:
    """Normalize vectors to unit length, safe for zero vectors."""
    norms = mx.linalg.norm(x, axis=axis, keepdims=True)
    safe_norms = mx.where(norms < eps, mx.ones_like(norms), norms)
    return x / safe_norms, norms


def np_from_mlx(arr: mx.array) -> np.ndarray:
    """Convert an MLX array to numpy (materializes on CPU)."""
    return np.array(arr)
