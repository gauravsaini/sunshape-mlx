"""Value codecs for hybrid SunShape MLX caches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from sunshape_mlx.kernels import (
    dequantize_values,
    pack_indices,
    quantize_scalar_to_indices,
    quantize_values,
    unpack_indices,
)
from sunshape_mlx.rotation import generate_rotation_matrix


_TURBOQUANT_CODEBOOKS: dict[int, tuple[list[float], list[float]]] = {
    1: (
        [-0.7978845608028654, 0.7978845608028654],
        [0.0],
    ),
    2: (
        [-1.510417608611893, -0.4527800346911237, 0.4527800346911237, 1.510417608611893],
        [-0.9815988216515084, 0.0, 0.9815988216515084],
    ),
    3: (
        [
            -2.151945705166112,
            -1.3439092791423422,
            -0.7560052816730181,
            -0.2450941791152904,
            0.2450941791152904,
            0.7560052816730181,
            1.3439092791423422,
            2.151945705166112,
        ],
        [
            -1.7479274921542272,
            -1.0499572804076802,
            -0.5005497303941542,
            0.0,
            0.5005497303941542,
            1.0499572804076802,
            1.7479274921542272,
        ],
    ),
    4: (
        [
            -2.732896755154294,
            -2.069364258154187,
            -1.618400443227723,
            -1.2565648452462146,
            -0.9426291036999694,
            -0.6569817464411519,
            -0.38818871416000605,
            -0.12844300124876415,
            0.12844300124876415,
            0.38818871416000605,
            0.6569817464411519,
            0.9426291036999694,
            1.2565648452462146,
            1.618400443227723,
            2.069364258154187,
            2.732896755154294,
        ],
        [
            -2.4011305066542405,
            -1.8438823506909552,
            -1.4374826442369688,
            -1.099596974473092,
            -0.7998054250705606,
            -0.522585230300579,
            -0.2583158577043851,
            0.0,
            0.2583158577043851,
            0.522585230300579,
            0.7998054250705606,
            1.099596974473092,
            1.4374826442369688,
            1.8438823506909552,
            2.4011305066542405,
        ],
    ),
}


def _packed_dim(head_dim: int, bits: int) -> int:
    if bits == 2:
        if head_dim % 16 != 0:
            raise ValueError(f"head_dim must be divisible by 16 for 2-bit packing, got {head_dim}")
        return head_dim // 16
    if bits == 3:
        return (head_dim + 9) // 10
    if bits == 4:
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be divisible by 8 for 4-bit packing, got {head_dim}")
        return head_dim // 8
    raise ValueError(f"Unsupported bits: {bits}")


@dataclass
class GroupedValueCodec:
    head_dim: int
    bits: int
    group_size: int = 64
    name: str = "grouped"
    last_quantize_backend: str = "mlx"

    @property
    def packed_dim(self) -> int:
        return _packed_dim(self.head_dim, self.bits)

    @property
    def n_groups(self) -> int:
        return self.head_dim // self.group_size

    def quantize(self, values: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        packed, scales, zeros, _ = quantize_values(values, bits=self.bits, group_size=self.group_size)
        self.last_quantize_backend = "mlx"
        return packed, scales, zeros

    def dequantize(self, packed: mx.array, scales: mx.array, zeros: mx.array) -> mx.array:
        return dequantize_values(
            packed,
            scales,
            zeros,
            D=self.head_dim,
            bits=self.bits,
            group_size=self.group_size,
        )


@dataclass
class TurboQuantValueCodec:
    head_dim: int
    bits: int
    seed: int = 43
    prefer_metal: bool = True
    name: str = "turboquant"
    last_quantize_backend: str = "none"

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4):
            raise ValueError(f"TurboQuant values support 2, 3, or 4 bits, got {self.bits}")
        centroids_list, boundaries_list = _TURBOQUANT_CODEBOOKS[self.bits]
        scale = 1.0 / math.sqrt(self.head_dim)
        self.centroids = mx.array([c * scale for c in centroids_list], dtype=mx.float32)
        self.boundaries = mx.array([b * scale for b in boundaries_list], dtype=mx.float32)
        self.rotation = generate_rotation_matrix(self.head_dim, seed=self.seed).astype(mx.float32)
        self.rotation_t = self.rotation.T

    @property
    def packed_dim(self) -> int:
        return _packed_dim(self.head_dim, self.bits)

    def quantize(self, values: mx.array) -> tuple[mx.array, mx.array]:
        values = values.astype(mx.float32)
        norms = mx.linalg.norm(values, axis=-1, keepdims=True)
        unit = values / mx.maximum(norms, 1e-8)
        rotated = unit @ self.rotation_t
        indices, backend = quantize_scalar_to_indices(
            rotated,
            self.boundaries,
            prefer_metal=self.prefer_metal,
        )
        packed = pack_indices(indices, bits=self.bits)
        self.last_quantize_backend = backend
        return packed, norms.squeeze(-1)

    def dequantize(self, packed: mx.array, norms: mx.array) -> mx.array:
        indices = unpack_indices(packed, D=self.head_dim, bits=self.bits).astype(mx.int32)
        rotated = self.centroids[indices]
        unit = rotated @ self.rotation
        return unit * norms[..., None]


def build_value_codec(
    value_backend: str,
    *,
    head_dim: int,
    value_bits: int,
    value_group_size: int = 64,
    value_seed: int = 43,
    prefer_metal: bool = True,
) -> GroupedValueCodec | TurboQuantValueCodec | None:
    """Build a value codec for compressed V storage."""
    if value_backend == "fp16":
        return None
    if value_backend == "grouped":
        return GroupedValueCodec(head_dim=head_dim, bits=value_bits, group_size=value_group_size)
    if value_backend == "turboquant":
        return TurboQuantValueCodec(
            head_dim=head_dim,
            bits=value_bits,
            seed=value_seed,
            prefer_metal=prefer_metal,
        )
    raise ValueError(f"Unsupported value backend: {value_backend}")
