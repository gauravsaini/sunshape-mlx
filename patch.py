"""Monkey-patch for mlx-lm's SDPA dispatch.

Replaces ``mlx_lm.models.base.scaled_dot_product_attention`` with a
router that checks the cache type and delegates to SunShape's
precomputed-dot attention when a ``SunShapeKVCache`` is detected,
falling back to the original SDPA otherwise.

Usage
-----
>>> from sunshape_mlx.patch import apply, revert
>>> apply()     # install the patch
>>> ...         # run inference with SunShapeKVCache
>>> revert()    # remove the patch
"""

from __future__ import annotations

import importlib

import mlx.core as mx

from sunshape_mlx.attention import sunshape_sdpa
from sunshape_mlx.cache import HybridSunShapeKVCache, SunShapeKVCache
from sunshape_mlx.turboquant_runtime import TurboQuantKVCache, turboquant_sdpa

# Try to import mlx-lm; if unavailable, patching will be a no-op
try:
    import mlx_lm.models.base as _base
    _original_sdpa = _base.scaled_dot_product_attention
    _HAS_MLX_LM = True
except ImportError:
    _base = None
    _original_sdpa = None
    _HAS_MLX_LM = False

_patched = False
_module_sdpa_refs: dict[str, object] = {}


def _patch_module_sdpa(fn) -> None:
    """Patch module-local SDPA aliases used by some mlx-lm model files."""
    global _module_sdpa_refs
    module_names = [
        "mlx_lm.models.qwen3",
        "mlx_lm.models.qwen3_next",
        "mlx_lm.models.qwen3_moe",
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "scaled_dot_product_attention"):
            if module_name not in _module_sdpa_refs:
                _module_sdpa_refs[module_name] = module.scaled_dot_product_attention
            module.scaled_dot_product_attention = fn


def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """SDPA router: SunShape cache → custom SDPA, else → original.

    Matches mlx-lm's signature:
        scaled_dot_product_attention(queries, keys, values, cache, scale, mask, sinks=None)
    """
    if isinstance(cache, (SunShapeKVCache, HybridSunShapeKVCache)):
        return sunshape_sdpa(queries, cache, scale, mask)
    if isinstance(cache, TurboQuantKVCache):
        return turboquant_sdpa(queries, cache, scale, mask)
    if _original_sdpa is not None:
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks=sinks)
    raise RuntimeError(
        "mlx-lm is not installed and no fallback SDPA is available. "
        "Install mlx-lm or provide a custom attention function."
    )


def apply() -> None:
    """Activate the SunShape SDPA patch.  Idempotent."""
    global _patched
    if _patched:
        return
    if not _HAS_MLX_LM:
        raise RuntimeError(
            "mlx-lm is not installed.  Install it with: pip install mlx-lm"
        )
    _base.scaled_dot_product_attention = _patched_sdpa
    _patch_module_sdpa(_patched_sdpa)
    _patched = True


def revert() -> None:
    """Remove the SunShape SDPA patch."""
    global _patched
    if not _patched:
        return
    if _HAS_MLX_LM and _original_sdpa is not None:
        _base.scaled_dot_product_attention = _original_sdpa
    for module_name, original in _module_sdpa_refs.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        module.scaled_dot_product_attention = original
    _patched = False


def is_patched() -> bool:
    """Return whether the SunShape SDPA patch is currently active."""
    return _patched
