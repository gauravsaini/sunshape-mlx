---
name: mlx_metal_kernel_optimization
description: >
  Systematic guidance for writing, reviewing, and optimizing Metal kernels in MLX
  on Apple Silicon. Auto-triggers when: writing mx.fast.metal_kernel source, profiling
  or reviewing GPU kernel performance in MLX, designing thread models for reduction or
  attention operations, handling quantized data in kernels, or porting CUDA kernels to
  MLX. Encodes both MLX API constraints and Apple Silicon GPU architecture principles
  that are frequently violated when applying generic GPU programming patterns without
  questioning the underlying hardware realities.
references:
  - https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  - https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
  - https://developer.apple.com/videos/play/wwdc2022/10159/
  - https://developer.apple.com/videos/play/tech-talks/111373/
---

# MLX Metal Kernel Optimization Skill

## Apple Silicon GPU Architecture Primer

Before writing any kernel, internalize the memory hierarchy your code actually runs on.
Apple Silicon's architecture differs from NVIDIA CUDA in important ways that invalidate
many standard GPU patterns.

### Memory Hierarchy (fastest → slowest)

| Level | Scope | Latency | Size | Notes |
|---|---|---|---|---|
| Registers | Per-thread | ~1 cycle | 128 × 32-bit/SIMD-group | Spill → device memory |
| Threadgroup (SRAM) | Per-threadgroup | ~1–4 cycles | 32 KB max | Apple reduces bandwidth vs NVIDIA; prefer SIMD shuffles for small reductions |
| L1 cache | Per-shader-core | ~10 cycles | ~8 KB | Buffers via `constant` space get prefetching |
| L2 / SLC | Chip-wide | ~100 cycles | Varies by SoC | Larger SLC than NVIDIA peers; UMA means no CPU→GPU copy |
| Unified (device) memory | All | ~100–500 cycles | Full RAM | Shared CPU/GPU — no explicit transfer, but access still costs bandwidth |

**Key Apple Silicon differences from CUDA:**
- No separate device VRAM: CPU and GPU share the same physical memory (UMA). This eliminates transfer overhead but means both compete for the same bandwidth.
- Threadgroup memory bandwidth is *lower* than NVIDIA's shared memory. Apple compensates with excellent SIMD-shuffle bandwidth and `simdgroup_matrix` instructions.
- SIMD-group width = **32 threads** on all Apple GPUs.
- Threadgroup limit: **1024 threads**, 32 KB shared memory.
- For Apple Family 9+ GPUs (M4+): `device` and `threadgroup` buffers share the same cache hierarchy when the working set fits in cache — profile before blindly copying to threadgroup memory.

---

## Phase 0: Check `mx.fast` Before Writing a Kernel

MLX ships optimized primitives in `mx.fast`. Always check these first:

```python
# These are often faster than custom kernels because Anthropic hand-tuned them
mx.fast.rms_norm(x, weight, eps)
mx.fast.layer_norm(x, weight, bias, eps)
mx.fast.rope(x, dims, traditional, base, scale, offset)
mx.fast.scaled_dot_product_attention(q, k, v, scale, mask)

# Quantized matmul (handles dequant internally, very fast)
mx.quantized_matmul(x, w_quantized, scales, biases, bits=4, group_size=64)

# mx.compile for fusion without a custom kernel
@mx.compile
def fused_op(x, w):
    return mx.relu(x @ w)
```

**Only write a custom Metal kernel when:**
1. The operation is not covered by `mx.fast` or standard MLX ops, AND
2. `mx.compile` cannot fuse the bottleneck (e.g., it requires in-kernel branching or quantized data formats not exposed to `mx.compile`), OR
3. Profiling confirms the standard path is bandwidth/compute bound in a way a custom kernel can fix.

---

## Phase 1: Data Flow Audit

**The most common source of performance bugs is not the kernel itself — it is what
happens to the data before the kernel is called.**

### 1.1 Trace Every Input from Storage to Kernel Parameter

For each kernel input, draw the chain:
```
storage format → Python property / method → mx.eval() → kernel parameter
```

At every stage, ask: **does data expand here?**

Common expansions to catch:
- Packed int (4-bit, 2-bit) → float32 dequantization *before* the kernel runs
- fp16 → fp32 upcasting in a Python property
- Sparse/compressed → dense materialization
- Lazy graph evaluation that creates intermediate tensors

**If data expands before reaching the kernel, the kernel sees amplified bandwidth.**
A 4-bit value cache pre-expanded to float32 costs 7× more bandwidth than necessary.

### 1.2 Calculate Actual vs Theoretical Bandwidth

```python
# For each input, compute:
theoretical_bytes = element_count * compressed_bytes_per_element
actual_bytes      = element_count * dtype_size  # after any pre-expansion

ratio = actual_bytes / theoretical_bytes
# ratio > 2x → fuse the decompression into the kernel
```

### 1.3 Eliminate Intermediate Tensors

Search for any `mx.array` created *solely* to be passed to the kernel and discarded.
These are fusion opportunities. Fusing avoids a global memory round-trip.

### 1.4 Call `mx.eval()` Before Kernel Dispatch

MLX uses lazy evaluation. All kernel inputs must be materialized before launch:
```python
mx.eval(input_a, input_b, ...)  # force any pending computation
outputs = kernel(inputs=[input_a, input_b, ...], ...)
```

---

## Phase 2: Thread Model Design

**This is the most consequential design decision. The "obvious" thread model is often wrong.**

### 2.1 Map Every Problem Dimension

Before writing a single line of kernel code, write down:
```
Output dimensions:    (batch, seq_len, n_heads, head_dim, ...)
Reduction dimensions: (vocab, time, blocks, ...)
Shared dimensions:    (codebook, lookup_table, weight matrix, ...)
```

### 2.2 Classify Each Dimension

For each dimension, answer two questions:

| Question | Answer → Thread Model |
|---|---|
| Are elements *independent* across this dimension? | Parallelize: give each thread its own slice |
| Does this dimension require *reduction* across threads? | Choose: SIMD shuffle, threadgroup atomic, or eliminate entirely |
| Is this dimension's data *shared identically* by all threads? | Load once into threadgroup memory; no per-thread work needed |

**The reduction elimination insight:** if all threads in a threadgroup read the same
value to compute the same reduction key (e.g., a shared score from a lookup table),
their reduction state stays synchronized *without a barrier or shuffle* — because
it was never divergent. This is the structural difference between standard
Flash Attention (independent per-thread dot products → needs SIMD reduction) and
lookup-based attention (shared scores → no reduction needed).

### 2.3 Threadgroup Sizing Rules

```
SIMD width: 32 threads (constant on all Apple GPUs)
Threadgroup size must be a multiple of 32
Good range: 64–256 threads per threadgroup for ML workloads
Max: 1024 threads per threadgroup (hard limit)
Rule of thumb: 1K–2K concurrent threads per GPU shader core for good occupancy
```

Smaller threadgroups distribute more evenly across GPU cores. Avoid large threadgroups
unless the algorithm requires it (e.g., large shared memory reductions).

### 2.4 Register Pressure Check

Count registers per thread before committing to a thread model:

```metal
// COSTLY: each thread holds a full accumulator array → register spill
float acc[128];   // 128 registers per thread

// CHEAP: one register per thread
float acc = 0.0f; // 1 register
```

Register spill writes to device memory, converting an ALU-bound kernel into a
bandwidth-bound one. If your current thread model requires large per-thread arrays,
reconsider whether to redistribute dimensions across threads.

---

## Phase 3: Memory Hierarchy — Placing Data Correctly

### 3.1 Threadgroup Memory: Data Read by All Threads

Any data that every thread in a threadgroup reads should be loaded once into
threadgroup (SRAM) memory:

```metal
// Pattern: cooperative load, then barrier
threadgroup float tg_data[BUFFER_SIZE];
for (uint i = thread_index_in_threadgroup; i < BUFFER_SIZE; i += threads_per_threadgroup) {
    tg_data[i] = global_input[base_offset + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);  // REQUIRED before reading
// All threads now read from tg_data[] instead of global memory
```

**Important MLX gotcha:** use `mem_flags::mem_threadgroup` not `mem_threadgroup`.
The `mem_flags::` namespace qualifier is required inside `mx.fast.metal_kernel`.

Common candidates for threadgroup memory:
- Lookup tables, codebooks, centroids
- Shared weight tiles in matrix multiplication
- Softmax normalization constants when broadcast to many threads
- Small query vectors used by all threads

**Apple Family 9+ caveat:** for small working sets on M4+ chips, the L2/SLC may
cache device-memory reads just as effectively as explicit threadgroup memory. Profile
with Xcode Metal Debugger before assuming threadgroup memory always wins.

### 3.2 `constant` Address Space: Read-Only, All Threads

For read-only data that does not change across the dispatch, `constant` address space
enables prefetching:

```metal
// In kernel body (MLX passes buffers as device pointers; cast if needed)
constant float* weights = (constant float*)raw_weights;
```

### 3.3 Per-Thread Data: Keep in Registers

With a well-chosen thread model, each thread's working data should be a small number
of scalar accumulators, not arrays. Thread-per-output-dimension layouts typically
reduce accumulators to a single `float` per thread.

### 3.4 Precompute Loop-Invariant Addressing

Move all index arithmetic that is constant across a loop outside the loop:

```metal
// Precompute ONCE before the token/reduction loop:
uint packed_dim  = head_dim / vals_per_word;
uint word_idx    = dim_id / vals_per_word;
uint shift       = (dim_id % vals_per_word) * bits;
uint group_idx   = dim_id / group_size;

// Inside loop: only use precomputed indices
for (uint t = 0; t < T; t++) {
    uint word = packed[t * packed_dim + word_idx];  // single global read
    ...
}
```

---

## Phase 4: In-Kernel Dequantization

When any kernel input is stored in a quantized/packed format, fuse the dequantization
into the kernel rather than pre-expanding to float before calling the kernel.

### 4.1 The Pattern

```metal
// Precompute (once, outside the loop — loop-invariant per thread):
uint vals_per_word = 32 / bits;          // e.g. 8 for 4-bit
uint packed_dim    = head_dim / vals_per_word;
uint word_idx      = dim_id / vals_per_word;
uint shift         = (dim_id % vals_per_word) * bits;
uint bit_mask      = (1u << bits) - 1u;  // e.g. 0xF for 4-bit
uint group_idx     = dim_id / group_size;

// Per-element (inside the loop):
uint word  = packed_tensor[t * packed_dim + word_idx];
uint qval  = (word >> shift) & bit_mask;
float val  = (float)qval * scales[t * n_groups + group_idx]
           + zeros[t * n_groups + group_idx];
```

### 4.2 Exposing Raw Storage on the Host Side

The Python/MLX layer must expose raw packed storage, not just a dequantized property:

```python
class QuantizedCache:
    @property
    def packed_words(self) -> mx.array:   # uint32, shape [T, packed_dim]
        return self._packed

    @property
    def scales(self) -> mx.array:         # float32, shape [T, n_groups]
        return self._scales

    @property
    def zeros(self) -> mx.array:          # float32, shape [T, n_groups]
        return self._zeros

    @property
    def dequantized(self) -> mx.array:    # float32 — keep for fallback paths only
        return mx.dequantize(self._packed, self._scales, self._zeros, ...)
```

The kernel fast path takes `packed_words`, `scales`, `zeros` directly.
The dequantized property is reserved for fallback paths and non-performance-critical use.

### 4.3 Bandwidth Savings Reference

| Format | Bytes/token pre-deq (head_dim=128) | Bytes/token fused | Savings |
|---|---|---|---|
| float32 | 512 | 512 (no change) | — |
| float16 | 256 | 256 (no change) | — |
| 8-bit | 512 (expanded) | 136 | 3.8× |
| 4-bit | 512 (expanded) | 72 | 7.1× |
| 2-bit | 512 (expanded) | 40 | 12.8× |

*(includes 2 bytes/element for scales+zeros at group_size=64)*

---

## Phase 5: MLX API Reference

### 5.1 `mx.fast.metal_kernel` — Key Facts

```python
kernel = mx.fast.metal_kernel(
    name="my_kernel",           # used as the generated function name
    input_names=["a", "b"],     # must match variable names used in source
    output_names=["out"],
    source=source_string,       # kernel BODY only — no [[kernel]] declaration
    header="",                  # optional: #include or helper function definitions
    ensure_row_contiguous=True, # default True; copies inputs to row-major if needed
    atomic_outputs=False,       # set True only if outputs use atomic writes
)

outputs = kernel(
    inputs=[a, b],
    template=[("T", mx.float32), ("N", 256)],  # type + integer + boolean params
    grid=(total_x, total_y, total_z),           # total threads, not groups
    threadgroup=(tg_x, tg_y, tg_z),
    output_shapes=[out_shape],
    output_dtypes=[mx.float32],
    init_value=0.0,   # pre-fill output buffer; omit if kernel writes all elements
    verbose=False,    # True prints generated Metal code for debugging
)
```

**Critical:** `source` is the function **body only**. MLX generates the function
signature automatically. Never write `[[kernel]] void ...` in the source string.

### 5.2 Automatically Available Variables in Source

MLX injects these without any declaration in source:

```metal
// Thread position
uint3 thread_position_in_grid;       // absolute thread index in 3D grid
uint3 thread_position_in_threadgroup;// thread index within its threadgroup
uint3 threadgroup_position_in_grid;  // which threadgroup this is
uint3 threads_per_threadgroup;       // threadgroup dimensions
uint  thread_index_in_threadgroup;   // flattened thread index within threadgroup
uint  thread_index_in_simdgroup;     // position within 32-thread SIMD group
uint  simdgroup_index_in_threadgroup;// which SIMD group within threadgroup

// Shape/stride info (when ensure_row_contiguous=False, or names appear in source)
// For each input named "a":
//   a_shape[i], a_strides[i], a_ndim
```

### 5.3 Template Parameters

```python
template=[
    ("T", mx.float32),   # type: passes as C++ typename, usable in source as T
    ("N", 256),          # integer: becomes a compile-time constant
    ("USE_BIAS", True),  # boolean: becomes a compile-time constant
]
```

Use integer template parameters for loop bounds that are known at compile time — this
enables the compiler to unroll loops and eliminate branches.

### 5.4 Strided / Non-Contiguous Input Handling

```python
kernel = mx.fast.metal_kernel(..., ensure_row_contiguous=False)
```

```metal
// In source: use elem_to_loc from mlx/backend/metal/kernels/utils.h
uint elem = thread_position_in_grid.x;
uint loc  = elem_to_loc(elem, a_shape, a_strides, a_ndim);
T val     = a[loc];
out[elem] = some_op(val);   // output is always row-contiguous
```

### 5.5 Output Initialization

```python
# Use init_value when the kernel may not write every output element
# (e.g., causal masking, sparse outputs):
outputs = kernel(..., init_value=0.0)

# Omit init_value when the kernel is guaranteed to write all elements —
# initialization wastes a global memory pass.
```

### 5.6 Debugging

```python
# Print generated Metal source code:
outputs = kernel(..., verbose=True)

# For correctness testing, compare against NumPy reference (not another MLX impl):
np_ref  = numpy_reference_implementation(a_np, b_np)
mlx_out = mx.array(outputs[0])
assert np.allclose(np_ref, np.array(mlx_out), atol=5e-3), "Kernel output mismatch"
```

---

## Phase 6: SIMD-Group Operations

Apple Silicon has strong SIMD-shuffle bandwidth. Prefer SIMD operations over
threadgroup-memory reductions for small reductions within a 32-thread SIMD group.

```metal
// Parallel reduction within a SIMD group (no barrier required):
float val = my_local_value;
val = simd_sum(val);          // sum all 32 threads
val = simd_max(val);          // max across 32 threads
val = simd_min(val);          // min across 32 threads
val = simd_prefix_exclusive_sum(val);

// Shuffle: read value from another thread in the SIMD group
float neighbor = simd_shuffle(val, target_lane);  // target_lane: 0–31
float shifted  = simd_shuffle_down(val, delta);
float rotated  = simd_shuffle_rotate_down(val, delta);

// Matrix multiply (Apple's "tensor core"):
simdgroup_float8x8 A, B, C;
simdgroup_load(A, src_a, ...);
simdgroup_load(B, src_b, ...);
simdgroup_multiply_accumulate(C, A, B, C);
simdgroup_store(C, dst, ...);
```

**When to use SIMD shuffle vs threadgroup memory:**
- Reduction across 32 threads within a SIMD group → SIMD shuffle (no barrier, fast)
- Communication across SIMD groups within a threadgroup → threadgroup memory + barrier
- Broadcasting a single value to all threads → threadgroup memory or `constant` space

---

## Phase 7: The Port-from-CUDA Checklist

Porting a CUDA kernel to MLX Metal is the highest-risk scenario. For each design
decision in the original, ask whether the *reason* for that decision still holds.

```
□ Thread model:
  - CUDA warps = 32; Metal SIMD groups = 32 ✓ (same)
  - CUDA __shared__ = Metal threadgroup  ✓
  - CUDA atomicAdd = Metal atomic_fetch_add_explicit ✓
  - CUDA half2 vectorized loads → use float2/float4 in Metal similarly ✓

□ Memory spaces:
  - CUDA constant memory → Metal `constant` address space ✓
  - CUDA texture memory → Metal texture sampling (or just device buffer + cache) ✓
  - CUDA __ldg (read-only cache) → Metal `constant` or just device buffer ✓

□ Things that do NOT translate:
  - CUDA warp-level __ballot_sync, __activemask → no direct Metal equivalent;
    use simd_vote_any / simd_vote_all
  - CUDA cooperative groups → no equivalent; restructure around threadgroup barriers
  - CUDA dynamic parallelism (kernel launching kernels) → not supported in MLX
  - CUDA L1/shared memory split (e.g., 48KB shared / 16KB L1) → fixed in Metal

□ Apple Silicon UMA implications:
  - No cudaMemcpy needed (CPU and GPU share memory)
  - BUT bandwidth is still finite and shared with CPU — measure before assuming "free"
  - Prefer .bfloat16 or .float16 tensors in MLX to halve bandwidth before the kernel

□ Port-fidelity trap — always ask:
  "Was this design choice inherent to the algorithm, or was it a CUDA-specific
   optimization?" Re-derive the thread model from the problem dimensions.
```

---

## Phase 8: Performance Limiters and Profiling

Profile with Xcode Metal Debugger / Instruments → GPU Tools before and after
optimization. Look for these performance limiters:

| Limiter | Symptom | Fix |
|---|---|---|
| **ALU** | High shader ALU utilization | Replace heavy math with lookup tables; use `half`; enable `-ffast-math` |
| **Buffer Read/Write** | High buffer bandwidth | Vectorize loads (`float4`); pack data tighter; move hot data to threadgroup mem |
| **Threadgroup/Imageblock** | High threadgroup mem usage | Reduce threadgroup atomics; align allocations to 16 bytes; reorder access patterns |
| **Occupancy** | Low concurrent threads | Reduce register count (use 16-bit types); reduce threadgroup size |
| **GPU Last Level Cache** | High LLC miss rate | Improve spatial locality; reduce working set size |

### Key metrics to track

```python
# Before and after any optimization, measure:
# 1. Wall time (ms) for the operation
# 2. Bandwidth: (bytes_read + bytes_written) / time
# 3. Theoretical bandwidth for your SoC (M1: ~68 GB/s, M2: ~100 GB/s, M3: ~120 GB/s, M4: ~120+ GB/s)
# 4. Compute utilization: FLOPs / (time × peak_TFLOPS)
# Which limiter you hit tells you what to optimize next.
```

---

## Phase 9: Testing and Correctness

### Always Test Against a NumPy Reference

Do not test one MLX implementation against another; differences may be consistent
but both wrong.

```python
import numpy as np
import mlx.core as mx

def numpy_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Explicit, readable implementation with no approximations."""
    ...

def test_kernel(a_mx: mx.array, b_mx: mx.array):
    a_np = np.array(a_mx)
    b_np = np.array(b_mx)

    expected = numpy_reference(a_np, b_np)
    actual   = np.array(my_kernel(a_mx, b_mx))

    max_err = np.abs(expected - actual).max()
    assert max_err < 5e-3, f"Max error {max_err:.2e} exceeds threshold"
```

### What to Test

```
□ Multiple input sizes: small (edge cases), medium (typical), large (performance)
□ Non-power-of-two sizes: reveals indexing off-by-ones
□ Boundary conditions: last tile, causal mask edge, T=1
□ Both float32 and float16 if the kernel is templated on type T
□ Multiple quantization bit-widths if applicable (2-bit, 4-bit, 8-bit)
□ Causal and non-causal variants if the kernel has a masking mode
□ All template parameter combinations that will be called in production
```

### Error Thresholds

| Computation | Acceptable max error | Typical |
|---|---|---|
| Float32 elementwise | < 1e-5 | ~0 |
| Online softmax / attention | < 5e-3 | < 1e-4 |
| Mixed float16 accumulation | < 1e-2 | ~1e-3 |
| 4-bit dequant + accumulation | < 5e-3 | < 1e-4 |

---

## Phase 10: MLX-Specific Patterns and Gotchas

```
□ Kernel is body-only — no [[kernel]], no function signature
□ mem_flags::mem_threadgroup — required namespace qualifier for barriers
□ mx.eval() all inputs BEFORE calling kernel()
□ Threadgroup size ≤ 1024, threadgroup memory ≤ 32 KB
□ Each threadgroup dimension ≤ corresponding grid dimension
□ MLX JIT-compiles kernels on first use; Metal caches compiled kernels across reboots
□ Use mx.fast.metal_kernel() once at module level, call it many times (avoid re-JIT)
□ ensure_row_contiguous=True (default) copies non-contiguous inputs — be aware of cost
□ init_value only when needed; it costs a global memory pass
□ Template integer params enable loop unrolling — use them for fixed inner-loop bounds
□ 16-bit literals need 'h' suffix: 1.0h, not 1.0f, to avoid silent fp32 promotion
□ Wrap kernel dispatch in try/except; return None and fall back to mx.fast or pure MLX
```

---

## Kernel Template — General Structure

```metal
// ── Kernel boilerplate (adapt dimensions to your problem) ──────────────────
// Grid:        (n_outputs, 1, 1)  or  (batch * heads * dim, 1, 1)
// Threadgroup: (THREADS_PER_GROUP, 1, 1)  — must be multiple of 32

constexpr uint THREADS = /* template param or literal */;
uint gid = thread_position_in_grid.x;
uint tid = thread_index_in_threadgroup;

// ── 1. Bounds check ────────────────────────────────────────────────────────
uint n_outputs = /* parsed from params or shape array */;
if (gid >= n_outputs) return;

// ── 2. Cooperative load of shared data ────────────────────────────────────
threadgroup float tg_shared[SHARED_SIZE];
for (uint i = tid; i < SHARED_SIZE; i += THREADS) {
    tg_shared[i] = global_shared_data[base + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ── 3. Precompute loop-invariant addressing ────────────────────────────────
// (e.g., for quantized inputs: word_idx, shift, group_idx per output element)

// ── 4. Main computation loop ───────────────────────────────────────────────
float acc = 0.0f;
// optional online reduction state (e.g., for softmax: m = -inf, d = 0)
for (uint i = 0; i < N; i++) {
    // read input (possibly with in-register dequant)
    // update acc
}

// ── 5. Write output ────────────────────────────────────────────────────────
out[gid] = acc;
```

---

## Quick Decision Reference

```
Need a custom kernel?
├── Does mx.fast / mx.compile cover it? → Use those. Done.
└── No → continue

Data flow:
├── Is any input pre-expanded from a compact format?
│   └── YES → pass raw storage, fuse decompression in kernel
└── Are any intermediate mx.arrays created just for the kernel?
    └── YES → fuse those ops into the kernel or remove them

Thread model:
├── Are output elements independent? → parallelize across threads
├── Is there a reduction?
│   ├── Within a SIMD group (≤32 threads)? → simd_sum / simd_max etc.
│   └── Across SIMD groups? → threadgroup memory + barrier
└── Is reduction data SHARED across threads? → load to threadgroup memory, no reduction needed

Memory placement:
├── Read by ALL threads in threadgroup? → threadgroup memory
├── Read-only, broadcast? → constant address space
└── Per-thread, small? → registers (scalar or small struct, not arrays)
```