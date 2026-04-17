"""Debug script to surface the actual Metal kernel compilation error."""
import mlx.core as mx
import numpy as np
from sunshape_mlx.kernels import _fused_attention_kernel

np.random.seed(111)
T_q, T_kv = 1, 64
n_blocks, n_centroids, block_dim = 16, 256, 8
head_dim = n_blocks * block_dim

centroids = np.random.randn(n_blocks, n_centroids, block_dim).astype(np.float32)
indices_np = np.random.randint(0, n_centroids, size=(T_kv, n_blocks), dtype=np.uint8)
values_np = np.random.randn(T_kv, head_dim).astype(np.float32)
query_np = np.random.randn(T_q, head_dim).astype(np.float32)

q_blocks = query_np.reshape(T_q, n_blocks, block_dim)
qdots_np = np.einsum('qbd,bcd->qbc', q_blocks, centroids)

qdots = mx.array(qdots_np)
indices = mx.array(indices_np)
values = mx.array(values_np)

params = mx.array([float(T_kv), float(n_blocks), float(n_centroids), float(head_dim)], dtype=mx.float32)
mx.eval(qdots, indices, values, params)

n_threads = 32
grid = (T_q * n_threads, 1, 1)
threadgroup = (n_threads, 1, 1)

try:
    output = _fused_attention_kernel(
        inputs=[qdots, indices, values, params],
        output_shapes=[(T_q, head_dim)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup,
        init_value=0.0,
        verbose=True,
    )
    out = output[0] if isinstance(output, (list, tuple)) else output
    mx.eval(out)
    print("SUCCESS!", out.shape)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
