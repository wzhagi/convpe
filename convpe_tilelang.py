import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
from tilelang.transform.pass_config import PassConfigKey


# ============================================================
# 对外类: ConvPE (tilelang 推理版)
# ============================================================
class ConvPE(nn.Module):
    """
    input:  x (b, s, dim_model)
    output: score_bias (b, s, 1, window_size)

    dtype = bfloat16
    s % 32 == 0
    dim_model % 256 == 0

    """

    _BLOCK_S = 32
    _BLOCK_K = 32
    _BLOCK_D = 256

    def __init__(self, dim_model, window_size, dim_convpe, **kwargs):
        super().__init__()
        assert dim_model % self._BLOCK_D == 0
        assert (dim_convpe * window_size) % self._BLOCK_K == 0
        self.window_size = window_size
        self.dim_convpe = dim_convpe
        self.proj = nn.Linear(dim_model, dim_convpe, bias=False)
        self.conv = nn.Linear(dim_convpe * window_size, window_size)

    def forward(self, x):
        b, s, d = x.shape
        w = self.window_size
        c = self.dim_convpe

        assert x.dtype == torch.bfloat16
        assert s % self._BLOCK_S == 0
        self.proj = self.proj.to(torch.bfloat16)
        self.conv = self.conv.to(torch.bfloat16)
        proj_w = self.proj.weight
        conv_w = self.conv.weight
        conv_b = self.conv.bias
        proj_k, conv_k = _get_kernels(
            b,
            s,
            d,
            w,
            c,
            "bfloat16",
            block_S=self._BLOCK_S,
            block_K=self._BLOCK_K,
            block_d=self._BLOCK_D,
            num_stages=1,
        )
        x_proj = proj_k(x.contiguous(), proj_w.contiguous())
        out = conv_k(x_proj, conv_w.contiguous(), conv_b.contiguous())
        return out.view(b, s, 1, w)


# ============================================================
# TileLang 融合 kernel
# ============================================================
_KERNEL_CACHE = {}


def _make_kernels(b, s, d, w, c, block_S, block_K, block_d, num_stages, dtype_str):
    wc = w * c

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def _proj_kernel():
        @T.prim_func
        def kernel(
            X: T.Tensor((b, s, d), dtype_str),
            ProjW: T.Tensor((c, d), dtype_str),
            Out: T.Tensor((b, s, c), dtype_str),
        ):
            with T.Kernel(T.ceildiv(s, block_S), b, threads=128) as (bs, bb):
                X_shared = T.alloc_shared((block_S, block_d), dtype_str)
                P_shared = T.alloc_shared((c, block_d), dtype_str)
                acc = T.alloc_fragment((block_S, c), "float32")
                T.clear(acc)
                for ko in T.Pipelined(d // block_d, num_stages=num_stages):
                    T.copy(X[bb, bs * block_S, ko * block_d], X_shared)
                    T.copy(ProjW[0, ko * block_d], P_shared)
                    T.gemm(X_shared, P_shared, acc, transpose_B=True)
                T.copy(acc, Out[bb, bs * block_S, 0])

        return kernel

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def _conv_kernel():
        @T.prim_func
        def kernel(
            XProj: T.Tensor((b, s, c), dtype_str),
            ConvW: T.Tensor((w, wc), dtype_str),  # 原始 (w, c*w) 布局
            Bias: T.Tensor((w,), dtype_str),
            Out: T.Tensor((b, s, w), dtype_str),
        ):
            with T.Kernel(T.ceildiv(s, block_S), b, threads=128) as (bs, bb):
                acc = T.alloc_fragment((block_S, w), "float32")
                T.clear(acc)
                for ko in T.Pipelined(wc // block_K, num_stages=num_stages):
                    XU_shared = T.alloc_shared((block_S, block_K), dtype_str)
                    C_shared = T.alloc_shared((w, block_K), dtype_str)

                    # 原始 conv weight: k = i_c * w + i_w
                    for si, ki in T.Parallel(block_S, block_K):
                        k_glob = ko * block_K + ki
                        ci_idx = k_glob // w
                        wi_idx = k_glob % w
                        s_out = bs * block_S + si
                        s_in = s_out + wi_idx - (w - 1)
                        XU_shared[si, ki] = T.if_then_else(
                            (s_in >= 0) & (s_in < s),
                            XProj[bb, s_in, ci_idx],
                            T.cast(0.0, dtype_str),
                        )

                    for wi, ki in T.Parallel(w, block_K):
                        C_shared[wi, ki] = ConvW[wi, ko * block_K + ki]

                    T.gemm(XU_shared, C_shared, acc, transpose_B=True)

                for si, wi in T.Parallel(block_S, w):
                    acc[si, wi] = acc[si, wi] + T.cast(Bias[wi], "float32")

                T.copy(acc, Out[bb, bs * block_S, 0])

        return kernel

    return _proj_kernel(), _conv_kernel()


def _get_kernels(
    b, s, d, w, c, dtype_str, block_S=32, block_K=32, block_d=256, num_stages=1
):
    key = (b, s, d, w, c, dtype_str, block_S, block_K, block_d, num_stages)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _make_kernels(
            b,
            s,
            d,
            w,
            c,
            block_S,
            block_K,
            block_d,
            num_stages,
            dtype_str,
        )
    return _KERNEL_CACHE[key]
