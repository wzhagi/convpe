import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse

dtype = torch.bfloat16
LOG2E = 1.44269504


def convpe_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor = None,
    score_bias: torch.Tensor = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    FlashAttention with banded score bias (per bias-head, GQA style) and sink.

    Args:
        q: Query tensor, shape ``(B, T, H, D)``.
        kv: Key (also used as Value), shape ``(B, T, H_kv, D)``.
        score_bias: Optional banded bias, shape ``(B, T, H_b, W)``.
            ``H_b`` must divide ``H``; head ``h`` uses bias-head ``h // (H // H_b)``.
            ``None`` = no bias.
        sink: Optional attention-sink per head, shape ``(H,)``. ``None`` = no sink.
        causal: If ``True`` (default), apply causal masking.
    """
    dtype_in = q.dtype
    q, kv = q.to(dtype), kv.to(dtype)
    if q.dim() != 4 or kv.dim() != 4:
        raise ValueError(
            f"q and kv must be 4D (B, T, H, D), "
            f"got q.dim()={q.dim()}, kv.dim()={kv.dim()}"
        )
    B, T, H, D = q.shape
    B_k, T_k, H_kv, D_k = kv.shape
    if (B, T, D) != (B_k, T_k, D_k):
        raise ValueError(
            f"q and kv must match on (B, T, D), "
            f"got q.shape={tuple(q.shape)}, kv.shape={tuple(kv.shape)}"
        )
    if H % H_kv != 0:
        raise ValueError(f"H={H} must be divisible by H_kv={H_kv} (MQA/GQA)")
    groups = H // H_kv

    if score_bias is not None:
        if score_bias.dim() != 4:
            raise ValueError(
                f"score_bias must be 4D (B, T, H_b, W), "
                f"got shape={tuple(score_bias.shape)}"
            )
        if score_bias.shape[:2] != (B, T):
            raise ValueError(
                f"score_bias must have shape (B={B}, T={T}, H_b, W), "
                f"got {tuple(score_bias.shape)}"
            )
        H_b = score_bias.shape[2]
        if H % H_b != 0:
            raise ValueError(f"H={H} must be divisible by H_b={H_b}")

    if sink is not None:
        if sink.dim() != 1 or sink.shape[0] != H:
            raise ValueError(
                f"sink must have shape (H={H},), got shape={tuple(sink.shape)}"
            )
    else:
        sink = torch.full((H,), -1e30, dtype=torch.float32, device=q.device)

    if not q.is_contiguous():
        q = q.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if sink is not None and sink.dtype != torch.float32:
        sink = sink.to(torch.float32)

    return _attention.apply(q, kv, score_bias, sink, causal, groups).to(dtype_in)


# ============================================================
# Forward：full causal + windowed bias（per bias-head）+ attention sink
# （保持原样，此 kernel 性能已达标）
# ============================================================
@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn_fwd(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M,
    block_N,
    groups=1,
    groups_b=1,
    W=128,
    num_stages=2,
):
    has_bias = W > 0
    scale = (1.0 / dim) ** 0.5 * LOG2E
    sqrt_d_val = dim**0.5
    head_kv = heads // groups
    head_b = heads // groups_b  # H_b
    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, head_kv, dim]
    bias_shape = [batch, seq_len, head_b, W] if has_bias else [1, 1, 1, 1]
    accum_dtype = T.float32

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        score_bias: T.Tensor(bias_shape, dtype),
        sink: T.Tensor([heads], accum_dtype),
        Output: T.Tensor([batch, seq_len, heads, dim], dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=256) as (
            bx,
            by,
            bz,
        ):
            T.use_swizzle(panel_size=10)

            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            # ★ 预加载当前 query block 的 bias tile 到 shared
            if is_causal and has_bias:
                bias_shared = T.alloc_shared([block_M, W], dtype)
                for i, ww in T.Parallel(block_M, W):
                    q_idx = bx * block_M + i
                    q_idx_c = T.min(q_idx, seq_len - 1)
                    bias_shared[i, ww] = score_bias[bz, q_idx_c, by // groups_b, ww]

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if is_causal:
                loop_start = 0
                loop_end = T.ceildiv((bx + 1) * block_M, block_N)
            else:
                loop_start = 0
                loop_end = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_start, loop_end, num_stages=num_stages):
                T.copy(
                    K[bz, k * block_N : (k + 1) * block_N, by // groups, :],
                    K_shared,
                )
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(acc_s.dtype),
                            0,
                        )
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                if is_causal and has_bias:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = k * block_N + j
                        w_idx = q_idx - k_idx
                        in_window = (w_idx >= 0) and (w_idx < W)
                        w_c = T.min(T.max(w_idx, 0), W - 1)
                        acc_s[i, j] = acc_s[i, j] + T.if_then_else(
                            in_window,
                            T.cast(bias_shared[i, w_c], accum_dtype) * sqrt_d_val,
                            accum_dtype(0),
                        )

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.if_then_else(
                        scores_max[i] == -T.infinity(accum_dtype),
                        T.float32(1),
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                    )
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] == -T.infinity(accum_dtype),
                        T.float32(0),
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                    )
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, K_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            # ---- 融入 sink ----
            sink2 = sink[by] * LOG2E
            for i in T.Parallel(block_M):
                m_scaled = scores_max[i] * scale
                new_m = T.max(m_scaled, sink2)
                scores_scale[i] = T.exp2(m_scaled - new_m)
                logsum[i] = logsum[i] * scores_scale[i] + T.exp2(sink2 - new_m)
                scores_max[i] = new_m / scale
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] = acc_o[i, j] * scores_scale[i] / logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd


# ============================================================
# Backward preprocess / postprocess（保持不变）
# ============================================================
@tilelang.jit(
    out_idx=[2],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(
                    dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do
                )
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep


def make_dq_layout(dQ):
    return T.Layout(
        dQ.shape,
        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@tilelang.jit(
    out_idx=[1],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (
            bx,
            by,
            bz,
        ):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_post


# ============================================================
# Backward main kernel
# ★ 优化点（不增加 shared 用量，反而净省 56KB）：
#   P0: dB_full 改 [B,T,H,W]，head 间无冲突（原来 groups_b 个 head 争用同一位置）
#   P1: bias 从 global gather 改成 bias_tile shared 预加载（+8KB）
#   P2: 加 T.use_swizzle
#   P3: 去掉 dk_shared（fp32 128x128=64KB），改为逐元素 atomic_add(dk[i,j]+dv[i,j])
# ============================================================
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_bwd_atomic_add(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M,
    block_N,
    threads=256,
    num_stages=1,
    groups=1,
    groups_b=1,
    W=128,
):
    has_bias = W > 0
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * LOG2E
    sqrt_d_val = dim**0.5
    head_kv = heads // groups
    head_b = heads // groups_b

    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, head_kv, dim]
    bias_shape = [batch, seq_len, head_b, W] if has_bias else [1, 1, 1, 1]
    dB_shape = [batch, seq_len, heads, W] if has_bias else [1, 1, 1, 1]  # per-head
    accum_dtype = T.float32

    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        score_bias: T.Tensor(bias_shape, dtype),
        sink: T.Tensor([heads], accum_dtype),
        dO: T.Tensor([batch, seq_len, heads, dim], dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(q_shape, accum_dtype),
        dK_full: T.Tensor([batch, seq_len, heads, dim], accum_dtype),
        dB_full: T.Tensor(dB_shape, accum_dtype),
        d_sink_bh: T.Tensor([batch, heads], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            T.use_swizzle(panel_size=10)

            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dkdv = T.alloc_fragment([block_M, dim], accum_dtype)  # ★ dK+dV 中转
            # ★ dk_shared 已删除（省 64KB）

            sink_acc = T.alloc_fragment([block_N], accum_dtype)
            sink_total = T.alloc_fragment([1], accum_dtype)
            T.fill(sink_acc, 0)
            T.fill(sink_total, 0)
            sink2 = sink[bx] * LOG2E

            T.annotate_layout({dQ: make_dq_layout(dQ)})

            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx // groups, :], K_shared)
            T.clear(dv)
            T.clear(dk)

            if is_causal:
                loop_st = T.floordiv(by * block_M, block_N)
                loop_ed = T.ceildiv(seq_len, block_N)
            else:
                loop_st = 0
                loop_ed = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(
                    K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )
                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)

                if is_causal and has_bias:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = k * block_N + j
                        k_idx = by * block_M + i
                        w_idx = q_idx - k_idx
                        in_causal = (
                            (w_idx >= 0) and (q_idx < seq_len) and (k_idx < seq_len)
                        )
                        in_window = in_causal and (w_idx < W)
                        w_c = T.min(T.max(w_idx, 0), W - 1)
                        q_idx_c = T.min(q_idx, seq_len - 1)
                        bias_val = T.if_then_else(
                            in_window,
                            T.cast(
                                score_bias[bz, q_idx_c, bx // groups_b, w_c],
                                accum_dtype,
                            )
                            * sqrt_d_val,
                            accum_dtype(0),
                        )
                        qkT[i, j] = T.if_then_else(
                            in_causal,
                            T.exp2((qkT[i, j] + bias_val) * scale - lse_shared[j]),
                            accum_dtype(0),
                        )
                elif is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = k * block_N + j
                        k_idx = by * block_M + i
                        in_causal = (
                            (q_idx >= k_idx) and (q_idx < seq_len) and (k_idx < seq_len)
                        )
                        qkT[i, j] = T.if_then_else(
                            in_causal,
                            T.exp2(qkT[i, j] * scale - lse_shared[j]),
                            accum_dtype(0),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])

                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(
                    K_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)

                if by == 0:
                    for j in T.Parallel(block_N):
                        q_idx = k * block_N + j
                        valid = q_idx < seq_len
                        p_sink = T.exp2(sink2 - lse_shared[j])
                        contrib = T.if_then_else(
                            valid, -p_sink * delta[j], T.float32(0)
                        )
                        sink_acc[j] += contrib

                if is_causal and has_bias:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = k * block_N + j
                        k_idx = by * block_M + i
                        w_idx = q_idx - k_idx
                        in_causal = (
                            (w_idx >= 0) and (q_idx < seq_len) and (k_idx < seq_len)
                        )
                        in_window = in_causal and (w_idx < W)
                        d_bias = qkT[i, j] * (dsT[i, j] - delta[j])
                        dsT_cast[i, j] = d_bias * sm_scale
                        # ★ per-head dB 无 conflict，普通 store，且 in_window 内才写
                        if in_window:
                            dB_full[bz, q_idx, bx, w_idx] = d_bias
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale

                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                # dQ 仍需 atomic（多 by 写同一 q 行）
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            if by == 0:
                T.reduce_sum(sink_acc, sink_total, dim=0)
                T.copy(sink_total, d_sink_bh[bz, bx : bx + 1])

            # ★ dK 无 conflict：dk+dv 到中转 fragment 后 T.copy 普通写
            for i, j in T.Parallel(block_M, dim):
                dkdv[i, j] = dk[i, j] + dv[i, j]
            T.copy(dkdv, dK_full[bz, by * block_M : (by + 1) * block_M, bx, :])

    return flash_bwd


# ============================================================
# Autograd wrapper
# ============================================================
@torch.compile
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, score_bias, sink, causal, groups=1):
        BATCH, N_CTX, H, D_HEAD = q.shape
        ctx.causal = causal

        if score_bias is not None:
            H_b = score_bias.shape[2]
            W = score_bias.shape[3]
            groups_b = H // H_b
            ctx.score_bias_orig_dtype = score_bias.dtype
            sb_bf16 = score_bias if score_bias.dtype == dtype else score_bias.to(dtype)
            sb_bf16 = sb_bf16.contiguous()
        else:
            W = 0
            groups_b = 1
            ctx.score_bias_orig_dtype = None
            sb_bf16 = None

        block_M, block_N = 128, 64
        mod = flashattn_fwd(
            BATCH,
            H,
            N_CTX,
            D_HEAD,
            causal,
            block_M,
            block_N,
            groups,
            groups_b,
            W,
        )

        sb_arg = (
            sb_bf16
            if sb_bf16 is not None
            else torch.zeros(1, 1, 1, 1, dtype=dtype, device=q.device)
        )
        o, lse = mod(q, k, sb_arg, sink)
        ctx.save_for_backward(q, k, sb_bf16, sink, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, sb_bf16, sink, o, lse = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD = q.shape
        HEAD_KV = k.shape[-2]
        groups = H // HEAD_KV

        has_bias = sb_bf16 is not None
        if has_bias:
            W = sb_bf16.shape[3]
            H_b = sb_bf16.shape[2]
            groups_b = H // H_b
        else:
            W = 0
            H_b = H
            groups_b = 1

        def maybe_contiguous(x):
            return x.contiguous() if x.stride(-1) != 1 else x

        do, q, k, o = [maybe_contiguous(x) for x in (do, q, k, o)]

        mod_prep = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)
        mod_post = flashattn_bwd_postprocess(BATCH, H, N_CTX, D_HEAD)
        delta = mod_prep(o, do)

        block_M, block_N = 128, 32
        kernel = flashattn_bwd_atomic_add(
            BATCH,
            H,
            N_CTX,
            D_HEAD,
            ctx.causal,
            block_M,
            block_N,
            threads=256,
            num_stages=1,
            groups=groups,
            groups_b=groups_b,
            W=W,
        )

        shape_q = [BATCH, N_CTX, H, D_HEAD]
        dq = torch.zeros(shape_q, dtype=torch.float32, device=q.device)
        dk_full = torch.zeros(
            [BATCH, N_CTX, H, D_HEAD], dtype=torch.float32, device=q.device
        )

        if has_bias:
            # ★ P0: per-head dB buffer（不再是 head_b），后续一次性 sum
            db_split = torch.zeros(
                [BATCH, N_CTX, H, W], dtype=torch.float32, device=q.device
            )
            sb_arg = sb_bf16
        else:
            db_split = torch.zeros([1, 1, 1, 1], dtype=torch.float32, device=q.device)
            sb_arg = torch.zeros(1, 1, 1, 1, dtype=dtype, device=q.device)

        d_sink_bh = torch.zeros([BATCH, H], dtype=torch.float32, device=q.device)
        kernel(q, k, sb_arg, sink, do, lse, delta, dq, dk_full, db_split, d_sink_bh)
        d_sink = d_sink_bh.sum(dim=0)

        dq = mod_post(dq)
        dk = dk_full.view(BATCH, N_CTX, HEAD_KV, groups, D_HEAD).sum(dim=3).to(dtype)

        if has_bias:
            # bx = hb * groups_b + g，即 view 成 [B,T,H_b,groups_b,W] 后 sum(g)
            d_score_bias = (
                db_split.view(BATCH, N_CTX, H_b, groups_b, W)
                .sum(dim=3)
                .to(dtype)
                .to(ctx.score_bias_orig_dtype)
            )
        else:
            d_score_bias = None

        return dq, dk, d_score_bias, d_sink, None, None


# ============================================================
# Reference（支持 score_bias 为 (B, T, H_b, W)，per-bias-head GQA）
# ============================================================
def ref_program(Q, K, sink, score_bias, is_causal):
    B, T, HQ, D = Q.shape
    q = Q.transpose(1, 2)
    k = K.transpose(1, 2)

    if is_causal:
        idx = torch.arange(T, device=Q.device)
        w_idx = idx.unsqueeze(1) - idx.unsqueeze(0)  # (T, T), q - k
        causal_ok = w_idx >= 0

        if score_bias is not None:
            score_bias = score_bias.to(q.dtype)
            H_b = score_bias.shape[2]
            W = score_bias.shape[3]
            groups_b = HQ // H_b
            in_window = causal_ok & (w_idx < W)  # (T, T)
            w_c = w_idx.clamp(0, W - 1)  # (T, T)

            sb_hq = score_bias.repeat_interleave(groups_b, dim=2)  # (B, T, HQ, W)
            w_c_exp = w_c.view(1, T, 1, T).expand(B, T, HQ, T)
            gathered = sb_hq.gather(3, w_c_exp)  # (B, T, HQ, T)
            bias_dense = torch.where(
                in_window.view(1, T, 1, T), gathered, torch.zeros_like(gathered)
            )
            bias_dense = bias_dense.permute(0, 2, 1, 3)  # (B, HQ, T, T)
        else:
            bias_dense = torch.zeros(B, HQ, T, T, device=Q.device, dtype=q.dtype)

        neg_inf = torch.full((), float("-inf"), device=Q.device, dtype=bias_dense.dtype)
        bias_dense = torch.where(causal_ok.view(1, 1, T, T), bias_dense, neg_inf)
    else:
        bias_dense = torch.zeros(B, HQ, T, T, device=Q.device, dtype=q.dtype)

    sink_expand = sink.view(1, HQ, 1, 1).expand(B, HQ, T, 1).to(bias_dense.dtype)
    bias_with_sink = torch.cat([bias_dense, sink_expand], dim=-1)

    k_pad = F.pad(k, (0, 0, 0, 1))
    v_pad = F.pad(k, (0, 0, 0, 1))

    out = F.scaled_dot_product_attention(
        q,
        k_pad,
        v_pad,
        attn_mask=bias_with_sink,
        is_causal=False,
        enable_gqa=True,
    )
    return out.transpose(1, 2)


# ============================================================
# Main
# ============================================================
def main(
    BATCH: int = 1,
    H: int = 32,
    N_CTX: int = 256,
    D_HEAD: int = 128,
    groups: int = 16,
    causal: bool = False,
    W: int = 128,
    H_b: int = 1,
):
    assert H % H_b == 0, f"H={H} must be divisible by H_b={H_b}"
    use_bias = W > 0
    fwd_flops = 4 * BATCH * H * N_CTX * N_CTX * D_HEAD
    bwd_flops = 10 * BATCH * H * N_CTX * N_CTX * D_HEAD
    if causal:
        fwd_flops *= 0.5
        bwd_flops *= 0.5

    Q = (
        torch.empty(BATCH, N_CTX, H, D_HEAD, dtype=dtype, device="cuda")
        .normal_()
        .requires_grad_()
    )
    head_kv = H // groups
    K = (
        torch.empty(BATCH, N_CTX, head_kv, D_HEAD, dtype=dtype, device="cuda")
        .normal_()
        .requires_grad_()
    )
    dO = (
        torch.empty(BATCH, N_CTX, H, D_HEAD, dtype=dtype, device="cuda")
        .normal_()
        .requires_grad_()
    )
    if use_bias:
        score_bias = (
            torch.randn(BATCH, N_CTX, H_b, W, dtype=torch.float32, device="cuda") * 0.1
        ).requires_grad_()
    else:
        score_bias = None
    sink = torch.full((H,), 0.5, dtype=torch.float32, device="cuda").requires_grad_()

    # ---------- 数值对拍 ----------
    O = convpe_attention(Q, K, sink, score_bias, causal)
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dB = None
    if use_bias:
        dB, score_bias.grad = score_bias.grad.clone(), None
    dS, sink.grad = sink.grad.clone(), None

    O_ref = ref_program(Q, K, sink, score_bias, causal)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dB_ref = None
    if use_bias:
        dB_ref, score_bias.grad = score_bias.grad.clone(), None
    dS_ref, sink.grad = sink.grad.clone(), None

    torch.testing.assert_close(O, O_ref, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(dK, dK_ref, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(dQ, dQ_ref, rtol=2e-2, atol=5e-2)
    if use_bias:
        torch.testing.assert_close(dB, dB_ref, rtol=2e-2, atol=6e-2)
    torch.testing.assert_close(dS, dS_ref, rtol=2e-2, atol=5e-2)

    from tilelang.profiler import do_bench

    def run_fwd():
        with torch.no_grad():
            convpe_attention(Q, K, sink, score_bias, causal)

    def run_bwd():
        O.backward(dO, retain_graph=True)

    latency = do_bench(run_fwd, warmup=500)
    print(f"tilelang fwd: {latency:.2f} ms, {fwd_flops / latency * 1e-9:.2f} TFlops")

    latency = do_bench(run_bwd, warmup=500)
    print(f"tilelang bwd: {latency:.2f} ms, {bwd_flops / latency * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--h", type=int, default=12)
    parser.add_argument("--n_ctx", type=int, default=1024 * 4)
    parser.add_argument("--d_head", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--w", type=int, default=128)
    parser.add_argument(
        "--h_b",
        type=int,
        default=1,
        help="number of bias heads; H must be divisible by H_b",
    )
    args = parser.parse_args()
    for w in [0, 1, 32, 64, 128]:
        for hb in [1, 2, 3, 4, 6, 12]:
            if w == 0 and hb > 1:
                continue
            print(f"w: {w}, hb: {hb}")
            main(
                args.batch,
                args.h,
                args.n_ctx,
                args.d_head,
                args.groups,
                args.causal,
                w,
                hb,
            )

"""
w: 0, hb: 1
tilelang fwd: 0.84 ms, 61.00 TFlops
tilelang bwd: 2.39 ms, 53.81 TFlops
w: 1, hb: 1
tilelang fwd: 0.85 ms, 60.32 TFlops
tilelang bwd: 2.50 ms, 51.47 TFlops
w: 1, hb: 2
tilelang fwd: 0.86 ms, 59.80 TFlops
tilelang bwd: 2.53 ms, 50.99 TFlops
w: 1, hb: 3
tilelang fwd: 0.85 ms, 60.48 TFlops
tilelang bwd: 2.50 ms, 51.48 TFlops
w: 1, hb: 4
tilelang fwd: 0.86 ms, 59.84 TFlops
tilelang bwd: 2.52 ms, 51.16 TFlops
w: 1, hb: 6
tilelang fwd: 0.85 ms, 60.47 TFlops
tilelang bwd: 2.54 ms, 50.67 TFlops
w: 1, hb: 12
tilelang fwd: 0.87 ms, 59.46 TFlops
tilelang bwd: 2.56 ms, 50.26 TFlops
w: 32, hb: 1
tilelang fwd: 0.86 ms, 60.27 TFlops
tilelang bwd: 2.71 ms, 47.57 TFlops
w: 32, hb: 2
tilelang fwd: 0.86 ms, 60.16 TFlops
tilelang bwd: 2.74 ms, 47.01 TFlops
w: 32, hb: 3
tilelang fwd: 0.86 ms, 60.24 TFlops
tilelang bwd: 2.63 ms, 49.05 TFlops
w: 32, hb: 4
tilelang fwd: 0.86 ms, 59.84 TFlops
tilelang bwd: 2.75 ms, 46.93 TFlops
w: 32, hb: 6
tilelang fwd: 0.86 ms, 59.79 TFlops
tilelang bwd: 2.64 ms, 48.79 TFlops
w: 32, hb: 12
tilelang fwd: 0.87 ms, 59.57 TFlops
tilelang bwd: 2.80 ms, 46.05 TFlops
w: 64, hb: 1
tilelang fwd: 0.87 ms, 59.28 TFlops
tilelang bwd: 2.75 ms, 46.85 TFlops
w: 64, hb: 2
tilelang fwd: 0.88 ms, 58.67 TFlops
tilelang bwd: 2.67 ms, 48.23 TFlops
w: 64, hb: 3
tilelang fwd: 0.94 ms, 54.85 TFlops
tilelang bwd: 2.67 ms, 48.22 TFlops
w: 64, hb: 4
tilelang fwd: 0.88 ms, 58.39 TFlops
tilelang bwd: 2.68 ms, 47.99 TFlops
w: 64, hb: 6
tilelang fwd: 0.95 ms, 54.13 TFlops
tilelang bwd: 2.71 ms, 47.53 TFlops
w: 64, hb: 12
tilelang fwd: 0.90 ms, 56.96 TFlops
tilelang bwd: 2.91 ms, 44.29 TFlops
w: 128, hb: 1
tilelang fwd: 0.87 ms, 59.35 TFlops
tilelang bwd: 2.81 ms, 45.82 TFlops
w: 128, hb: 2
tilelang fwd: 0.87 ms, 59.06 TFlops
tilelang bwd: 2.93 ms, 43.99 TFlops
w: 128, hb: 3
tilelang fwd: 0.88 ms, 58.42 TFlops
tilelang bwd: 2.84 ms, 45.39 TFlops
w: 128, hb: 4
tilelang fwd: 0.88 ms, 58.75 TFlops
tilelang bwd: 2.96 ms, 43.49 TFlops
w: 128, hb: 6
tilelang fwd: 0.90 ms, 57.56 TFlops
tilelang bwd: 2.93 ms, 44.00 TFlops
w: 128, hb: 12
tilelang fwd: 0.93 ms, 55.63 TFlops
tilelang bwd: 3.08 ms, 41.80 TFlops
"""
