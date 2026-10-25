"""
benchmark_convpe.py

固定投影维度下对比 ConvPE 各配置的 fwd / bwd 耗时、显存与 TFLOPS。
默认 autocast bf16：模型/输入保持 fp32，forward 在 torch.autocast 下执行。

用法：
    python benchmark_convpe.py
    python benchmark_convpe.py --amp none
    python benchmark_convpe.py --compile --compile_mode max-autotune
    python benchmark_convpe.py --groups 1 2 4 8 12 --ranks 0 8
"""

import argparse
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConvPE
# ---------------------------------------------------------------------------
class ConvPE(nn.Module):
    """
    input:  x (b, s, dm)
    output: score_bias (b, s, h, w),  h = out_dim // w

    - groups：唯一分组因子；h 仅在最后 reshape 出现
    - tie_weights=True : g 组共享同一个卷积核（跨组对应位置 tied）
    - tie_weights=False: g 组各自独立卷积核（标准 group conv）
    - rank > 0         : 每组卷积核低秩分解
    """

    def __init__(
        self,
        dim_model,
        window_size,
        heads,
        dim_convpe,
        rank=0,
        groups=1,
        tie_weights=False,
    ):
        super().__init__()
        w, g = window_size, groups
        out_dim = heads * window_size
        assert out_dim % w == 0
        assert out_dim % g == 0
        assert dim_convpe % g == 0

        self.window_size, self.out_dim = w, out_dim
        self.dim_convpe, self.groups = dim_convpe, g
        self.rank, self.tie_weights = rank, tie_weights
        self.heads = out_dim // w
        self.c  = dim_convpe // g
        self.og = out_dim // g

        self.proj = nn.Linear(dim_model, dim_convpe, bias=False)
        in_dim = self.c * w

        if rank == 0:
            self.nk = 1 if (tie_weights and g > 1) else g
            self.conv = nn.Linear(in_dim, self.og * self.nk)
            self.convd = None
        else:
            # low rank：down shared，up per-group
            assert rank <= in_dim, \
                f"rank({rank}) > 每组输入维 c*w({in_dim})，low rank 会膨胀"
            self.conv = None
            self.convd = nn.Linear(in_dim, rank, bias=False)   # shared
            self.convu = nn.Linear(rank, self.og * g)          # per-group

    def forward(self, x):
        b, s, _ = x.shape
        w, g, c, og, r = self.window_size, self.groups, self.c, self.og, self.rank

        x = self.proj(x).view(b, s, g, c)
        x = F.pad(x, (0, 0, 0, 0, w - 1, 0))
        xu = x.unfold(1, w, 1)  # (b,s,g,c,w)

        if r == 0:
            cw = self.conv.weight.view(self.nk, og, c, w)
            bias = self.conv.bias
            if self.nk == 1:
                o = torch.einsum("bsgcw,ocw->bsgo", xu, cw[0])
                o = o + bias.view(1, 1, 1, og)
            else:
                o = torch.einsum("bsgcw,gocw->bsgo", xu, cw)
                o = o + bias.view(1, 1, g, og)
        else:
            dw = self.convd.weight                             # (r, c*w)，shared
            uw = self.convu.weight.view(g, og, r)              # (g, og, r)，per-group
            xu_flat = xu.reshape(b, s, g, c * w)
            tmp = torch.einsum("bsgf,rf->bsgr", xu_flat, dw)   # shared down
            o   = torch.einsum("bsgr,gor->bsgo", tmp, uw)      # per-group up
            o   = o + self.convu.bias.view(1, 1, g, og)

        return o.reshape(b, s, self.heads, w)


# ---------------------------------------------------------------------------
# FLOPs 计算（手算，1 MAC = 2 FLOPs；bwd ≈ 2 × fwd）
# ---------------------------------------------------------------------------
def count_flops(b, s, dm, D, out_dim, w, g, rank):
    """
    返回 (fwd_flops, bwd_flops)，单位 FLOPs。

    fwd 组成：
      proj:  2 * b*s*dm*D                        (bias=False，无加法)
      conv:  rank=0 → 2 * b*s*out_dim*D*w // g   (MACs = b*s*out_dim*(D/g)*w)
             rank>0 → 2 * b*s*r*(D*w + out_dim)  (down: b*s*r*D*w, up: b*s*out_dim*r)
      bias:  b*s*out_dim                         (conv 的 bias 加法)
    bwd ≈ 2 × fwd（grad_x 与 grad_W 各一次，均为 fwd 量级的 MACs）。
    注意：rank>0 时 FLOPs 与 g 无关（g*c = D 且 g*og = out_dim，抵消）。
    """
    fwd_proj = 2 * b * s * dm * D
    if rank == 0:
        fwd_conv = 2 * b * s * out_dim * D * w // g
    else:
        fwd_conv = 2 * b * s * rank * (D * w + out_dim)
    fwd_bias = b * s * out_dim
    fwd = fwd_proj + fwd_conv + fwd_bias
    bwd = 2 * fwd
    return fwd, bwd


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def make_amp_ctx(amp, device):
    if amp == "none":
        return nullcontext()
    if amp == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    raise ValueError(f"unknown amp: {amp}")


def make_configs(groups_list, dim_convpe, out_dim, window_size, rank_values):
    configs = []
    for g in groups_list:
        if dim_convpe % g != 0 or out_dim % g != 0:
            continue
        c = dim_convpe // g
        og = out_dim // g
        max_rank = min(c * window_size, og)
        for tie in (False, True):
            for r in rank_values:
                if r > 0 and r > max_rank:
                    continue
                configs.append((g, tie, r))
    return configs


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_fwd(model, x, amp_ctx, iters, warmup, use_cuda):
    for _ in range(warmup):
        with amp_ctx:
            _ = model(x)
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with amp_ctx:
            _ = model(x)
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def time_fwd_bwd(model, x, grad_out, amp_ctx, iters, warmup, use_cuda):
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(x)
        out.backward(grad_out)
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(x)
        out.backward(grad_out)
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim_model", type=int, default=768)
    p.add_argument("--dim_convpe", type=int, default=16)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--groups", type=int, nargs="+", default=[1, 4, 12])
    p.add_argument("--ranks", type=int, nargs="+", default=[0, 512, 256, 128])
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", type=str, default="default")
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    out_dim = args.heads * args.window_size

    print(
        f"# dim_model={args.dim_model}  dim_convpe={args.dim_convpe}  "
        f"w={args.window_size}  heads={args.heads}  out_dim={out_dim}"
    )
    print(
        f"# batch={args.batch}  seq={args.seq}  device={device}  "
        f"amp={args.amp}  compile={args.compile}({args.compile_mode})"
    )
    print(f"# model dtype=fp32 (master weights)，forward 在 autocast 下执行")
    print(f"# FLOPs: 1 MAC=2 FLOPs；bwd≈2×fwd；TFLOPS = FLOPs / 时间 / 1e12\n")

    x = torch.randn(
        args.batch, args.seq, args.dim_model, device=device, dtype=torch.float32
    )

    configs = make_configs(
        args.groups, args.dim_convpe, out_dim, args.window_size, args.ranks
    )

    header = (
        f"{'g':>3} {'tie':>5} {'rank':>4} {'#params':>10} "
        f"{'fwd_GF':>8} {'fwd(ms)':>9} {'fwd_TF/s':>8} "
        f"{'fwd+bwd(ms)':>12} {'fwd+bwd_TF/s':>8} {'mem(MB)':>9}"
    )
    print(header)
    print("-" * len(header))

    for g, tie, r in configs:
        model = ConvPE(
            args.dim_model,
            args.window_size,
            args.heads,
            args.dim_convpe,
            rank=r,
            groups=g,
            tie_weights=tie,
        ).to(device=device)
        n_params = count_params(model)

        if args.compile:
            model = torch.compile(model, mode=args.compile_mode)

        amp_ctx = make_amp_ctx(args.amp, device)

        # 编译触发 + 确定 grad dtype
        model.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(x)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)

        # 计时
        fwd_t = time_fwd(model, x, amp_ctx, args.iters, args.warmup, use_cuda)
        fb_t = time_fwd_bwd(
            model, x, grad_out, amp_ctx, args.iters, args.warmup, use_cuda
        )

        # FLOPs / TFLOPS
        fwd_flops, bwd_flops = count_flops(
            args.batch,
            args.seq,
            args.dim_model,
            args.dim_convpe,
            out_dim,
            args.window_size,
            g,
            r,
        )
        fwd_tflops = fwd_flops / fwd_t / 1e12
        fb_tflops = (fwd_flops + bwd_flops) / fb_t / 1e12

        # 显存
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            model.zero_grad(set_to_none=True)
            with amp_ctx:
                out = model(x)
            out.backward(grad_out)
            torch.cuda.synchronize()
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            mem_mb = 0.0

        print(
            f"{g:>3} {str(tie):>5} {r:>4} {n_params:>10} "
            f"{fwd_flops/1e9:>8.1f} {fwd_t*1000:>9.3f} {fwd_tflops:>8.2f} "
            f"{fb_t*1000:>12.3f} {fb_tflops:>8.2f} {mem_mb:>9.1f}"
        )

    print("\n# GFwd   : 前向 GFLOPs（手算）")
    print("# fwd_TF : 前向 TFLOPS = GFwd / fwd_time")
    print("# fb_TF  : (fwd+bwd) TFLOPS = (GFwd + GBwd) / (fwd+bwd)_time")
    print("# 理论峰值参考：A100 bf16≈312 TFLOPS, H100 bf16≈989 TFLOPS (dense)")


if __name__ == "__main__":
    main()

# dim_model=768  dim_convpe=16  w=128  heads=12  out_dim=1536
# batch=4  seq=1024  device=cuda  amp=bf16  compile=False(default)
# model dtype=fp32 (master weights)，forward 在 autocast 下执行
# FLOPs: 1 MAC=2 FLOPs；bwd≈2×fwd；TFLOPS = FLOPs / 时间 / 1e12

#   g   tie rank    #params   fwd_GF   fwd(ms) fwd_TF/s  fwd+bwd(ms) fwd+bwd_TF/s   mem(MB)
# -----------------------------------------------------------------------------------------
#   1 False    0    3159552     25.9     0.534    48.44        1.596    48.63     212.1
#   1 False  512    1848832     15.1     0.375    40.37        1.084    41.90     209.6
#   1 False  256     931328      7.6     0.274    27.83        0.888    25.77     201.8
#   1 False  128     472576      3.9     0.281    13.78        0.889    13.04     197.7
#   1  True    0    3159552     25.9     0.526    49.20        1.577    49.23     212.1
#   1  True  512    1848832     15.1     0.371    40.82        1.075    42.26     209.6
#   1  True  256     931328      7.6     0.272    28.04        0.879    26.02     201.8
#   1  True  128     472576      3.9     0.279    13.84        0.819    14.15     197.7
#   4 False    0     800256      6.5     0.363    18.05        0.919    21.37     199.7
#   4 False  256     538112      7.6     0.338    22.54        0.938    24.38     205.2
#   4 False  128     275968      3.9     0.294    13.15        0.914    12.69     199.7
#   4  True    0     209280      6.5     0.231    28.32        0.717    27.40     195.2
#   4  True  256     538112      7.6     0.342    22.27        0.935    24.47     205.2
#   4  True  128     275968      3.9     0.292    13.24        0.885    13.11     199.7

# GFwd   : 前向 GFLOPs（手算）
# fwd_TF : 前向 TFLOPS = GFwd / fwd_time
# fb_TF  : (fwd+bwd) TFLOPS = (GFwd + GBwd) / (fwd+bwd)_time
# 理论峰值参考：A100 bf16≈312 TFLOPS, H100 bf16≈989 TFLOPS (dense)