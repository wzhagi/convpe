import torch
from torch import Tensor, nn
from torch.nn import functional as F
from attn_tilelang import window_score_bias_attention as attention

# from attn_flex import window_score_bias_attention as attention


class Transformer(nn.Module):

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dim_head: int = None,
        dim_ffn: int = None,
        convpe_win: int = 0,
        convpe_dim: int = 0,
        convpe_heads: int = 1,
        convpe_rank: int = 0,
        convpe_groups: int = 1,
        convpe_share=False,
        rope=None,
        max_seq_len=4096,
        **kwargs,
    ) -> None:
        super().__init__()
        dim_head = dim_head or dim_model // num_heads
        self.num_layers = num_layers
        self.attns = nn.ModuleList(
            [
                MultiQueryAttention(
                    dim_model,
                    num_heads,
                    dim_head,
                    (
                        ConvPE(
                            dim_model,
                            convpe_win,
                            convpe_dim,
                            convpe_heads,
                            convpe_rank,
                            convpe_groups,
                            convpe_share,
                        )
                        if convpe_win > 0 and convpe_dim > 0
                        else None
                    ),
                    rope,
                )
                for i in range(num_layers)
            ]
        )
        self.ffns = nn.ModuleList(
            [
                GatedFeedForward(dim_model, dim_ffn or dim_model * 8 // 3)
                for _ in range(num_layers)
            ]
        )
        self.post_norm = nn.RMSNorm(dim_model)

    def forward(self, x: Tensor) -> Tensor:
        for l in range(len(self.attns)):
            x = x + self.attns[l](x)
            x = x + self.ffns[l](x)
        return self.post_norm(x)


class GatedFeedForward(nn.Module):

    def __init__(self, dim_model, dim_ffn):
        super().__init__()
        self.pre_norm = nn.RMSNorm(dim_model)
        self.wg = nn.Linear(dim_model, dim_ffn)
        self.wv = nn.Linear(dim_model, dim_ffn)
        self.wo = nn.Linear(dim_ffn, dim_model)

    def forward(self, x):
        x = self.pre_norm(x)
        g = F.silu(self.wg(x)).clamp_max_(10)
        v = self.wv(x).clamp_(-10, 10)
        return self.wo(g * v)


class MultiQueryAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        convpe,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.convpe = convpe
        self.rope = rope
        self.dim_head = dim_head

        self.register_parameter("sink", nn.Parameter(torch.ones(num_heads) * -1e4))
        self.pre_norm = nn.RMSNorm(dim_model)
        self.wq = nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.wkv = nn.Linear(dim_model, dim_head, False)
        self.kv_norm = nn.RMSNorm(dim_head)
        self.wo = nn.Linear(num_heads * dim_head, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (b,s,d) q: (b,h,s,d) kv:(b,1,s,d) o:(b,h,s,d)
        x = self.pre_norm(x)
        q = self.wq(x).unflatten(-1, (self.num_heads, -1))
        q = F.normalize(q, dim=-1)
        kv = self.wkv(x).unsqueeze(-2)
        kv = self.kv_norm(kv)
        if self.rope is not None:
            q = self.rope(q)
            kv = self.rope(kv)
        score_bias = None if self.convpe is None else self.convpe(x).bfloat16()
        o = attention(q.bfloat16(), kv.bfloat16(), self.sink, score_bias)
        if self.rope is not None:
            o = self.rope(o, inverse=True)
        return self.wo(o.flatten(-2))


class ConvPE(nn.Module):
    """
    input:  x (b, s, d)
    output: score_bias (b, s, hc, w)
    """

    def __init__(
        self,
        dim_model,
        window_size,
        dim_convpe,
        heads=1,
        rank=0,
        groups=1,
        share=False,
    ):
        super().__init__()
        self.out_dim = heads * window_size
        assert self.out_dim % window_size == 0
        assert self.out_dim % groups == 0
        assert dim_convpe % groups == 0
        self.window_size = window_size
        self.dim_convpe = dim_convpe
        self.groups = groups
        self.rank = rank
        self.share_weights = share
        self.heads = heads
        self.kernel_in = dim_convpe // groups
        self.kernel_out = self.out_dim // groups
        in_dim = self.kernel_in * window_size

        self.proj = nn.Linear(dim_model, dim_convpe, bias=False)
        if rank == 0:
            self.num_kernels = 1 if share else groups
            self.conv = nn.Linear(in_dim, self.kernel_out * self.num_kernels)
        else:
            self.convd = nn.Linear(in_dim, rank, bias=False)
            self.convu = nn.Linear(rank, self.kernel_out * groups)

    def forward(self, x):
        b, s, _ = x.shape
        w, g, ki, ko = self.window_size, self.groups, self.kernel_in, self.kernel_out
        xu = F.pad(self.proj(x), (0, 0, w - 1, 0)).unfold(1, w, 1).view(b, s, g, ki, w)
        if self.rank == 0:
            if self.num_kernels == 1:
                cw = self.conv.weight.view(ko, ki, w)
                o = torch.einsum("bsgcw,ocw->bsgo", xu, cw)
                o = o + self.conv.bias.view(1, 1, 1, ko)
            else:
                cw = self.conv.weight.view(g, ko, ki, w)
                o = torch.einsum("bsgcw,gocw->bsgo", xu, cw)
                o = o + self.conv.bias.view(1, 1, g, ko)
            return o.reshape(b, s, self.heads, w)
        else:
            dw = self.convd.weight.view(self.rank, ki, w)
            tmp = torch.einsum("bsgcw,rcw->bsgr", xu, dw)
            uw = self.convu.weight.view(g, ko, self.rank)
            o = torch.einsum("bsgr,gor->bsgo", tmp, uw)
            o = o + self.convu.bias.view(1, 1, g, ko)
            return o.reshape(b, s, self.heads, w)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_rope: int,
        max_seq_len: int = 0,
        base: int = 10000,
        base_dim: int = 64,
    ):
        super().__init__()
        self.dim_rope = dim_rope
        self.base = base
        self.base_dim = base_dim
        self.max_seq_len = max_seq_len
        self.cos = None
        self.sin = None

    def forward(self, x: Tensor, pos: int = 0, inverse: bool = False) -> Tensor:
        """
        x: (..., s, d)
        pos: start position
        inverse: inverse rotation
        """
        seq_len = x.size(-3)
        if self.max_seq_len > 0:
            if self.cos is None or self.cos.device != x.device:
                self.cos, self.sin = self.calculate_cos_sin(
                    0, self.max_seq_len, x.device
                )
            cos = self.cos[pos : pos + seq_len, :]
            sin = self.sin[pos : pos + seq_len, :]
        else:
            cos, sin = self.calculate_cos_sin(pos, seq_len, x.device)
        return self.apply_rope(x, cos, sin, inverse)

    def extra_repr(self):
        return f"dim_rope={self.dim_rope}, max_seq_len={self.max_seq_len}, online={self.max_seq_len == 0}"

    def calculate_cos_sin(self, pos, seq_len, device):
        dtype = torch.float32
        theta = self.base ** -(
            torch.arange(0, self.dim_rope, 2, device=device, dtype=dtype)
            / self.base_dim
        )
        seq_idx = torch.arange(pos, pos + seq_len, device=device, dtype=dtype)
        idx_theta = torch.outer(seq_idx, theta)
        cos = torch.cos(idx_theta)
        sin = torch.sin(idx_theta)
        return cos, sin

    def apply_rope(self, x: Tensor, cos, sin, inverse=False):
        dtype = x.dtype
        x_rope = x[..., : self.dim_rope].to(torch.float32)
        x_rope_pairs = x_rope.unflatten(-1, (-1, 2))
        x0 = x_rope_pairs[..., 0]
        x1 = x_rope_pairs[..., 1]
        cos = cos.unsqueeze(0).unsqueeze(-2)
        sin = sin.unsqueeze(0).unsqueeze(-2)
        if inverse:
            out0 = x0 * cos + x1 * sin
            out1 = x1 * cos - x0 * sin
        else:
            out0 = x0 * cos - x1 * sin
            out1 = x1 * cos + x0 * sin
        x_rope_out = torch.stack([out0, out1], dim=-1).flatten(-2).to(dtype)
        return torch.cat([x_rope_out, x[..., self.dim_rope :]], dim=-1)
