import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.attention.flex_attention import flex_attention


class ConvPE2D(nn.Module):
    def __init__(self, dim_model, window_size, dim_convpe, bias_heads=1):
        super().__init__()
        assert window_size % 2 == 1
        self.window_size = window_size
        self.num_bias_heads = bias_heads
        self.proj = nn.Linear(dim_model, dim_convpe, bias=False)
        self.conv = Conv2d(
            dim_convpe,
            bias_heads * window_size**2,
            window_size,
            padding=window_size // 2,
            stride=1,
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        b, _, _ = x.shape
        hb, w = self.num_bias_heads, self.window_size
        x = self.proj(x).mT.unflatten(-1, (H, W))
        return self.conv(x).reshape(b, hb, w, w, H, W)


class Conv2d(nn.Module):
    def __init__(
        self, channels_in, channels_out, kernel_size, padding, stride, bias=True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.linear = nn.Linear(
            channels_in * kernel_size * kernel_size, channels_out, bias=bias
        )

    def forward(self, x):
        w = self.kernel_size
        weight = self.linear.weight.view(self.linear.out_features, -1, w, w)
        return F.conv2d(x, weight, self.linear.bias, self.stride, self.padding)


class VisionTransformer(nn.Module):

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dim_head: int = None,
        dim_ffn: int = None,
        rope=None,
        convpe_win: int = 0,
        convpe_dim: int = 0,
        convpe_heads: int = 1,
        max_seq_len=4096,
        **kwargs,
    ) -> None:
        super().__init__()
        dim_head = dim_head or dim_model // num_heads
        self.num_layers = num_layers
        self.attns = nn.ModuleList(
            [
                MultiHeadSelfAttention(
                    dim_model,
                    num_heads,
                    dim_head,
                    rope,
                    (
                        ConvPE2D(dim_model, convpe_win, convpe_dim, convpe_heads)
                        if convpe_win > 0 and convpe_dim > 0
                        else None
                    ),
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

    def forward(self, x: Tensor, H, W) -> Tensor:
        for l in range(len(self.attns)):
            x = x + self.attns[l](x, H, W)
            x = x + self.ffns[l](x)
        return self.post_norm(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        rope=None,
        convpe=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.rope = rope
        self.convpe: ConvPE2D = convpe
        self.dim_head = dim_head

        self.pre_norm = nn.RMSNorm(dim_model)
        self.wq = nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.wk = nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.wv = nn.Linear(dim_model, num_heads * dim_head)
        self.k_norm = nn.RMSNorm(num_heads * dim_head)
        self.wo = nn.Linear(num_heads * dim_head, dim_model)

    def forward(self, x: Tensor, H, W) -> Tensor:
        # x: (b,s,d) q: (b,h,s,d) kv:(b,1,s,d) o:(b,h,s,d)
        x = self.pre_norm(x)
        q = self.wq(x).unflatten(-1, (self.num_heads, -1)).transpose(-2, -3)
        q = F.normalize(q, dim=-1)
        k = (
            self.k_norm(self.wk(x))
            .unflatten(-1, (self.num_heads, -1))
            .transpose(-2, -3)
        )
        v = self.wv(x).unflatten(-1, (self.num_heads, -1)).transpose(-2, -3)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        score_bias = None if self.convpe is None else self.convpe(x, H, W)
        o = convpe2d_attention(q, k, v, score_bias)
        return self.wo(o.transpose(-2, -3).flatten(-2))


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
        seq_len = x.size(-2)
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
        if inverse:
            out0 = x0 * cos + x1 * sin
            out1 = x1 * cos - x0 * sin
        else:
            out0 = x0 * cos - x1 * sin
            out1 = x1 * cos + x0 * sin
        x_rope_out = torch.stack([out0, out1], dim=-1).flatten(-2).to(dtype)
        return torch.cat([x_rope_out, x[..., self.dim_rope :]], dim=-1)


def convpe2d_attention(
    q: Tensor,  # bhsd
    k: Tensor,  # bhsd
    v: Tensor,  # bhsd
    score_bias: Tensor,  # (B, hb, w, w, H, W)
    group_size: int = 1,
) -> Tensor:
    B, hb, w, _, H, W = score_bias.shape
    assert w % 2 == 1
    pad = w // 2
    w_min, w_max = 0, w - 1
    bf = score_bias.reshape(B, hb, w, w, H * W)

    def score_mod(score, b, h, q_idx, kv_idx):
        hb_idx = h // group_size
        qy = q_idx // W
        qx = q_idx % W
        ky = kv_idx // W
        kx = kv_idx % W
        di = qy - ky
        dj = qx - kx
        ki = (di + pad).clamp(w_min, w_max)
        kj = (dj + pad).clamp(w_min, w_max)
        in_window = (di.abs() <= pad) & (dj.abs() <= pad)
        val = bf[b, hb_idx, ki, kj, q_idx]
        return score + torch.where(in_window, val, torch.zeros_like(val))

    return flex_attention(q, k, v, score_mod=score_mod)
