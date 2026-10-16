import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    AuxRequest,
)
from functools import lru_cache


class CausualTransformer(nn.Module):

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dim_head: int = None,
        dim_ffn: int = None,
        rope=None,
        convpe_list: list = None,
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
                    rope,
                    convpe_list[i] if convpe_list is not None else None,
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

    def forward(self, x: Tensor) -> Tensor:
        for l in range(len(self.attns)):
            x = x + self.attns[l](x)
            x = x + self.ffns[l](x)
        return x


class MultiQueryAttention(nn.Module):
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
        self.convpe = convpe
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
        q = self.wq(x).unflatten(-1, (self.num_heads, -1)).transpose(-2, -3)
        q = F.normalize(q, dim=-1)
        kv = self.wkv(x).unsqueeze(-3)
        kv = self.kv_norm(kv)
        if self.rope is not None:
            q = self.rope(q)
            kv = self.rope(kv)
        score_bias = None if self.convpe is None else self.convpe(x)
        o = score_bias_attention(q, kv, self.sink, score_bias)
        if self.rope is not None:
            o = self.rope(o, inverse=True)
        return self.wo(o.transpose(-2, -3).flatten(-2))


class ConvPE(nn.Module):
    """
    dim_shared or dim_headwise == 0 means no ConvPE
    dim_shared or dim_headwise == -1 means use x as projection output
    input: x  (b, s, d)
    output: score_bias (b, h, s, w)
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        window_size: int = 128,
        dim_shared: int = 16,
        dim_headwise: int = 4,
        groups_shared: int = 16,
        groups_headwise: int = 1,
        window_size_out: int = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.pad = nn.ConstantPad1d((window_size - 1, 0), 0)
        window_size_out = window_size_out or window_size

        if dim_shared == 0:
            self.ws = None
            self.convs = None
        elif dim_shared == -1:
            self.ws = None
            self.convs = nn.Conv1d(
                dim_model,
                num_heads * window_size_out,
                window_size,
                groups=groups_shared,
            )
        else:
            self.ws = nn.Linear(dim_model, dim_shared, bias=False)
            self.convs = nn.Conv1d(
                dim_shared,
                num_heads * window_size_out,
                window_size,
                groups=groups_shared,
            )

        if dim_headwise == 0:
            self.wh = None
            self.convh = None
        elif dim_headwise == -1:
            self.wh = None
            self.convh = nn.Conv1d(
                dim_model // num_heads,
                window_size,
                window_size_out,
                groups=groups_headwise,
            )
        else:
            self.wh = nn.Linear(dim_model, num_heads * dim_headwise, bias=False)
            self.convh = nn.Conv1d(
                dim_headwise, window_size, window_size_out, groups=groups_headwise
            )

    def forward(self, x: Tensor):
        if self.convs is not None:
            xs = x if self.ws is None else self.ws(x)
            xs = self.convs(self.pad(xs.mT))
            bs = xs.unflatten(-2, (-1, self.num_heads)).permute(0, 2, 3, 1)
        else:
            bs = 0
        if self.convh is not None:
            xh = x if self.wh is None else self.wh(x)
            xh = xh.mT.unflatten(-2, (self.num_heads, -1)).flatten(0, 1)
            bh = self.convh(self.pad(xh)).unflatten(0, (-1, self.num_heads)).mT
        else:
            bh = 0
        sb = bs + bh
        return None if sb is 0 else sb


class VisionTransformer(nn.Module):

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dim_head: int = None,
        dim_ffn: int = None,
        rope=None,
        convpe_list: list = None,
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
                    convpe_list[i] if convpe_list is not None else None,
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

    def forward(self, x: Tensor) -> Tensor:
        for l in range(len(self.attns)):
            x = x + self.attns[l](x)
            x = x + self.ffns[l](x)
        return x


class ConvPE2D(nn.Module):
    """
    Input : x (b, s, d),  s = H * W
    Output: attention bias (b, h, s, s)
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        H: int,
        W: int,
        window_size: int = 3,
        dim_shared: int = 16,
        dim_headwise: int = 4,
        groups_shared: int = 16,
        groups_headwise: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.H, self.W = H, W
        self.s = H * W

        if dim_shared == 0:
            self.ws = None
            self.convs = None
        elif dim_shared == -1:
            self.ws = None
            self.convs = nn.Conv2d(
                dim_model,
                num_heads * self.s,
                window_size,
                padding=window_size // 2,
                groups=groups_shared,
            )
        else:
            self.ws = nn.Linear(dim_model, dim_shared, bias=False)
            self.convs = nn.Conv2d(
                dim_shared,
                num_heads * self.s,
                window_size,
                padding=window_size // 2,
                groups=groups_shared,
            )

        if dim_headwise == 0:
            self.wh = None
            self.convh = None
        elif dim_headwise == -1:
            self.wh = None
            self.convh = nn.Conv2d(
                dim_model // num_heads,
                self.s,
                window_size,
                padding=window_size // 2,
                groups=groups_headwise,
            )
        else:
            self.wh = nn.Linear(dim_model, num_heads * dim_headwise, bias=False)
            self.convh = nn.Conv2d(
                dim_headwise,
                self.s,
                window_size,
                padding=window_size // 2,
                groups=groups_headwise,
            )

    def forward(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        assert s == self.s, f"s={s} != H*W={self.s}"
        H, W = self.H, self.W
        if self.convs is not None:
            xs = x if self.ws is None else self.ws(x)  # (b, s, C_in)
            xs = xs.reshape(b, H, W, -1).permute(0, 3, 1, 2)  # (b, C_in, H, W)
            xs = self.convs(xs)  # (b, h*s, H, W)
            bs = xs.reshape(b, self.num_heads, self.s, H * W)  # (b, h, s, s)
        else:
            bs = 0
        if self.convh is not None:
            xh = x if self.wh is None else self.wh(x)  # (b, s, h*c_h)
            xh = xh.reshape(b, s, self.num_heads, -1)  # (b, s, h, c_h)
            xh = xh.permute(0, 2, 3, 1)  # (b, h, c_h, s)
            xh = xh.reshape(b * self.num_heads, -1, H, W)  # (b*h, c_h, H, W)
            xh = self.convh(xh)  # (b*h, s, H, W)
            bh = xh.reshape(b, self.num_heads, self.s, H * W)  # (b, h, s, s)
        else:
            bh = 0
        sb = bs + bh
        return None if sb is 0 else sb


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
        self.convpe = convpe
        self.dim_head = dim_head

        self.pre_norm = nn.RMSNorm(dim_model)
        self.wq = nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.wk = nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.wv = nn.Linear(dim_model, num_heads * dim_head)
        self.k_norm = nn.RMSNorm(num_heads * dim_head)
        self.wo = nn.Linear(num_heads * dim_head, dim_model)

    def forward(self, x: Tensor) -> Tensor:
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
        score_bias = None if self.convpe is None else self.convpe(x)
        score_bias = None if score_bias is 0 else score_bias
        o = F.scaled_dot_product_attention(q, k, v, score_bias)
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

    def forward(self, x: Tensor, pos: int = 0, inverse: bool = False) -> Tensor:
        """
        x: (..., s, d)
        pos: start position
        inverse: inverse rotation
        """
        seq_len = x.size(-2)
        if self.max_seq_len >= 0:
            if not hasattr(self, "cos"):
                cos, sin = self.calculate_cos_sin(
                    0, self.max_seq_len, x.device, x.dtype
                )
                self.register_buffer("cos", cos, persistent=False)
                self.register_buffer("sin", sin, persistent=False)
            cos = self.cos[pos : pos + seq_len, :]
            sin = self.sin[pos : pos + seq_len, :]
        else:
            cos, sin = self.calculate_cos_sin(pos, seq_len, x.device, x.dtype)
        return self.apply_rope(x, cos, sin, inverse)

    def extra_repr(self):
        return f"dim_rope={self.dim_rope}, max_seq_len={self.max_seq_len}, online={self.max_seq_len == 0}"

    def calculate_cos_sin(self, pos, seq_len, device, dtype) -> Tensor:
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
        x_rope = x[..., : self.dim_rope]
        x_rope_pairs = x_rope.unflatten(-1, (-1, 2))
        x0 = x_rope_pairs[..., 0]
        x1 = x_rope_pairs[..., 1]
        if inverse:
            out0 = x0 * cos + x1 * sin
            out1 = x1 * cos - x0 * sin
        else:
            out0 = x0 * cos - x1 * sin
            out1 = x1 * cos + x0 * sin
        x_rope_out = torch.stack([out0, out1], dim=-1).flatten(-2)
        return torch.cat([x_rope_out, x[..., self.dim_rope :]], dim=-1)


def score_bias_attention(
    q: Tensor,
    kv: Tensor,
    sink: Tensor,
    score_bias: Tensor = None,
) -> Tensor:
    """
    Args:
        q (Tensor): q.shape = (B, H, T, D)
        kv (Tensor): kv.shape = (B, Hkv, S, D)
        sink (Tensor): sink.shape = (H)
        score_bias (Tensor): score_bias.shape = (B, H, T, Score_Bias_Window_Size)
        is_causal (bool): Defaults to True.
    """

    if score_bias is None:
        score_mod = None
    else:
        W = score_bias.size(-1)

        def score_mod(score, b, h, q_idx, kv_idx):
            w_idx = kv_idx - q_idx + W - 1
            in_window = (w_idx >= 0) & (w_idx < W)
            w_idx_safe = w_idx.clamp(0, W - 1)
            return score + torch.where(
                in_window, score_bias[b, h, q_idx, w_idx_safe], 0
            )

    block_mask = create_causal_mask(q.size(-2), kv.size(-2), q.device)
    o, aux = flex_attention(
        q,
        kv,
        kv,
        score_mod,
        block_mask,
        enable_gqa=True,
        return_aux=AuxRequest(lse=True),
    )
    # attention sink rescaling
    scale = torch.exp(aux.lse - torch.logaddexp(aux.lse, sink.view(1, -1, 1)))
    return o * scale.unsqueeze(-1)


@torch.compiler.disable
@lru_cache(maxsize=64)
def create_causal_mask(q_len, kv_len, device):

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(mask_mod, None, None, q_len, kv_len, device)
    return block_mask
