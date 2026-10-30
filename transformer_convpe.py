from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from convpe_exp import ConvPE
from attn_tilelang2 import convpe_attention as attention

# from attn_tilelang_fix_seq_len import convpe_attention as attention
# from attn_flex import convpe_attention as attention


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

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple]]:
        current_key_values = [] if use_cache else None
        for l in range(len(self.attns)):
            if past_key_values is not None:
                kvcache, sbcache = past_key_values[l]
            else:
                kvcache, sbcache = None, None
            attn_out, new_kvcache, new_sbcache = self.attns[l](
                x, mask=mask, kvcache=kvcache, sbcache=sbcache, use_cache=use_cache
            )
            x = x + attn_out
            x = x + self.ffns[l](x)
            if use_cache:
                current_key_values.append((new_kvcache, new_sbcache))
        x = self.post_norm(x)
        current_key_values = tuple(current_key_values) if use_cache else None
        return x, current_key_values


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

        self.register_parameter("sink", nn.Parameter(torch.zeros(num_heads)))
        self.pre_norm = nn.RMSNorm(dim_model)
        self.wq = nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.wkv = nn.Linear(dim_model, dim_head, False)
        self.kv_norm = nn.RMSNorm(dim_head)
        self.wo = nn.Linear(num_heads * dim_head, dim_model)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kvcache: Optional[Tensor] = None,
        sbcache: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # x: (b,s,d) q: (b,s,h,d) kv:(b,s,1,d) o:(b,s,h,d)
        if kvcache is not None:
            pos = kvcache.size(1)
        else:
            pos = 0
        x = self.pre_norm(x)
        q = self.wq(x).unflatten(-1, (self.num_heads, -1))
        q = F.normalize(q, dim=-1)
        kv = self.kv_norm(self.wkv(x)).unsqueeze(-2)
        if self.rope is not None:
            q = self.rope(q, pos=pos)
            kv = self.rope(kv, pos=pos)
        if kvcache is not None:
            kv = torch.cat([kvcache, kv], dim=1)
        new_kvcache = kv if use_cache else None
        if self.convpe is None:
            score_bias, new_sbcache = None, None
        else:
            score_bias, new_sbcache = self.convpe(x, sbcache, use_cache)
        o = attention(q, kv, self.sink, score_bias)
        if self.rope is not None:
            o = self.rope(o, pos=pos, inverse=True)
        return self.wo(o.flatten(-2)), new_kvcache, new_sbcache


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
        x: (...,s, h, d)
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
        cos = torch.cos(idx_theta).unsqueeze(-2)
        sin = torch.sin(idx_theta).unsqueeze(-2)
        return cos, sin  # (s, 1, d)

    def apply_rope(self, x: Tensor, cos, sin, inverse=False):
        dtype = x.dtype  # (...,s, h, d)
        x_rope = x[..., : self.dim_rope].to(torch.float32)
        x_rope_pairs = x_rope.unflatten(-1, (-1, 2))
        x0 = x_rope_pairs[..., 0]  # (...,s, h, d)
        x1 = x_rope_pairs[..., 1]  # (...,s, h, d)
        if inverse:
            out0 = x0 * cos + x1 * sin
            out1 = x1 * cos - x0 * sin
        else:
            out0 = x0 * cos - x1 * sin
            out1 = x1 * cos + x0 * sin
        x_rope_out = torch.stack([out0, out1], dim=-1).flatten(-2).to(dtype)
        return torch.cat([x_rope_out, x[..., self.dim_rope :]], dim=-1)
