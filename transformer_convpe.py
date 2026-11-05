from typing import Optional, Tuple
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from convpe_exp import ConvPE
from attn_tilelang import convpe_attention as attention

# from attn_tilelang_pretrain import convpe_attention as attention
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


class RoPE(nn.Module):
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


class YaRN(RoPE):
    """
    YaRN baseline，所有依赖参数自动从 max_seq_len / original_max_len 推导：
      - scale    = max_seq_len / original_max_len
      - mscale   = 0.1 * ln(scale) + 1               (scale > 1 时)
      - low/high = beta_fast / beta_slow 对应的 correction range（浮点，不取整）
    beta_fast / beta_slow 保持默认 32 / 1（1K→8K 场景下无需调整）。
    inverse 与 mscale 严格对称：inverse(forward(x)) == x。
    """

    def __init__(
        self,
        dim_rope: int = 64,
        max_seq_len: int = 8192,
        original_max_len: int = 1024,
        base: int = 10000,
        base_dim: int = 64,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__(dim_rope, max_seq_len, base, base_dim)
        assert (
            max_seq_len >= original_max_len > 0
        ), "要求 max_seq_len >= original_max_len > 0"
        self.original_max_len = original_max_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # ---- 自动推导 ----
        self.scale = max_seq_len / original_max_len
        self.mscale = self._get_mscale(self.scale)
        self._L = self.dim_rope // 2
        self._low, self._high = self._compute_correction_range()

    @staticmethod
    def _get_mscale(scale: float) -> float:
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def _find_correction_dim(self, num_rotations: float) -> float:
        """返回 pair index（浮点，在 [0, dim_rope/2) 范围内），对应周期
        T = original_max_len / num_rotations。不取整，避免 ramp 量化误差。"""
        denom = num_rotations * 2 * math.pi
        if self.original_max_len <= denom:
            return 0.0
        return (
            self.base_dim
            * math.log(self.original_max_len / denom)
            / (2 * math.log(self.base))
        )

    def _compute_correction_range(self):
        # 保留浮点，不取整；与 YaRN 参考实现（DeepSeek / HF）对齐
        low = max(0.0, self._find_correction_dim(self.beta_fast))
        high = min(float(self._L - 1), self._find_correction_dim(self.beta_slow))
        if high <= low:
            high = low + 1e-3
        return low, high

    def calculate_cos_sin(self, pos, seq_len, device):
        dtype = torch.float32
        theta = self.base ** -(
            torch.arange(0, self.dim_rope, 2, device=device, dtype=dtype)
            / self.base_dim
        )
        if self.scale > 1.0:
            # 高频区外推、低频区插值、中间线性 ramp 过渡
            theta_extra = theta.clone()          # 外推（不缩放）
            theta_inter = theta / self.scale     # 插值（线性缩放）
            ramp = torch.clamp(
                (torch.arange(self._L, device=device, dtype=dtype) - self._low)
                / (self._high - self._low),      # 浮点 low/high，无 +1e-8 污染
                0.0,
                1.0,
            )
            theta = theta_inter * ramp + theta_extra * (1 - ramp)

        seq_idx = torch.arange(pos, pos + seq_len, device=device, dtype=dtype)
        idx_theta = torch.outer(seq_idx, theta)
        return (
            torch.cos(idx_theta).unsqueeze(-2),
            torch.sin(idx_theta).unsqueeze(-2),
        )

    def forward(self, x: Tensor, pos: int = 0, inverse: bool = False) -> Tensor:
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

        if inverse:
            cos = cos / self.mscale
            sin = sin / self.mscale
        else:
            cos = cos * self.mscale
            sin = sin * self.mscale
        return self.apply_rope(x, cos, sin, inverse)