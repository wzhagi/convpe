import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvPE(nn.Module):
    """
    input:  x (b, s, d)
    output: score_bias (b, s, heads, window_size)
    """

    def __init__(
        self,
        dim_model: int,
        window_size: int,
        dim_convpe: int,
        bias_heads: int = 1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.dim_convpe = dim_convpe
        self.heads = bias_heads
        self.proj = nn.Linear(dim_model, dim_convpe, bias=False)
        self.conv = nn.Linear(dim_convpe * window_size, bias_heads * window_size)

    def forward(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        w, c, o = self.window_size, self.dim_convpe, self.conv.out_features
        xu = F.pad(self.proj(x), (0, 0, w - 1, 0)).unfold(1, w, 1)
        cw = self.conv.weight.view(o, c, w)
        bias = torch.einsum("bscw,ocw->bso", xu, cw)
        bias = bias + self.conv.bias.view(1, 1, o)
        return bias.reshape(b, s, self.heads, w)
