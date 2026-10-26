import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvPE(nn.Module):
    """
    input:  x (b, s, d)
    output: score_bias (b, s, 1, w)
    """

    def __init__(self, dim_model, window_size, dim_convpe, *args, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.dim_convpe = dim_convpe
        self.proj = nn.Linear(dim_model, dim_convpe, bias=False)
        self.conv = nn.Linear(dim_convpe * window_size, window_size)

    def forward(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        w = self.window_size
        c = self.dim_convpe
        x = self.proj(x)
        x = F.pad(x, (0, 0, w - 1, 0))
        x = x.unfold(1, w, 1)
        out = torch.einsum("bscw,ocw->bso", x, self.conv.weight.view(w, c, w))
        out = out + self.conv.bias
        return out.unsqueeze(2)
