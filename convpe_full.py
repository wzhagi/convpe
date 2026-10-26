import torch
from torch import Tensor, nn
from torch.nn import functional as F


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

    def forward(self, x: Tensor) -> Tensor:
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
