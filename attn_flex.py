import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    AuxRequest,
)
from functools import lru_cache


def convpe_attention(q, kv, sink, score_bias=None):
    q = q.transpose(1, 2)
    kv = kv.transpose(1, 2)
    q_len, kv_len = q.size(-2), kv.size(-2)
    offset = kv_len - q_len

    if score_bias is None:
        score_mod = None
    else:
        W = score_bias.size(-1)

        def score_mod(score, b, h, q_idx, kv_idx):
            w_idx = kv_idx - (q_idx + offset) + W - 1
            in_window = (w_idx >= 0) & (w_idx < W)
            w_idx_safe = w_idx.clamp(0, W - 1)
            return score + torch.where(
                in_window, score_bias[b, h, q_idx, w_idx_safe], 0
            )

    block_mask = create_causal_mask(q_len, kv_len, offset, q.device)

    o, aux = flex_attention(
        q, kv, kv, score_mod, block_mask,
        enable_gqa=True,
        return_aux=AuxRequest(lse=True),
    )
    scale = torch.exp(aux.lse - torch.logaddexp(aux.lse, sink.view(1, -1, 1)))
    return (o * scale.unsqueeze(-1)).transpose(1, 2)


@torch.compiler.disable
@lru_cache(maxsize=64)
def create_causal_mask(q_len, kv_len, offset, device):
    if q_len == 1:
        return None  # decode 不需要 mask

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx + offset >= kv_idx

    return create_block_mask(mask_mod, None, None, q_len, kv_len, device)