from torch import nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformer_convpe import *


class LanguageModelConfig(PretrainedConfig):
    model_type = "language_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tie_word_embeddings = True


class LanguageModel(PreTrainedModel):

    config_class = LanguageModelConfig
    _tied_weights_keys = {"lm_head.weight": "tok_emb.weight"}

    def __init__(self, config: LanguageModelConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim_model)
        if config.dim_rope > 0:
            self.rope = RotaryPositionalEmbedding(config.dim_rope, config.max_seq_len)
        else:
            self.rope = None

        self.attns = nn.ModuleList(
            [
                MultiQueryAttention(
                    config.dim_model,
                    config.num_heads,
                    config.dim_head,
                    self.rope,
                    (
                        ConvPE(
                            config.dim_model,
                            config.num_heads,
                            config.convpe_window_size,
                            config.convpe_dim_shared,
                            config.convpe_dim_headwise,
                            config.convpe_groups_shared,
                            config.convpe_groups_headwise,
                        )
                        if config.convpe_window_size > 0
                        else None
                    ),
                )
                for _ in range(config.num_layers)
            ]
        )

        self.ffns = nn.ModuleList(
            [
                GatedFeedForward(
                    config.dim_model,
                    (
                        config.dim_ffn
                        if hasattr(config, "dim_ffn")
                        else config.dim_model * 8 // 3
                    ),
                )
                for _ in range(config.num_layers)
            ]
        )
        self.post_norm = nn.LayerNorm(config.dim_model)
        self.lm_head = nn.Linear(config.dim_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, labels=None, **kwargs) -> CausalLMOutputWithPast:
        x = self.tok_emb(input_ids)
        x = self.post_norm(x)
        for l in range(len(self.attns)):
            x = x + self.attns[l](x)
            x = x + self.ffns[l](x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(loss, logits)
