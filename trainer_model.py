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
        self.transformer = Transformer(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dim_ffn=config.dim_model * 8 // 3,
            convpe_win=config.convpe_win,
            convpe_dim=config.convpe_dim,
            convpe_heads=config.convpe_heads,
            convpe_rank=config.convpe_rank,
            convpe_groups=config.convpe_groups,
            convpe_share=config.convpe_share,
            rope=self.rope,
            max_seq_len=config.max_seq_len,
        )
        self.lm_head = nn.Linear(config.dim_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self, input_ids, labels=None, attention_mask=None
    ) -> CausalLMOutputWithPast:
        x = self.tok_emb(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(loss, logits)
