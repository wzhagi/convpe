from torch import nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    GenerationMixin,
    GenerationConfig,
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformer_convpe import *


class LanguageModelConfig(PretrainedConfig):
    model_type = "language_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = self.num_layers
        self.hidden_size = self.dim_model
        self.tie_word_embeddings = True
        self.is_encoder_decoder = False


class LanguageModel(PreTrainedModel,GenerationMixin):

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
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
    ) -> CausalLMOutputWithPast:
        if isinstance(past_key_values, DynamicCache):
            past_key_values = None
        x = self.tok_emb(input_ids)
        x, current_key_values = self.transformer(
            x, attention_mask, past_key_values, use_cache
        )
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(
            loss, logits, current_key_values if use_cache else None
        )
