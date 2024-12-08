
import os
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from math import sqrt

__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']


import torch
from torch import nn
import math
import numpy as np
from einops import rearrange

import torch




class LlamaAttentionTopk(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.token_budget = config.token_budget
        self.init_budget = config.init_budget
        self.recent_budget = config.recent_budget


        self.print_offloading_flag = False
        self.offloading_length = 20000

    def __repr__(self):
        return f"{super().__repr__()}\nSparsification Setting(topk:{self.token_budget}  edge:{self.init_budget, self.recent_budget})"

    def ensure_gpu(self, past_key_value, device):
        if (past_key_value is not None 
            and  len(past_key_value.key_cache) > self.layer_idx 
            and (not past_key_value.key_cache[self.layer_idx].is_cuda)):
            #print("onboarding layer", self.layer_idx)
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].to(device)
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].to(device)

    def offload_if_necessary_cpu(self, past_key_value):
        if (past_key_value is not None 
            and  len(past_key_value.key_cache) > self.layer_idx  
            and past_key_value.key_cache[self.layer_idx].shape[2] >=self.offloading_length):
            if self.print_offloading_flag == False and self.layer_idx == 0:
                print("OFFLOADING ENABLED >>")
                self.print_offloading_flag = True
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].cpu()
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].cpu()

    def _reset_state(self):
        self.print_offloading_flag = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        self.ensure_gpu(past_key_value, hidden_states.device)
        if q_len > 1:
            return_value =  self.flash_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            self.offload_if_necessary_cpu(past_key_value)
            return return_value

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask


        kv_seq_len=key_states.shape[2]
        attn_weights_for_topk = attn_weights.clone() # clone as we will mess with this
        causal_heavy_recent_mask = torch.tril(torch.ones(q_len,kv_seq_len,device=attn_weights.device), diagonal=kv_seq_len-q_len-self.recent_budget).bool()
        causal_heavy_recent_mask[:,:self.init_budget] = False # True => mask , False => use
        attn_weights_for_topk.masked_fill_(torch.logical_not(causal_heavy_recent_mask), torch.finfo(attn_weights.dtype).min)
        _, indices = attn_weights_for_topk.topk(self.token_budget, dim=-1)

        mask = torch.ones_like(attn_weights).bool()
        mask.scatter_(3, indices, False)
        mask[:,:,:,:self.init_budget] = False
        mask[:,:,:,-self.recent_budget:] = False
        attn_weights.masked_fill_(mask, torch.finfo(attn_weights.dtype).min) # mask what is set true for final attention


        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def convert_exact_topk(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_exact_topk(module, config)

        if isinstance(module, LlamaAttention):
            device = next(module.parameters()).device
            new_module = LlamaAttentionTopk(config, module.layer_idx).bfloat16().to(device)
            new_module.load_state_dict(module.state_dict())
            model._modules[name] = new_module
            model._modules[name].flash_forward = module.forward
            
    return model



def reset_topk(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = reset_usa(module)

        if isinstance(module, LlamaAttentionTopk):
            module._reset_state()

    return model
