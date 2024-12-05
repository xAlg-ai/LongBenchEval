import os
import pdb
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


from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
logger = logging.get_logger(__name__)

__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']


#class LlamaAttention_heavy_hitter(nn.Module):
#    """Multi-headed attention from 'Attention Is All You Need' paper"""
#
#    def __init__(self, config: LlamaConfig):
#        super().__init__()
#        self.config = config
#        self.hidden_size = config.hidden_size
#        self.num_heads = config.num_attention_heads
#        self.head_dim = self.hidden_size // self.num_heads
#        self.max_position_embeddings = config.max_position_embeddings
#
#        if (self.head_dim * self.num_heads) != self.hidden_size:
#            raise ValueError(
#                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                f" and `num_heads`: {self.num_heads})."
#            )
#        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
#        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
#        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
#        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
#        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
#
#        self.heavy_budget_ratio = config.heavy_ratio
#        self.recent_budget_ratio = config.recent_ratio
#        self.attention_masks_next = None 
#        self.heavy_budget = None
#        self.recent_budget = None
#        self.cache_budget = None
#        self.previous_scores = None
#        self.input_length = []
#        self.cache_budget_records = []
#
#    def _reset_masks(self):
#        self.attention_masks_next = None 
#        self.heavy_budget = None
#        self.recent_budget = None
#        self.cache_budget = None
#        self.previous_scores = None
#
#    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
#
#    def forward(
#        self,
#        hidden_states: torch.Tensor,
#        attention_mask: Optional[torch.Tensor] = None,
#        position_ids: Optional[torch.LongTensor] = None,
#        past_key_value: Optional[Tuple[torch.Tensor]] = None,
#        output_attentions: bool = False,
#        use_cache: bool = False,
#    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#        bsz, q_len, _ = hidden_states.size()
#
#        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#
#        kv_seq_len = key_states.shape[-2]
#        if past_key_value is not None:
#            kv_seq_len += past_key_value[0].shape[-2]
#        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#        # [bsz, nh, t, hd]
#
#        if past_key_value is not None:
#            # reuse k, v, self_attention
#            key_states = torch.cat([past_key_value[0], key_states], dim=2)
#            value_states = torch.cat([past_key_value[1], value_states], dim=2)
#
#        past_key_value = (key_states, value_states) if use_cache else None
#
#        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
#
#        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#            raise ValueError(
#                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                f" {attn_weights.size()}"
#            )
#
#        if attention_mask is not None:
#            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                raise ValueError(
#                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#                )
#            attn_weights = attn_weights + attention_mask
#            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
#
#
#        if self.attention_masks_next is not None:
#            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min
#
#        # upcast attention to fp32
#        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#
#
#        # attn_weights (BS, heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
#        current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)
#        # offset = attn_weights.gt(0).sum(0).sum(1)
#
#        # Accumulate attention scores
#        if not self.previous_scores == None:
#            current_scores_sum[:, :-1] += self.previous_scores #(Enlarged Sequence)
#        else:
#            self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
#            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
#            self.cache_budget = self.heavy_budget + self.recent_budget
#            self.cache_budget_records.append(self.cache_budget)
#            self.input_length.append(attn_weights.shape[-1])
#
#            # current_scores_sum = current_scores_sum / offset
#        dtype_attn_weights = attn_weights.dtype
#        attn_weights_devices = attn_weights.device
#        assert attn_weights.shape[0] == 1
#        self.previous_scores = current_scores_sum #(heads, k-tokens)
#        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)
#
#        attn_tokens_all = self.previous_scores.shape[-1]
#    
#        if attn_tokens_all > self.cache_budget:
#            # activate most recent k-cache
#            if not self.recent_budget == 0:
#                attn_mask[:, :-self.recent_budget] = 0
#                selected_set = self.previous_scores[:, :-self.recent_budget]
#            else:
#                # activate historical best self.cache_budget - self.recent_budget tokens.
#                # self.previous_scores # (k-Cache - 1)
#                selected_set = self.previous_scores
#
#            if not self.heavy_budget == 0:
#                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
#                attn_mask = attn_mask.scatter(-1, keep_topk, 1)
#
#        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)
#
#        score_mask = attn_mask[:,:-1]
#        score_mask[:, -self.recent_budget:] = 1
#        self.previous_scores = self.previous_scores * score_mask
#
#        attn_output = torch.matmul(attn_weights, value_states)
#
#        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#            raise ValueError(
#                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                f" {attn_output.size()}"
#            )
#
#        attn_output = attn_output.transpose(1, 2)
#        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
#
#        attn_output = self.o_proj(attn_output)
#
#        if not output_attentions:
#            attn_weights = None
#
#        return attn_output, attn_weights, past_key_value



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaAttentionH2O(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    Previous version of H2O wsa not coded up for GQA . So updating the implementation to latest
    """

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


        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

    def _reset_masks(self):
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

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

        q = query_states.shape[2]

        if self.layer_idx == 0:
            print("keys, queries" , key_states.shape[2], q, self.attention_masks_next.shape if (self.attention_masks_next is not None) else "-", flush=True)
        if q != 1: # not in decoding yet
            self._reset_masks()

        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # attn_weights (BS, heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
        current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)
        # offset = attn_weights.gt(0).sum(0).sum(1)

        # Accumulate attention scores
        if not self.previous_scores == None:
            current_scores_sum[:, :-1] += self.previous_scores #(Enlarged Sequence)
        else:
            self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = self.heavy_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])

            # current_scores_sum = current_scores_sum / offset
        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1
        self.previous_scores = current_scores_sum #(heads, k-tokens)
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]
    
        if attn_tokens_all > self.cache_budget:
            # activate most recent k-cache
            if not self.recent_budget == 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                # activate historical best self.cache_budget - self.recent_budget tokens.
                # self.previous_scores # (k-Cache - 1)
                selected_set = self.previous_scores

            if not self.heavy_budget == 0:
                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)

        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

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

        self.offload_if_necessary_cpu(past_key_value)
        return attn_output, attn_weights, past_key_value


def convert_h2o(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_h2o(module, config)

        if isinstance(module, LlamaAttention):
            device = next(module.parameters()).device
            new_module = LlamaAttentionH2O(config, module.layer_idx).to(torch.bfloat16).to(device)
            new_module.load_state_dict(module.state_dict())
            model._modules[name] = new_module

    return model

def reset_h2o(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = reset_h2o(module)

        if isinstance(module, LlamaAttention_heavy_hitter):
            module._reset_masks()

    return model

