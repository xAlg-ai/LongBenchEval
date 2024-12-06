import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import gc

from transformers.cache_utils import DynamicCache
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

def pseudo_quantize(tensor, q_bit):
    max_quant = 2 ** q_bit - 1

    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    scale = max_quant / range_val
    quantized = torch.round((tensor - min_val) * scale).clamp(0, max_quant)

    dequantized = quantized / scale + min_val

    return dequantized

class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # channel config
        self.token_budget = 128
        self.init_budget = 128
        self.recent_budget = 128
        self.chunk_size = 16
        self.label_bits = 16

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # stats
        self.collect_stats = False
        self.overlaps = {}
        self.recalls = {}
        self.precision = {}
        self.print_offloading_flag = False
        self.offloading_length = 25000


    def __repr__(self):
        return f"{super().__repr__()}\n Quest Sparsification Setting(topk:{self.token_budget}, pagesize:{self.chunk_size},label_bits:{self.label_bits},edge:{self.init_budget, self.recent_budget},stats:{self.collect_stats})"
        
    def compute_stats(self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ):
        ''' independent computation from forward pass to enable logging even when we do full attention
             Expects that the KV Cache is already on the GPU and that handling is outside the function
        '''

        raise NotImplementedError
        if (past_key_value is None or 
            len(past_key_value.key_cache) <= self.layer_idx or
            past_key_value.key_cache[self.layer_idx].shape[-2]  % 1024 != 0):
            return 

        # prepare keys and queries.
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
           
            
        if position_embeddings is None:
            print(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(query_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = past_key_value.key_cache[self.layer_idx] # keys already appended in cache
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        kv_seq_len = key_states.shape[2]


        # target
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 
        # causal_mask + recent budget + init budget
        causal_heavy_recent_mask = torch.tril(torch.ones(q_len,kv_seq_len,device=attn_weights.device), diagonal=kv_seq_len-q_len-self.local_const).bool()
        causal_heavy_recent_mask[:,:self.init_const] = False
        
        attn_weights.masked_fill_(torch.logical_not(causal_heavy_recent_mask), torch.finfo(attn_weights.dtype).min)
        target = torch.zeros_like(attn_weights)
        _,idx = torch.topk(attn_weights, dim=-1, k=32) # [B,A,S,T]
        view_idx = idx.view(-1,idx.shape[-1])
        view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * attn_weights.shape[-1]
        target.view(-1)[view_idx.view(-1)] = 1.0


        # predicted # dont want it to be conditioned on retrieval algorithm .. since I just want to measure the quality
        assert self.head_dim % self.group_factor == 0
        assert self.sorted_channel is not None
        sorted_query_states = query_states.transpose(1,2)
        sorted_key_states = key_states.transpose(1,2)
        sorted_query_states = torch.gather(sorted_query_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, q_len, -1, -1)).transpose(1,2)
        sorted_key_states = torch.gather(sorted_key_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, kv_seq_len, -1, -1)).transpose(1,2)
        # outlier channel only
        outlier_num = self.head_dim // self.group_factor
        grouped_query = sorted_query_states[:,:,:,:outlier_num]
        grouped_key = sorted_key_states[:,:,:,:outlier_num]
        # quantization
        if self.label_bits < 16:
            grouped_query = pseudo_quantize(grouped_query, self.label_bits)
            grouped_key = pseudo_quantize(grouped_key, self.label_bits)
        grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // self.group_factor)
        span = grouped_attn_weights        
        span.masked_fill_(torch.logical_not(causal_heavy_recent_mask),torch.finfo(span.dtype).min)
        pred = torch.zeros_like(span)
        _,idx = torch.topk(span, dim=-1, k=kv_seq_len // 16) # [B,A,S,T]
        view_idx = idx.view(-1,idx.shape[-1])
        view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * span.shape[-1]
        pred.view(-1)[view_idx.view(-1)] = 1.0

        # stats.
        overlap = pred * target
        overlap_ratio = torch.sum(overlap, dim=-1) / torch.sum(target, dim=-1)
        
        ## add to collection
        if kv_seq_len not in self.overlaps.keys():
            self.overlaps[kv_seq_len] = [0,0,0,0,0] # sum, sqsum, ct, mean, std
            
        self.overlaps[kv_seq_len][0] += overlap_ratio.sum().item()
        self.overlaps[kv_seq_len][1] += torch.square(overlap_ratio).sum().item()
        self.overlaps[kv_seq_len][2] += overlap_ratio.numel()
        self.overlaps[kv_seq_len][3] = self.overlaps[kv_seq_len][0] / self.overlaps[kv_seq_len][2]
        self.overlaps[kv_seq_len][4] = sqrt(self.overlaps[kv_seq_len][1] / self.overlaps[kv_seq_len][2] - self.overlaps[kv_seq_len][3]**2)

        if self.layer_idx == 17:
            print(self.overlaps)        

        
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
            if self.print_offloading_flag == False:
                print("OFFLOADING ENABLED >>")
                self.print_offloading_flag = True
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].cpu()
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].cpu()


    # untouched from the original code
    def local_heavy_hitter_mask(self, attn_weights, token_budget, chunk_size):
        # attn_weights (BS, head, query, keys)
    
        # expend attn_weights to be divisible by chunk_size
        seq_length = attn_weights.shape[-1]
        padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
        attn_weights = torch.cat(
            [
                attn_weights,
                torch.ones(
                    (
                        attn_weights.shape[0],
                        attn_weights.shape[1],
                        attn_weights.shape[2],
                        padding_length,
                    ),
                    device=attn_weights.device,
                )
                * torch.tensor(torch.finfo(attn_weights.dtype).min),
            ],
            dim=-1,
        )
    
        # chunk attn_weights into chunk_size tokens
        chunk_attn_weights = attn_weights.reshape(
            attn_weights.shape[0],
            attn_weights.shape[1],
            attn_weights.shape[2],
            attn_weights.shape[3] // chunk_size,
            chunk_size,
        ).amax(dim=-1)
    
        _, topk = chunk_attn_weights.topk(
            k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
        )
        # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
        topk = topk.unsqueeze(-1).repeat(
            1, 1, 1, 1, chunk_size
        ) * chunk_size + torch.arange(chunk_size, device=topk.device)
        topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom.scatter_(-1, topk, True)
    
        # remove the padding
        mask_bottom = mask_bottom[:, :, :, :seq_length]
    
        return mask_bottom

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # if self.config.num_hidden_layers != 32:
        #     gc.collect()
        #     torch.cuda.empty_cache()
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
            if self.collect_stats:
                self.compute_stats(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    position_embeddings
                    )
            self.offload_if_necessary_cpu(past_key_value)
            return return_value

        #function is the same from original code starting here with added edge budget and 
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        
        # New cache format
        if isinstance(past_key_value, DynamicCache):
            kv_seq_len = past_key_value.get_seq_length()
        # Legacy cache format
        else:
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                assert isinstance(past_key_value, tuple)
                kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, nh, t, hd]
        # New cache format
        if isinstance(past_key_value, DynamicCache):
            if use_cache:
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
        # Legacy cache format
        else:
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None
        
        kv_seq_len= key_states.shape[2]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        sign = (query_states > 0) + (~(query_states > 0)) * -1
        max_key = key_states * sign
        postive_query = query_states * sign
    
        # expend max_key to be divisible by chunk_size
        seq_length = max_key.shape[-2]
        padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
        max_key = torch.cat(
            [
                max_key,
                torch.ones(
                    (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                    device=max_key.device,
                )
                * torch.tensor(torch.finfo(max_key.dtype).min),
            ],
            dim=-2,
        )
    
        # chunk max_key into chunk_size tokens
        chunk_max_key = max_key.reshape(
            max_key.shape[0],
            max_key.shape[1],
            max_key.shape[2] // self.chunk_size,
            self.chunk_size,
            max_key.shape[3],
        ).amax(dim=-2)
    
        # duplicate chunk_max_key chunk_size times
        chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
        # reshape chunk_max_key to the original shape
        chunk_max_key = chunk_max_key.reshape(
            chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
        )[:, :, :seq_length, :]
    


        if self.label_bits < 16:
            chunk_max_key = pseudo_quantize(chunk_max_key, self.label_bits)
            postive_query = pseudo_quantize(postive_query, self.label_bits)

        quantized_weight = torch.matmul(
            postive_query.float(),
            chunk_max_key.transpose(2, 3),
        )
    
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
    
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            quantized_weight = quantized_weight + attention_mask
            quantized_weight = torch.max(
                quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
            )
    
        token_budget = min(kv_seq_len, self.token_budget)
    
        attn_weights_for_selection = quantized_weight
        # remove edge from topk selection
        attn_weights_for_selection[:,:,:,:self.init_budget] = torch.finfo(quantized_weight.dtype).min
        attn_weights_for_selection[:,:,:,-self.recent_budget:] = torch.finfo(quantized_weight.dtype).min
    
        if token_budget > 0:
            mask_bottom = self.local_heavy_hitter_mask(
                attn_weights_for_selection, token_budget, self.chunk_size
            )  # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)
    
        mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
        mask_bottom[:,:,:,:self.init_budget] = True
        mask_bottom[:,:,:,-self.recent_budget:] = True
        attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
    
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
    
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    
        attn_output = self.o_proj(attn_output)
    
        if not output_attentions:
            attn_weights = None
    
        return attn_output, attn_weights, past_key_value



def convert_quest(model, config, collect_stats):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_quest(module, config, collect_stats)

        if isinstance(module, LlamaAttention):
            device = next(module.parameters()).device
            new_module = LlamaAttention_heavy_hitter(config, module.layer_idx).bfloat16().to(device)
            new_module.load_state_dict(module.state_dict())
            new_module.token_budget = config.token_budget
            new_module.chunk_size = config.chunk_size
            new_module.init_budget = config.init_budget
            new_module.recent_budget = config.recent_budget
            new_module.label_bits = config.label_bits
            new_module.collect_stats = collect_stats
            model._modules[name] = new_module
            model._modules[name].flash_forward = module.forward

    return model
