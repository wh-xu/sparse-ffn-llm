import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    repeat_kv,
    apply_rotary_pos_emb,
)

from transformers.cache_utils import Cache


def linear_topk(x, linear, keep_ratio):
    # bsz, seq, dim
    out = linear(x)

    # keep top% activation
    if keep_ratio < 1.0:
        num_out = out.shape[-1]
        val, ind = torch.topk(torch.abs(out), k=int(num_out * keep_ratio), largest=True)
        res = torch.zeros_like(out)
        res.scatter_(-1, ind, val)
        res = res.view(*out.size())
        out = res * out.sign()
    return out


def svd_mask(x, gate_weight_Q_svd, gate_weight_R_svd, density_threshold):
    pred = x @ (gate_weight_Q_svd @ gate_weight_R_svd).T

    mask = pred.abs() > density_threshold
    return mask


# Redefine gated FFN layer to enable sampling
def mlp_gate3_sample(self, x):
    self.list_inp.append(x.detach().clone().bfloat16().cpu())

    out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    self.list_out.append(out.detach().clone().bfloat16().cpu())
    return out


# Redefine gated FFN layer to enable sampling
def wkv_sample(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    self.list_inp.append(hidden_states.detach().clone().cpu())

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and cache_position is not None:
        causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# Redefine gated FFN layer with Topk activation
def mlp_gate3_topk(self, x):
    # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    out = linear_topk(x, self.gate_proj, self.keep_ratio)

    # out = self.gate_proj(x)  # bsz, seq, dim

    # # keep top% activation
    # if self.keep_ratio < 1.0:
    #     num_out = out.shape[-1]
    #     val, ind = torch.topk(
    #         torch.abs(out), k=int(num_out * self.keep_ratio), largest=True
    #     )
    #     res = torch.zeros_like(out)
    #     res.scatter_(-1, ind, val)
    #     res = res.view(*out.size())
    #     out = res * out.sign()

    # sparsity_ratio = (out == 0).float().mean()
    # print(sparsity_ratio)

    out = self.down_proj(self.act_fn(out) * self.up_proj(x))
    return out


# MLP layer with gated activation function
def mlp_gate3_svd(self, x):
    # pred = x @ (self.gate_weight_Q_svd @ self.gate_weight_R_svd).T
    # mask = pred.abs() > self.density_threshold[self.keep_ratio]

    mask = svd_mask(
        x,
        self.gate_weight_Q_svd,
        self.gate_weight_R_svd,
        self.density_threshold[self.keep_ratio],
    )

    out = mask * self.act_fn(self.gate_proj(x))  # bsz, seq, dim

    out = self.down_proj(out * self.up_proj(x))
    return out


def mlp_gate3_svd_topk(self, x):
    pred = x @ (self.gate_weight_Q_svd @ self.gate_weight_R_svd).T

    out = self.act_fn(self.gate_proj(x))  # bsz, seq, dim

    # keep top% activation
    if self.keep_ratio < 1.0:
        num_out = out.shape[-1]
        _, ind = torch.topk(
            torch.abs(pred), k=int(num_out * self.keep_ratio), largest=True
        )
        res = torch.zeros_like(out)
        val = torch.gather(out.abs(), -1, ind)
        res.scatter_(-1, ind, val)
        res = res.view(*out.size())
        out = res * out.sign()

    out = self.down_proj(out * self.up_proj(x))
    return out


def qkvo_topk(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # bsz, q_len, _ = hidden_states.size()

    # query_states = (
    #     self.q_proj(hidden_states)
    #     .view(bsz, q_len, self.num_heads, self.head_dim)
    #     .transpose(1, 2)
    # )
    # key_states = (
    #     self.k_proj(hidden_states)
    #     .view(bsz, q_len, self.num_heads, self.head_dim)
    #     .transpose(1, 2)
    # )
    # value_states = (
    #     self.v_proj(hidden_states)
    #     .view(bsz, q_len, self.num_heads, self.head_dim)
    #     .transpose(1, 2)
    # )

    # kv_seq_len = key_states.shape[-2]
    # print(past_key_value)
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value[0].shape[-2]

    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(
    #     query_states, key_states, cos, sin, position_ids
    # )
    # # [bsz, nh, t, hd]

    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # past_key_value = (key_states, value_states) if use_cache else None

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
    #     self.head_dim
    # ) # bsz, num_heads, q_len, kv_len

    # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #         f" {attn_weights.size()}"
    #     )

    # if attention_mask is not None:
    #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights + attention_mask
    #     attn_weights = torch.max(
    #         attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
    #     )

    # ### Heavy + Recent
    # # self.keep_ratio = 0.2
    # self.heavy_budget_ratio = 0.5
    # self.recent_budget_ratio = 0.5
    # heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
    # recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])

    # # Heavy Hitter Mask (Based on global statistics)
    # tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
    #     attn_weights.dtype
    # ) # bsz, num_heads, q_len, kv_len
    # tmp_sum = torch.sum(tmp_attn, dim=-2) # bsz, num_heads, kv_len
    # _, tmp_topk = tmp_sum.topk(k=heavy_budget, dim=-1) # bsz, num_heads, topk

    # zeros = torch.zeros_like(tmp_sum, dtype=torch.bool) # bsz, num_heads, kv_len
    # mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(2) # bsz, num_heads, kv_len
    # mask_bottom = mask_bottom.expand(
    #     mask_bottom.shape[0],
    #     mask_bottom.shape[1],
    #     attn_weights.shape[-2],
    #     mask_bottom.shape[-1],
    # ) # bsz, num_heads, q_len, kv_len

    # ones = torch.ones_like(attn_weights, dtype=torch.bool)
    # ones = torch.tril(ones, diagonal=recent_budget)
    # ones = torch.triu(ones, diagonal=-recent_budget)
    # mask_bottom = torch.logical_or(mask_bottom, ones)
    # # mask_bottom = ones
    # attn_weights[~mask_bottom] = torch.finfo(attn_weights.dtype).min

    # # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
    #     query_states.dtype
    # )
    # attn_output = torch.matmul(attn_weights, value_states)

    # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )

    # attn_output = attn_output.transpose(1, 2)
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    # attn_output = self.o_proj(attn_output)

    # if not output_attentions:
    #     attn_weights = None

    # return attn_output, attn_weights, past_key_value

    bsz, q_len, _ = hidden_states.size()

    # query_states = linear_topk(hidden_states, self.q_proj, self.keep_ratio)
    # key_states = linear_topk(hidden_states, self.k_proj, self.keep_ratio)
    # value_states = linear_topk(hidden_states, self.v_proj, self.keep_ratio)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and cache_position is not None:
        causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    #     query_states,
    #     key_states,
    #     value_states,
    #     attn_mask=causal_mask,
    #     dropout_p=self.attention_dropout if self.training else 0.0,
    # )

    # ############################################################
    # bsz, n_head, seq, head_dim
    attn_weight = (
        query_states @ key_states.transpose(-2, -1) / math.sqrt(query_states.size(-1))
    )
    attn_weight += causal_mask
    # attn_weight = torch.softmax(attn_weight, dim=-1)
    # bsz, n_head, q_seq, kv_seq

    ### Heavy + Recent
    # self.keep_ratio = 0.2
    self.heavy_budget_ratio = 0.1
    self.recent_budget_ratio = 0.1
    heavy_budget = int(self.heavy_budget_ratio * attn_weight.shape[-1])
    recent_budget = int(self.recent_budget_ratio * attn_weight.shape[-1])

    # Heavy Hitter Mask (Based on global statistics)
    tmp_attn = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
        attn_weight.dtype
    )  # bsz, num_heads, q_len, kv_len
    tmp_sum = torch.sum(tmp_attn, dim=-2)  # bsz, num_heads, kv_len
    _, tmp_topk = tmp_sum.topk(k=heavy_budget, dim=-1)  # bsz, num_heads, topk

    zeros = torch.zeros_like(tmp_sum, dtype=torch.bool)  # bsz, num_heads, kv_len
    mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(
        2
    )  # bsz, num_heads, kv_len
    mask_bottom = mask_bottom.expand(
        mask_bottom.shape[0],
        mask_bottom.shape[1],
        attn_weight.shape[-2],
        mask_bottom.shape[-1],
    )  # bsz, num_heads, q_len, kv_len

    ones = torch.ones_like(attn_weight, dtype=torch.bool)
    ones = torch.tril(ones, diagonal=recent_budget)
    ones = torch.triu(ones, diagonal=-recent_budget)
    mask_bottom = torch.logical_or(mask_bottom, ones)
    # mask_bottom = ones
    attn_weight[~mask_bottom] = torch.finfo(attn_weight.dtype).min

    sparsity = (attn_weight < -1e6).float().mean()
    print(f"attn sparsity = {sparsity:.3f}")

    # upcast attention to fp32
    attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weight, value_states)

    sparsity = (attn_output == 0).float().mean()
    print(f"attnout sparsity = {sparsity:.3f}")

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

    # attn_output = attn_weight @ value_states
    # # bsz, n_head, kv_seq, head_dim -> bsz, n_head, q_seq, head_dim

    # tmp = attn_output.norm(dim=-1)  # bsz, heads, seq
    # _, keep_mask = torch.topk(
    #     tmp, dim=1, k=int(self.num_key_value_heads * self.keep_ratio)
    # )  # bsz, topk, seq
    # attn_mask = torch.zeros(
    #     bsz,
    #     self.num_key_value_heads,
    #     q_len,
    #     device=attn_output.device,
    #     dtype=attn_output.dtype,
    # ).scatter_(1, keep_mask, 1).unsqueeze(-1)

    # # attn_output = (attn_weight * attn_mask) @ value_states
    # attn_output = (attn_weight @ value_states) * attn_mask
    # # attn_output = (attn_weight @ value_states)
    # # bsz, n_head, seq, head_dim
    # ############################################################

    # attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    # attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def qkvo_svd(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    ############################################################
    # mask = svd_mask(
    #     hidden_states,
    #     self.q_weight_Q_svd,
    #     self.q_weight_R_svd,
    #     self.q_density_threshold[self.keep_ratio],
    # )
    # query_states = self.q_proj(hidden_states) * mask

    # mask = svd_mask(
    #     hidden_states,
    #     self.k_weight_Q_svd,
    #     self.k_weight_R_svd,
    #     self.k_density_threshold[self.keep_ratio],
    # )
    # key_states = self.k_proj(hidden_states) * mask

    # mask = svd_mask(
    #     hidden_states,
    #     self.v_weight_Q_svd,
    #     self.v_weight_R_svd,
    #     self.v_density_threshold[self.keep_ratio],
    # )
    # value_states = self.v_proj(hidden_states)
    ############################################################

    ############################################################
    # Softmax Approximation
    ############################################################
    # query_appr = hidden_states @ (self.q_weight_Q_svd @ self.q_weight_R_svd).T
    # key_appr = hidden_states @ (self.k_weight_Q_svd @ self.k_weight_R_svd).T
    # value_appr = hidden_states @ (self.v_weight_Q_svd @ self.v_weight_R_svd).T

    # query_appr = query_appr.view(bsz, q_len, self.num_heads, self.head_dim).transpose(
    #     1, 2
    # )
    # key_appr = key_appr.view(
    #     bsz, q_len, self.num_key_value_heads, self.head_dim
    # ).transpose(1, 2)
    # value_appr = value_appr.view(
    #     bsz, q_len, self.num_key_value_heads, self.head_dim
    # ).transpose(1, 2)

    # cos, sin = self.rotary_emb(value_appr, position_ids)
    # query_appr, key_appr = apply_rotary_pos_emb(query_appr, key_appr, cos, sin)

    # past_key_value = getattr(self, "past_key_value", past_key_value)
    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; position_ids needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_appr, value_appr = past_key_value.update(
    #         key_appr, value_appr, self.layer_idx, cache_kwargs
    #     )

    # key_appr = repeat_kv(key_appr, self.num_key_value_groups)
    # value_appr = repeat_kv(value_appr, self.num_key_value_groups)

    # causal_mask = attention_mask
    # if attention_mask is not None and cache_position is not None:
    #     causal_mask = causal_mask[:, :, cache_position, : key_appr.shape[-2]]

    # if query_appr.device.type == "cuda" and causal_mask is not None:
    #     query_appr = query_appr.contiguous()
    #     key_appr = key_appr.contiguous()
    #     value_appr = value_appr.contiguous()

    # scale_factor = 1 / math.sqrt(query_appr.size(-1))
    # attn_weight = query_appr @ key_appr.transpose(-2, -1) * scale_factor
    # attn_weight += causal_mask
    # attn_weight = torch.softmax(attn_weight, dim=-1)  # bsz, seq, n_head, n_head

    # attn_output = attn_weight @ value_appr # bsz, n_head, q_seq, head_dim
    # tmp = attn_output.norm(dim=-1)  # bsz, heads, seq

    # _, keep_mask = torch.topk(
    #     tmp, dim=1, k=int(self.num_key_value_heads * self.keep_ratio)
    # )  # bsz, topk, seq
    # attn_mask = torch.zeros(
    #     bsz,
    #     self.num_key_value_heads,
    #     q_len,
    #     device=tmp.device,
    #     dtype=tmp.dtype,
    # ).scatter_(1, keep_mask, 1).unsqueeze(-1)

    # attn_output = attn_weight @ value_states
    ############################################################

    ############################################################
    # L, S = query_appr.size(-2), key_appr.size(-2)
    # scale_factor = 1 / math.sqrt(query_appr.size(-1))
    # attn_bias = torch.zeros(L, S, dtype=query_appr.dtype)

    # if is_causal:
    #     assert attn_mask is None
    #     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    #     attn_bias.to(query.dtype)

    # if attn_mask is not None:
    #     if attn_mask.dtype == torch.bool:
    #         attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    #     else:
    #         attn_bias += attn_mask

    # attn_weight = query_appr @ key_appr.transpose(-2, -1) * scale_factor
    # attn_weight += attn_bias
    # attn_weight = torch.softmax(attn_weight, dim=-1) # bsz, seq, n_head, n_head
    # attn_output = attn_weight @ value_states
    ############################################################

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and cache_position is not None:
        causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    ############################################################
    scale_factor = 1 / math.sqrt(query_states.size(-1))
    attn_weight = query_states @ key_states.transpose(-2, -1) * scale_factor
    attn_weight += causal_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)  # bsz, n_head, q_seq, kv_seq

    # tmp = attn_weight.norm(dim=-1)  # bsz, heads, seq
    # tmp = attn_weight.norm(dim=1)  # bsz, q_seq, kv_seq

    attn_output = attn_weight @ value_states  # bsz, n_head, q_seq, head_dim
    tmp = attn_output.norm(dim=-1)  # bsz, heads, seq
    _, keep_mask = torch.topk(
        tmp, dim=1, k=int(self.num_key_value_heads * self.keep_ratio)
    )  # bsz, topk, seq
    attn_mask = (
        torch.zeros(
            bsz,
            self.num_key_value_heads,
            q_len,
            device=tmp.device,
            dtype=tmp.dtype,
        )
        .scatter_(1, keep_mask, 1)
        .unsqueeze(-1)
    )

    # attn_output = (attn_weight * attn_mask) @ value_states
    attn_output = (attn_weight @ value_states) * attn_mask
    ############################################################

    ############################################################
    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    #     query_states,
    #     key_states,
    #     value_states,
    #     attn_mask=causal_mask,
    #     dropout_p=self.attention_dropout if self.training else 0.0,
    # )
    ############################################################

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
