from . import layers
from .common_ops import get_layers, svd_reconstruct_with_rank

import os, math
import types
from typing import List, Optional, Tuple, Union

from . import utils

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention,
    LlamaSdpaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    repeat_kv,
    apply_rotary_pos_emb,
)

# from .modeling_sparsellama import SparseLlamaMLP

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)

from transformers.cache_utils import Cache, DynamicCache, StaticCache


def llama_mlp_forward_with_svd_pred(self, x):
    act = self.act_fn(self.gate_proj(x))
    gt_pos_idx = (act > 0).type(act.dtype)

    pred = x @ self.svd_gate.T
    pred_pos_idx = (pred > self.thres).type(act.dtype)

    act = act * pred_pos_idx
    down_proj = self.down_proj(act * self.up_proj(x))

    # Save history data
    sparsity_pred = 1 - float(pred_pos_idx.view(-1).mean())
    sparsity_gt = 1 - float(gt_pos_idx.view(-1).mean())
    self.history["sparsity"].append([sparsity_pred, sparsity_gt])

    recall = torch.sum(gt_pos_idx * pred_pos_idx) / torch.sum(gt_pos_idx)
    self.history["recall"].append(float(recall))

    return down_proj


def llama_attention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

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

    # Top-k
    # for t in [query_states, key_states, value_states]:
    #     ratio = 0.7
    #     topk = int((1 - ratio) * self.head_dim)
    #     t = t.view(-1)
    #     _, top_idx = torch.topk(t.abs(), k=topk, largest=False)
    #     t = t.scatter_(-1, top_idx, 0).view(bsz, q_len, self.num_heads, self.head_dim)
    ###########################################

    # Top-k
    ################
    # def filter_small_values(inp, keep_ratio):
    #     # print(inp.shape)
    #     # bsz, n_head, n_seq, n_head_dim
    #     if inp.shape[2]>0:
    #         inp = inp.reshape(-1)
    #         topk = int((1 - keep_ratio) * inp.numel())
    #         _, filter_idx = torch.topk(inp.abs(), k=topk, largest=False)
    #         inp = inp.scatter_(-1, filter_idx, 0)
    #     return inp

    # keep_ratio = 0.7
    # query_states = filter_small_values(query_states, keep_ratio).reshape(bsz, self.num_heads, q_len, self.head_dim)
    # key_states = filter_small_values(key_states, keep_ratio).reshape(bsz, self.num_heads, q_len, self.head_dim)
    # value_states = filter_small_values(value_states, keep_ratio).reshape(bsz, self.num_heads, q_len, self.head_dim)
    ###############

    past_key_value = getattr(self, "past_key_value", past_key_value)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask
        if cache_position is not None:
            causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_weights = attn_output.detach().clone()

    ####
    keep_ratio = 0.5
    n_head = self.num_heads
    # print(attn_weights.shape)

    # tmp = attn_weights.norm(dim=-1).transpose(1,2) # bsz, q_len, heads
    # print(tmp)
    # _, keep_mask = torch.topk(
    # tmp, dim=-1, k=int(n_head * keep_ratio)
    # )  # bsz, topk, q_len
    # print(keep_mask)
    # print(keep_mask.shape)
    # layer_head_mask = torch.zeros(
    # n_head, q_len, device=attn_output.device, dtype=attn_output.dtype
    # ).scatter_(0, keep_mask, 1)
    # print(layer_head_mask)
    # print(layer_head_mask.shape)

    # if layer_head_mask is not None:
    # attn_output = attn_output * layer_head_mask.view(bsz, self.num_heads, q_len, 1)
    ####

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    # if not output_attentions:
    # attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        # layer_head_mask=layer_head_mask,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def llama_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # print(hidden_states.shape)
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if isinstance(next_decoder_cache, Cache)
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


count = 0
cached_svd_gate_list = []


def enable_llama_mlp_forward_with_svd_pred(
    model, cache_svd_file_prefix=None, rank=256, thres=-0.75
):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_mlp_forward_with_svd_pred(
                module, cache_svd_file_prefix, rank, thres
            )

        if isinstance(module, LlamaModel):
            model._modules[name].forward = types.MethodType(
                llama_model, model._modules[name]
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_attention, model._modules[name]
            )

        if isinstance(module, LlamaMLP):
            global count
            cache_fname = f"{cache_svd_file_prefix}_rank_{rank}_{count}.pt"
            if os.path.exists(cache_fname):
                print(f"SVD cache exists. Loading..")
                svd_gate = torch.load(cache_fname)
            else:
                print(f"SVD cache not exists. Caching..")
                svd_gate = svd_reconstruct_with_rank(
                    inp=module.gate_proj.weight, rank=rank
                ).to("cuda")
                torch.save(svd_gate, cache_fname)

            model._modules[name].thres = thres
            model._modules[name].svd_gate = svd_gate
            model._modules[name].history = {"sparsity": [], "recall": []}

            model._modules[name].forward = types.MethodType(
                llama_mlp_forward_with_svd_pred, model._modules[name]
            )

            count += 1


layer = 0
sparsity_list = []
recall_list = []


def get_llama_mlp_sparsity(model):
    for i, (name, module) in enumerate(reversed(model._modules.items())):
        if len(list(module.children())) > 0:
            get_llama_mlp_sparsity(module)

        if isinstance(module, LlamaMLP):
            sparsity_list.extend(module.history["sparsity"])
            recall_list.extend(module.history["recall"])

            m, v = utils.calc_mean_var(module.history["sparsity"])
            m_r, v_r = utils.calc_mean_var(module.history["recall"])
            global layer
            layer += 1
            print(
                f"@ Layer {layer}\t pred/gt sparsity {m[0]:.3f}+/-{v[0]:.3f} and {m[1]:.3f}+/-{v[1]:.3f}\trecall {m_r:.4f} +/- {v_r:.4f}"
            )


def llama_mlp_forward_(self, x):
    act = self.act_fn(self.gate_proj(x))

    # bsz, seq, dim_inter = act.shape
    # pos_freq = (act > 0).float().sum(dim=0).sum(dim=0).cpu()

    # if self.mlp_act_freq is None:
    #     self.mlp_act_freq = pos_freq
    # else:
    #     self.mlp_act_freq += pos_freq

    # self.mlp_act_total += seq * bsz

    act_trace = (act.reshape(-1, act.shape[-1]) > 0).float()
    seq, _ = act_trace.shape
    pos_freq = act_trace.sum(dim=0).cpu()

    sparsity = (1 - act_trace.mean(dim=-1).cpu()).flatten().tolist()

    self.mlp_act_total += seq
    if self.mlp_act_freq is None:
        self.mlp_act_freq = pos_freq
        self.mlp_act_sparsity = sparsity
    else:
        self.mlp_act_freq += pos_freq
        self.mlp_act_sparsity.extend(sparsity)

    down_proj = self.down_proj(act * self.up_proj(x))

    return down_proj


# def llama_decoder_layer(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: Optional[bool] = False,
#     use_cache: Optional[bool] = False,
#     cache_position: Optional[torch.LongTensor] = None,
#     **kwargs,
# ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
#     residual = hidden_states

#     hidden_states = self.input_layernorm(hidden_states)

#     # Self Attention
#     hidden_states, self_attn_weights, present_key_value = self.self_attn(
#         hidden_states=hidden_states,
#         attention_mask=attention_mask,
#         position_ids=position_ids,
#         past_key_value=past_key_value,
#         output_attentions=output_attentions,
#         use_cache=use_cache,
#         cache_position=cache_position,
#         # layer_head_mask=layer_head_mask,
#         **kwargs,
#     )
#     hidden_states = residual + hidden_states

#     # Fully Connected
#     residual = hidden_states
#     hidden_states = self.post_attention_layernorm(hidden_states)
#     hidden_states = self.mlp(hidden_states)
#     hidden_states = residual + hidden_states

#     outputs = (hidden_states,)

#     if output_attentions:
#         outputs += (self_attn_weights,)

#     if use_cache:
#         outputs += (present_key_value,)

#     return outputs


# def llama_model(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     cache_position: Optional[torch.LongTensor] = None,
# ) -> Union[Tuple, BaseModelOutputWithPast]:
#     output_attentions = (
#         output_attentions
#         if output_attentions is not None
#         else self.config.output_attentions
#     )
#     output_hidden_states = (
#         output_hidden_states
#         if output_hidden_states is not None
#         else self.config.output_hidden_states
#     )
#     use_cache = use_cache if use_cache is not None else self.config.use_cache
#     return_dict = (
#         return_dict if return_dict is not None else self.config.use_return_dict
#     )

#     if (input_ids is None) ^ (inputs_embeds is not None):
#         raise ValueError(
#             "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
#         )

#     if inputs_embeds is None:
#         inputs_embeds = self.embed_tokens(input_ids)

#     past_seen_tokens = 0
#     if use_cache:  # kept for BC (cache positions)
#         if not isinstance(past_key_values, StaticCache):
#             past_key_values = DynamicCache.from_legacy_cache(past_key_values)
#         past_seen_tokens = past_key_values.get_seq_length()

#     if cache_position is None:
#         cache_position = torch.arange(
#             past_seen_tokens,
#             past_seen_tokens + inputs_embeds.shape[1],
#             device=inputs_embeds.device,
#         )

#     if position_ids is None:
#         position_ids = cache_position.unsqueeze(0)

#     causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

#     # embed positions
#     hidden_states = inputs_embeds

#     # decoder layers
#     all_hidden_states = () if output_hidden_states else None
#     all_self_attns = () if output_attentions else None
#     next_decoder_cache = None

#     # print(hidden_states.shape)
#     for decoder_layer in self.layers:
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         layer_outputs = decoder_layer(
#             hidden_states,
#             attention_mask=causal_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_values,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#         )

#         hidden_states = layer_outputs[0]

#         if use_cache:
#             next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#         if output_attentions:
#             all_self_attns += (layer_outputs[1],)

#     hidden_states = self.norm(hidden_states)

#     # add hidden states from the last decoder layer
#     if output_hidden_states:
#         all_hidden_states += (hidden_states,)

#     next_cache = None
#     if use_cache:
#         next_cache = (
#             next_decoder_cache.to_legacy_cache()
#             if isinstance(next_decoder_cache, Cache)
#             else next_decoder_cache
#         )
#     if not return_dict:
#         return tuple(
#             v
#             for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
#             if v is not None
#         )
#     return BaseModelOutputWithPast(
#         last_hidden_state=hidden_states,
#         past_key_values=next_cache,
#         hidden_states=all_hidden_states,
#         attentions=all_self_attns,
#     )


def dump_prosparse_llama_sparsity_trace(model):
    for name, module in reversed(model._modules.items()):
        # if isinstance(module, LlamaModel):
        #     model._modules[name].forward = types.MethodType(
        #         llama_model, model._modules[name]
        #     )

        # print(module, type(module), name)
        # if isinstance(module, SparseLlamaMLP):
        if name == "mlp":
            model._modules[name].forward = types.MethodType(
                llama_mlp_forward_, model._modules[name]
            )

            model._modules[name].mlp_act_total = 0
            model._modules[name].mlp_act_freq = None
            model._modules[name].mlp_act_sparsity = None

        if len(list(module.children())) > 0:
            dump_prosparse_llama_sparsity_trace(module)


mlp_act_total = []
mlp_act_freq = []
mlp_act_sparsity = []


def collect_mlp_act_trace(model):
    global mlp_act_total, mlp_act_freq, mlp_act_sparsity
    for i, (name, module) in enumerate(model._modules.items()):
        if name == "mlp":
            mlp_act_total.append(model._modules[name].mlp_act_total)
            mlp_act_freq.append(model._modules[name].mlp_act_freq)
            mlp_act_sparsity.append(model._modules[name].mlp_act_sparsity)

        if len(list(module.children())) > 0:
            collect_mlp_act_trace(module)


def save_prosparse_llama_mlp_act_trace(model, fname):
    collect_mlp_act_trace(model)

    import numpy as np

    global mlp_act_total, mlp_act_freq, mlp_act_sparsity

    np.savez(
        fname,
        mlp_act_total=np.array(mlp_act_total),
        mlp_act_freq=np.array(mlp_act_freq),
        mlp_act_sparsity=np.array(mlp_act_sparsity),
    )
    print(f"Dumping sparsity trace to {fname}")


def extract_fc_weights(model):

    fc1_weights = []
    for i, layer in enumerate(model.model.decoder.layers):
        fc1_weights.append(layer.fc1.weight.detach().clone().float().cpu())

    return fc1_weights


def enable_llama2_topk(model, keep_ratio):
    # Modify layers in Llama2
    for layer in model.model.layers:
        list_mlp_layer = get_layers(module=layer, layers=[LlamaMLP, LlamaSdpaAttention])

        for name, module in list_mlp_layer.items():
            if isinstance(module, LlamaMLP):
                module.keep_ratio = keep_ratio
                module.forward = types.MethodType(layers.mlp_gate3_topk, module)

            # if isinstance(module, LlamaSdpaAttention):
            #     module.keep_ratio = keep_ratio
            #     module.forward = types.MethodType(layers.qkvo_topk, module)


def enable_llama2_ffn_sample(model):
    # Modify layers in Llama2
    for layer in model.model.layers:
        list_mlp_layer = get_layers(module=layer, layers=[LlamaMLP])

        for name, module in list_mlp_layer.items():
            if isinstance(module, LlamaMLP):
                module.list_inp = []
                module.list_out = []
                module.forward = types.MethodType(layers.mlp_gate3_sample, module)


def enable_llama2_qkv_sample(model):
    # Modify layers in Llama2
    for layer in model.model.layers:
        list_mlp_layer = get_layers(module=layer, layers=[LlamaSdpaAttention])

        for name, module in list_mlp_layer.items():
            if isinstance(module, LlamaSdpaAttention):
                module.list_inp = []
                module.forward = types.MethodType(layers.wkv_sample, module)


def extract_llama2_mlp_input_weights(model):
    list_mlp_inp = []
    list_mlp_out = []
    list_mlp_W = []
    for i, layer in enumerate(model.model.layers):
        list_mlp_layer = get_layers(module=layer, layers=[LlamaMLP])

        for name, module in list_mlp_layer.items():
            print(f"Layer-{i}")
            if isinstance(module, LlamaMLP):
                list_mlp_inp.append(module.list_inp)
                list_mlp_out.append(module.list_out)
                list_mlp_W.append(
                    {
                        "gate": module.gate_proj.weight.detach()
                        .clone()
                        .bfloat16()
                        .cpu(),
                        "down": module.down_proj.weight.detach()
                        .clone()
                        .bfloat16()
                        .cpu(),
                        "up": module.up_proj.weight.detach().clone().bfloat16().cpu(),
                    }
                )

                #
                del module.list_inp
                del module.list_out

                # print(len(module.list_inp))
                # print(module.list_inp[0].shape)

    return {"mlp_inp": list_mlp_inp, "mlp_out": list_mlp_out, "mlp_weight": list_mlp_W}


def enable_llama2_mlp_svd(
    model, dict_gate_weight_svd, dict_density_threshold, keep_ratio
):
    num_layers = 0
    for i, layer in enumerate(model.model.layers):
        list_mlp_layer = get_layers(module=layer, layers=[LlamaMLP])
        num_layers += len(list_mlp_layer)
        
    
    for i, layer in enumerate(model.model.layers):
        list_mlp_layer = get_layers(module=layer, layers=[LlamaMLP])

        skip_layers = 3
        for name, module in list_mlp_layer.items():
            layer_name = f"{name}-{i}"

            if i>=skip_layers and i<num_layers-skip_layers and isinstance(module, LlamaMLP):
            # if isinstance(module, LlamaMLP):
                print(f"{layer_name} of {num_layers}")
                module.forward = types.MethodType(layers.mlp_gate3_svd, module)
                module.gate_weight_Q_svd = (
                    dict_gate_weight_svd[layer_name + "-Q"].bfloat16().to("cuda")
                )
                module.gate_weight_R_svd = (
                    dict_gate_weight_svd[layer_name + "-R"].bfloat16().to("cuda")
                )
                module.keep_ratio = keep_ratio
                module.density_threshold = dict_density_threshold[layer_name]


def enable_llama2_qkv_svd(
    model, dict_gate_weight_svd, dict_density_threshold, keep_ratio
):
    for i, layer in enumerate(model.model.layers):
        list_mlp_layer = get_layers(module=layer, layers=[LlamaSdpaAttention])

        for name, module in list_mlp_layer.items():

            if isinstance(module, LlamaSdpaAttention):
                module.forward = types.MethodType(layers.qkvo_svd, module)
                module.keep_ratio = keep_ratio

                module.q_weight_Q_svd = (
                    dict_gate_weight_svd[f"{name}-q-{i}-Q"].bfloat16().to("cuda")
                )
                module.q_weight_R_svd = (
                    dict_gate_weight_svd[f"{name}-q-{i}-R"].bfloat16().to("cuda")
                )
                module.q_density_threshold = dict_density_threshold[f"{name}-q-{i}"]

                module.k_weight_Q_svd = (
                    dict_gate_weight_svd[f"{name}-k-{i}-Q"].bfloat16().to("cuda")
                )
                module.k_weight_R_svd = (
                    dict_gate_weight_svd[f"{name}-k-{i}-R"].bfloat16().to("cuda")
                )
                module.k_density_threshold = dict_density_threshold[f"{name}-k-{i}"]

                module.v_weight_Q_svd = (
                    dict_gate_weight_svd[f"{name}-v-{i}-Q"].bfloat16().to("cuda")
                )
                module.v_weight_R_svd = (
                    dict_gate_weight_svd[f"{name}-v-{i}-R"].bfloat16().to("cuda")
                )
                module.v_density_threshold = dict_density_threshold[f"{name}-v-{i}"]
