# %%
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from collections import namedtuple

from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock


n_trial = 2000
seq_len = 2048
bsz = [1, 4, 8, 16]
device = "cuda"

named_config = namedtuple(
    "config",
    [
        "hidden_size",
        "intermediate_size",
        "hidden_act",
        "num_local_experts",
        "num_experts_per_tok",
        "pretraining_tp",
    ],
)


# LLM Models
class LLMConfig:
    """
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        type_mlp (`str`, *optional*, defaults to `"gate"`):
            The type of MLP used in the decoder. Can be either `"ffn"` or `"gate"`.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
    """

    model_type = None

    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=11008,
        type_mlp="gate",
        num_local_experts=None,
        num_experts_per_tok=None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.type_mlp = type_mlp
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act


MODELS = {
    "OPT-6.7B": LLMConfig(
        hidden_size=4096,
        intermediate_size=16384,
        type_mlp="ffn",
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="relu",
        max_position_embeddings=2048,
    ),
    "OPT-13B": LLMConfig(
        hidden_size=5120,
        intermediate_size=20480,
        type_mlp="ffn",
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=None,
        hidden_act="relu",
        max_position_embeddings=2048,
    ),
    "Llama2-7B": LLMConfig(
        hidden_size=4096,
        intermediate_size=11008,
        type_mlp="gate",
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
    ),
    "Llama2-13B": LLMConfig(
        hidden_size=5120,
        intermediate_size=13824,
        type_mlp="gate",
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        hidden_act="silu",
        max_position_embeddings=4096,
    ),
    "Mixtral-8x7B": LLMConfig(
        hidden_size=4096,
        intermediate_size=14336,
        type_mlp="moe",
        num_experts_per_tok=2,
        num_local_experts=8,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
    ),
    "DS-MoE-16B": LLMConfig(
        hidden_size=2048,
        intermediate_size=1408,
        type_mlp="moe",
        num_experts_per_tok=8,
        num_local_experts=64,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=4096,
    ),
}


# def reject_outliers(data, m=2):
# return data[abs(data - np.mean(data)) < m * np.std(data)]


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    return data[s < m]


def calc_mean_var(inp, m=2):
    inp = reject_outliers(inp, m)
    m, v = np.mean(inp), np.var(inp)
    return m, v


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self, in_features, head_num, bias=False, activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                "`in_features`({}) should be divisible by `head_num`({})".format(
                    in_features, head_num
                )
            )
        self.in_features = in_features
        self.head_num = head_num

    def forward(self, q, k, v, mask=None):
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        # y = self._reshape_from_batches(y)
        return y

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )


class utest:
    def __init__(
        self,
        dim_emb,
        dim_inter,
        n_head,
        seq_len,
        n_experts,
        n_act_experts,
        device="cuda",
    ) -> None:
        self.dim_emb = dim_emb
        self.dim_inter = dim_inter

        self.n_head = n_head
        self.seq_len = seq_len

        self.WQ = nn.Linear(self.dim_emb, self.dim_emb, dtype=torch.bfloat16).to(device)
        self.WK = nn.Linear(self.dim_emb, self.dim_emb, dtype=torch.bfloat16).to(device)
        self.WV = nn.Linear(self.dim_emb, self.dim_emb, dtype=torch.bfloat16).to(device)
        self.WO = nn.Linear(self.dim_emb, self.dim_emb, dtype=torch.bfloat16).to(device)

        self.config = named_config(
            dim_emb, dim_inter, "relu", n_experts, n_act_experts, 0
        )
        self.ffn = LlamaMLP(self.config).bfloat16().to(device)
        self.mha = (
            MultiHeadAttention(in_features=self.dim_emb, head_num=self.n_head)
            .bfloat16()
            .to(device)
        )
        self.moe = MixtralSparseMoeBlock(self.config).bfloat16().to(device)

    def QKVO(self, inp):
        Q = self.WQ(inp)
        K = self.WK(inp)
        V = self.WV(inp)
        O = self.WO(inp)
        return Q, K, V, O

    def MHA(self, q, k, v):
        return self.mha(q, k, v)

    def FFN(self, inp):
        return self.ffn(inp)

    def MoE(self, inp):
        return self.moe(inp)


# df = pd.DataFrame(columns=["model", "batch_size", "QKVO", "MHA", "FFN", "MoE"])
df = None
for m_name in MODELS.keys():
    model = MODELS[m_name]
    if model.num_local_experts == None:
        dim_emb, dim_inter, n_head, n_experts, n_act_experts = (
            model.hidden_size,
            model.intermediate_size,
            model.num_attention_heads,
            1,
            1,
        )
    else:
        dim_emb, dim_inter, n_head, n_experts, n_act_experts = (
            model.hidden_size,
            model.intermediate_size,
            model.num_attention_heads,
            model.num_local_experts,
            model.num_experts_per_tok,
        )

    ut = utest(dim_emb, dim_inter, n_head, seq_len, n_experts, n_act_experts)

    for bs in bsz:
        # Evaluate QKVO latency
        time_hist = []
        for i in range(n_trial):
            inp = torch.rand(size=(bs, dim_emb)).bfloat16().to(device)

            start = time.time()
            out = ut.QKVO(inp)
            end = time.time()

            elpased_us = (end - start) * 1e6
            time_hist.append(elpased_us)

        m_qkvo, v_qkvo = calc_mean_var(np.array(time_hist))
        print(f"{m_name}: Time_us-QKVO bsz={bs} = {m_qkvo:.3f} +/- {v_qkvo:.3f}")

        # Evaluate MHA latency
        time_hist = []
        k = torch.rand(size=(bs, seq_len, dim_emb)).bfloat16().to(device)
        v = torch.rand(size=(bs, seq_len, dim_emb)).bfloat16().to(device)
        for i in range(n_trial):
            q = torch.rand(size=(bs, 1, dim_emb)).bfloat16().to(device)

            start = time.time()
            out = ut.mha(q, k, v)
            end = time.time()

            elpased_us = (end - start) * 1e6
            time_hist.append(elpased_us)

        m_mha, v_mha = calc_mean_var(np.array(time_hist))
        print(f"{m_name}: Time_us-MHA bsz={bs} = {m_mha:.3f} +/- {v_mha:.3f}")

        # Evaluate FFN latency
        time_hist = []
        m_ffn, v_ffn, m_moe, v_moe = 0, 0, 0, 0
        if n_experts == 1:
            for i in range(n_trial):
                inp = torch.rand(size=(bs, dim_emb)).bfloat16().to(device)

                start = time.time()
                out = ut.ffn(inp)
                end = time.time()

                elpased_us = (end - start) * 1e6
                time_hist.append(elpased_us)

            m_ffn, v_ffn = calc_mean_var(np.array(time_hist))
            print(f"{m_name}: Time_us-FFN bsz={bs} = {m_ffn:.3f} +/- {v_ffn:.3f}")
        else:
            # Evaluate MoE latency
            for i in range(n_trial):
                inp = torch.rand(size=(bs, 1, dim_emb)).bfloat16().to(device)

                start = time.time()
                out = ut.moe(inp)
                end = time.time()

                elpased_us = (end - start) * 1e6
                time_hist.append(elpased_us)

            m_moe, v_moe = calc_mean_var(np.array(time_hist))
            print(f"{m_name}: Time_us-MoE bsz={bs} = {m_moe:.3f} +/- {v_moe:.3f}")

        dict_res = {
            "model": m_name,
            "batch_size": bs,
            "seq_len": seq_len,
            "QKVO": m_qkvo,
            "MHA": m_mha,
            "FFN": m_ffn,
            "MoE": m_moe,
        }

        if df is None:
            df = pd.DataFrame([dict_res])
        else:
            df.loc[len(df)] = dict_res


df.to_csv("runtime_baseline_gpu.csv", index=False)
