# %%
import os, argparse

import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

device = "cuda"

# args = parse_args()
# print(args)

args = argparse.Namespace(
    **{
        "model_name_or_path": "facebook/opt-1.3b",
        "dataset_name": "wikitext",
        "task": "wikitext-2-raw-v1",
        "split": "test",
        "num_samples": 10,
        "num_eval_tokens": None,
        "output_dir": "./",
        "enable_start_recent_kv_cache": False,
        "start_size": 1,
        "recent_size": 255,
        "enable_pos_shift": False,
        "enable_sparse_ffn": True,
    }
)

data = load_dataset(args.dataset_name, args.task, split=args.split)
# print(data["text"][0:4])

model, tokenizer = load(args.model_name_or_path, output_attentions=True)
print(model.config)
print(model)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None


if args.enable_sparse_ffn:
    if "opt" in model.config.model_type:
        from streaming_llm.sparse_opt import (
            enable_opt_sparsity_record,
            extract_fc_weights,
        )

        enable_opt_sparsity_record(model)


if args.enable_sparse_ffn:
    sample_data = {
        "text": [],
        "inp_fc1": [],
        "out_fc1": [],
        "w_fc1": [],
        "out_ffn": [],
        "mha_inp": [],
        "mha_out": [],
    }
    sample_data["w_fc1"] = extract_fc_weights(model)


if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")


# os.makedirs(args.output_dir, exist_ok=True)
# f = open(f"{args.output_dir}/log.txt", "w")


num_eval_tokens = 0
for text in tqdm(data["text"][: args.num_samples]):
    if args.enable_sparse_ffn:
        sample_data["text"].append(text)
        sample_data["mha_inp"].append([])
        sample_data["mha_out"].append([])
        sample_data["inp_fc1"].append([])
        sample_data["out_fc1"].append([])
        sample_data["out_ffn"].append([])

    encodings = tokenizer(text, return_tensors="pt")
    # print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    # print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1), leave=False)

    # model.attn_head_score = [None]
    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)

            if args.enable_sparse_ffn:
                L = len(outputs.hidden_states)
                sample_data["mha_inp"][-1].append(
                    [outputs.hidden_states[i] for i in range(0, L, 5)]
                )
                sample_data["mha_out"][-1].append(
                    [outputs.hidden_states[i] for i in range(1, L, 5)]
                )
                sample_data["inp_fc1"][-1].append(
                    [outputs.hidden_states[i] for i in range(2, L, 5)]
                )
                sample_data["out_fc1"][-1].append(
                    [outputs.hidden_states[i] for i in range(3, L, 5)]
                )
                sample_data["out_ffn"][-1].append(
                    [outputs.hidden_states[i] for i in range(4, L, 5)]
                )

        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

# f.close()

ppl = torch.exp(torch.stack(nlls).mean())

print(f"Final ppl = {ppl.item()}")

# with open(f"{args.output_dir}/ppl.txt", "w") as f:
#     f.write(f"{ppl.item()}\n")


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def cos_sim(x, y):
    norm_x = norm(x, axis=-1)
    norm_y = norm(y, axis=-1)
    cos_sim = np.sum(x * y, axis=-1) / norm_x / norm_y
    return cos_sim


def pw_cos_sim(x, y):
    norm_x = norm(x, axis=-1)
    norm_y = norm(y, axis=-1)
    norm_ = norm_x.reshape(x.shape[0], 1) * norm_y.reshape(1, y.shape[0])
    cos_sim = x @ y.T / norm_
    return cos_sim


# %% MHA similarity
n = 4
x = np.stack(sample_data["mha_inp"][n]).squeeze()
x_fx = np.stack(sample_data["mha_out"][n]).squeeze()

x, y = x[0], x_fx[0]
sim = cos_sim(x, y)
print(sim)

# %%
act_l = np.stack(sample_data["mha_inp"][n]).squeeze()
# lm_w = model.get_submodule("lm_head").weight.detach().clone().float().cpu().numpy()

i = 0
x, y = act_l[i][0:-2], act_l[i][1:-1]
sim = cos_sim(x, y)
print(sim)

# all_lm_score = act_l @ lm_w.T

# sim = cos_sim(all_lm_score[0], all_lm_score[0])
# print(sim)


# %% MHA Sparsity


# %%
i_sample = 3
j_seq = 3
layer = 3
fc_inp = sample_data["inp_fc1"][i_sample][j_seq][layer]
fc_w = sample_data["w_fc1"][layer].T
gt = sample_data["out_fc1"][i_sample][j_seq][layer]

out = fc_inp @ fc_w
out[out <= 0] = 0.0

err = (out - gt).abs().mean()
print(err)

# Dump to file
torch.save(sample_data, "./sampled_data/sampled-fc1.OPT-1.3b.pt")
# torch.save(sample_data['w_fc1'], )
# torch.save(sample_data['out_fc1'], )
