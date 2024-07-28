import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os.path as osp
import ssl
import urllib.request
import os
import json

from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")
    parser.add_argument("--enable_sparse_ffn", action="store_true")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


def load(model_name_or_path, **kargs):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        cache_dir="./cached_models",
    )
    # tokenizer.add_bos_token = False
    # if tokenizer.pad_token_id is None:
    #     if tokenizer.eos_token_id is not None:
    #         tokenizer.pad_token_id = tokenizer.eos_token_id
    #     else:
    #         tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir="./cached_models",
        device_map="auto",
        # max_memory={0:"10GiB","cpu": "32GiB"},
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
        **kargs,
    )
    model.eval()
    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict


def calc_mean_var(inp):
    m, v = np.mean(inp, axis=0), np.var(inp, axis=0)
    return m, v


def get_calibration_data(
    data="pileval",
    tokenizer=None,
    n_samples=512,
    block_size=512,
    dataset_cache_dir="./cached_dataset",
):
    if data == "pileval":
        dataset = load_dataset(
            "mit-han-lab/pile-val-backup",
            split="validation",
            cache_dir=dataset_cache_dir,
        )
    else:
        raise NotImplementedError

    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return torch.cat(
        [cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)],
        dim=0,
    )



# https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
def eval_ppl(model, tokenizer, dataset="wikitext"):
    if dataset == "wikitext":
        testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise NotImplementedError

    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating PPL..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    return ppl.item()
