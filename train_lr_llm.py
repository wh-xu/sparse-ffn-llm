# %%
from tqdm.auto import tqdm
import os, argparse

import torch

from sparse_llm import lr
from sparse_llm.utils import load, eval_ppl, get_calibration_data
from sparse_llm.common_ops import get_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--method", type=str, default="tqr")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--n_sample", type=int, default=128, help="# of calibration samples"
    )
    # parser.add_argument(
    # "--model_args",
    # type=lambda x: {k: v for k, v in (i.split("=") for i in x.split(","))},
    # help="comma-separated field=position pairs, e.g. Date=0,Amount:2",
    # )
    args = parser.parse_args()
    print(args)

    model_config: list = args.model.split(":")
    model, tokenizer = load(model_config[0], device_map="cpu", load_in_8bit=True)
    print(model.config)

    #############################################################
    # Load calibration data and run
    #############################################################
    data_calib = get_calibration_data(
        tokenizer=tokenizer, n_samples=args.n_sample, block_size=256
    )

    # Run calibration
    if "Llama-2" in model_config[0]:
        from sparse_llm.sparse_llama import (
            enable_llama2_ffn_sample,
            enable_llama2_qkv_sample,
        )

        enable_llama2_ffn_sample(model)
        # enable_llama2_qkv_sample(model)

    for i in tqdm(range(data_calib.shape[0]), desc="Running calibration..."):
        model(data_calib[i : i + 1].to(model.device))

    # model, tokenizer = load(
    #     model_config[0], device_map="cpu", torch_dtype=torch.bfloat16
    # )

    method = args.method
    rank = args.rank

    keep_ratio = [0.9, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]

    #############################################################
    # Train and dump
    #############################################################
    dict_gate_weight_svd, dict_gate_ratio_threshold = {}, {}
    # dict_qkv_weight_svd, dict_qkv_ratio_threshold = {}, {}
    for i, layer in enumerate(model.model.layers):

        if "Llama-2" in model_config[0]:
            from transformers.models.llama.modeling_llama import (
                LlamaMLP,
                # LlamaSdpaAttention,
            )

            list_mlp_layer = get_layers(module=layer, layers=[LlamaMLP])

        for name, module in list_mlp_layer.items():
            print(f"Layer-{i}: {name}", flush=True)

            inp_batch = torch.concat(module.list_inp, dim=1).bfloat16().to("cuda")

            if isinstance(module, LlamaMLP):
                # print(inp_batch)
                # print(module.gate_proj.weight.data)
                gate_weight_Q_svd, gate_weight_R_svd = lr.generate_pred_model(
                    weight=module.gate_proj.weight.data,
                    inp=inp_batch,
                    method=method,
                    rank=rank,
                    epochs=200,
                )

                gate_weight_svd = gate_weight_Q_svd @ gate_weight_R_svd
                gate_act_appr = inp_batch @ gate_weight_svd.bfloat16().to("cuda").T
                list_threshold = lr.get_threshold_from_density(
                    gate_act_appr, keep_ratio
                )

                dict_gate_weight_svd[f"{name}-{i}-Q"] = gate_weight_Q_svd.cpu()
                dict_gate_weight_svd[f"{name}-{i}-R"] = gate_weight_R_svd.cpu()
                dict_gate_ratio_threshold[f"{name}-{i}"] = dict(
                    map(lambda i, j: (i, j), keep_ratio, list_threshold)
                )

            # if isinstance(module, LlamaSdpaAttention):
            #     for m in ["q", "k", "v"]:
            #         gate_weight_Q_svd, gate_weight_R_svd = lr.generate_pred_model(
            #             weight=module.q_proj.weight.data,
            #             inp=inp_batch,
            #             method=method,
            #             rank=rank,
            #             epochs=200,
            #         )

            #         gate_weight_svd = gate_weight_Q_svd @ gate_weight_R_svd
            #         gate_act_appr = inp_batch @ gate_weight_svd.bfloat16().to("cuda").T
            #         list_threshold = lr.get_threshold_from_density(
            #             gate_act_appr, keep_ratio
            #         )

            #         dict_qkv_weight_svd[f"{name}-{m}-{i}-Q"] = gate_weight_Q_svd.cpu()
            #         dict_qkv_weight_svd[f"{name}-{m}-{i}-R"] = gate_weight_R_svd.cpu()
            #         dict_qkv_ratio_threshold[f"{name}-{m}-{i}"] = dict(
            #             map(lambda i, j: (i, j), keep_ratio, list_threshold)
            #         )
            #         print(dict_qkv_weight_svd.keys())
    #############################################################

    #############################################################
    # Dump trained data
    #############################################################
    model_name = model_config[0].split("/")[1]
    torch.save(
        dict_gate_weight_svd,
        f"./cached_svd/{model_name}.gate_weight.{method}.{rank}.pt",
    )
    torch.save(
        dict_gate_ratio_threshold,
        f"./cached_svd/{model_name}.gate_ratio_threshold.{method}.{rank}.pt",
    )

    # torch.save(
    #     dict_qkv_weight_svd,
    #     f"./cached_svd/{model_name}.qkv_weight.{method}.{rank}.pt",
    # )
    # torch.save(
    #     dict_qkv_ratio_threshold,
    #     f"./cached_svd/{model_name}.qkv_ratio_threshold.{method}.{rank}.pt",
    # )
    #############################################################


if __name__ == "__main__":
    main()
