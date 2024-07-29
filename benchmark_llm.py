# %%
import os, argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LM_HARNESS_CACHE_PATH"] = "./cache_for_lm_harness"

import torch

from lm_eval import evaluator
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM

from sparse_llm.utils import load, eval_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2b")
    parser.add_argument("--eval", nargs="+", type=str, default=["lm_eval", "ppl"])
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["winogrande", "piqa", "hellaswag", "lambada"],
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--model_args",
        type=lambda x: {k: v for k, v in (i.split("=") for i in x.split(","))},
        help="comma-separated field=position pairs, e.g. Date=0,Amount:2",
    )
    args = parser.parse_args()
    print(args.model_args)

    model_config: list = args.model.split(":")
    model_name = model_config[0].split("/")[-1]

    #############################################################
    # Parse and pre-process customized model
    #############################################################
    model, tokenizer = load(model_config[0])
    print(model.config)

    if len(model_config) > 1 and "baseline" != model_config[1]:
        if "Llama-2" in model_config[0]:
            if "topk" in model_config[1].lower():
                from sparse_llm.sparse_llama import enable_llama2_topk

                enable_llama2_topk(model, float(args.model_args["keep_ratio"]))

            elif "svd" in model_config[1].lower():
                from sparse_llm.sparse_llama import (
                    enable_llama2_mlp_svd,
                    enable_llama2_qkv_svd,
                )

                method = args.model_args["method"]
                rank = int(args.model_args["rank"])
                keep_ratio = float(args.model_args["keep_ratio"])

                dict_gate_weight_svd = torch.load(
                    f"./cached_svd/{model_name}.gate_weight.{method}.{rank}.pt"
                )
                dict_density_threshold = torch.load(
                    f"./cached_svd/{model_name}.gate_ratio_threshold.{method}.{rank}.pt"
                )
                enable_llama2_mlp_svd(
                    model, dict_gate_weight_svd, dict_density_threshold, keep_ratio
                )

                # dict_qkv_weight_svd = torch.load(
                #     f"./cached_svd/{model_name}.qkv_weight.{method}.{rank}.pt"
                # )
                # dict_qkv_density_threshold = torch.load(
                #     f"./cached_svd/{model_name}.qkv_ratio_threshold.{method}.{rank}.pt"
                # )
                # enable_llama2_qkv_svd(
                #     model, dict_qkv_weight_svd, dict_qkv_density_threshold, keep_ratio
                # )

            elif "sample" in model_config[1].lower():
                from sparse_llm.sparse_llama import enable_llama2_mlp_sample

                enable_llama2_mlp_sample(model)

        elif "gemma" in model_config[0]:
            from streaming_llm.sparse_gemma import (
                enable_gemma_sparsity,
                enable_gemma_test,
            )

            if "sparse" in model_config[1].lower():
                enable_gemma_sparsity(model)
            elif "topk" in model_config[1].lower():
                enable_gemma_test(model)

        elif "ReluLLaMA" in model_config[0] and "svd" in model_config[1].lower():
            from streaming_llm.sparse_llama import (
                enable_llama_mlp_forward_with_svd_pred,
            )

            # recall = float(args.model_args[0])
            rank = int(args.model_args[0])
            thres = float(args.model_args[1])
            print(f"Rank = {rank} - thres = {thres}")

            model_name = model_config[0].split("/")[1]
            cache_svd_path = f"./cached_svd/{model_name}"
            enable_llama_mlp_forward_with_svd_pred(
                model, cache_svd_file_prefix=cache_svd_path, rank=rank, thres=thres
            )
        elif "opt" in model_config[0].lower() and "trace" in model_config[1].lower():
            from streaming_llm.sparse_opt import dump_opt_sparsity_trace

            dump_opt_sparsity_trace(model)
            print(f"Added code to {model_config[0]} for trace dump")

        elif (
            "prosparse-llama" in model_config[0].lower()
            and "trace" in model_config[1].lower()
        ):
            from streaming_llm.sparse_llama import dump_prosparse_llama_sparsity_trace

            dump_prosparse_llama_sparsity_trace(model)
            print(f"Added code to {model_config[0]} for trace dump")
    #############################################################

    #############################################################
    # Benchmark model performance
    #############################################################
    if "lm_eval" in args.eval:
        lm = HFLM(pretrained=model, tokenizer=tokenizer)
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            log_samples=False,
            limit=args.limit,
            batch_size="auto",
            cache_requests=True,
        )
        print(make_table(results))

    if "ppl" in args.eval:
        ppl = eval_ppl(model=model, tokenizer=tokenizer, dataset="wikitext")
        print(f"PPL = {ppl:.3f}")
    #############################################################


if __name__ == "__main__":
    main()
