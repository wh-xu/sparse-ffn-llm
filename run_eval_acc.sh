# TASKS="arc_easy arc_challenge boolq openbookqa winogrande piqa hellaswag lambada"
# MODEL=deepseek-ai/deepseek-moe-16b-base

# Task settings
TASKS="arc_easy arc_challenge boolq openbookqa winogrande piqa hellaswag"
LIMIT=200



# Llama2-7b baseline
MODEL=meta-llama/Llama-2-7b-hf:baseline
OUT_FILE=logs/log_llama2-7b_baseline.log

# MODEL=meta-llama/Llama-2-13b-hf:baseline
# OUT_FILE=logs/log_llama2-13b_baseline.log

MODEL=facebook/opt-6.7b 
OUT_FILE=logs/log_opt-6.7b_baseline.log

python benchmark_llm.py --model $MODEL --tasks $TASKS --limit $LIMIT | tee $OUT_FILE

# Llama2-7b topk-sparse
# MODEL=meta-llama/Llama-2-7b-hf:topk
# for r in 0.9 0.75 0.5 0.25
for r in 0.75
do
    OUT_FILE=logs/log_llama2-7b_topk_$r.log
    # python benchmark_llm.py --model $MODEL --tasks $TASKS --limit $LIMIT --model_args keep_ratio=$r | tee $OUT_FILE
done

###################################################

# MODEL=meta-llama/Llama-2-7b-hf:svd

# for METHOD in tqr
# do
#     for RANK in 512
#     do
#         for r in 0.9 0.75 0.5 0.4 0.3 0.2 0.1
#         do
#             OUT_FILE=logs/log_llama2-7b_${METHOD}_${RANK}_$r.new.log
#             echo $OUT_FILE
#             python benchmark_llm.py --model $MODEL --tasks $TASKS --limit $LIMIT --model_args method=$METHOD,rank=$RANK,keep_ratio=$r | tee $OUT_FILE
#         done
#     done
# done


# # MODEL=meta-llama/Llama-2-7b-hf:svd
# r=0.75
# METHOD=tqr
# RANK=1024
# python benchmark_llm.py --model $MODEL --tasks $TASKS --limit $LIMIT --model_args method=$METHOD,rank=$RANK,keep_ratio=$r
