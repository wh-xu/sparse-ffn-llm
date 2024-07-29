# TASKS="arc_easy arc_challenge boolq openbookqa winogrande piqa hellaswag lambada"
# MODEL=deepseek-ai/deepseek-moe-16b-base

# Task settings
TASKS="arc_easy arc_challenge boolq openbookqa winogrande piqa hellaswag"
LIMIT=200

LOG_PATH=/pvc/logs
mkdir $LOG_PATH -p

# Llama2 baseline
MODEL=meta-llama/Llama-2-7b-hf
# WORKLOAD=baseline
# OUT_FILE=$LOG_PATH/log_llama2-13b_baseline.log
# python benchmark_llm.py --model $MODEL:$WORKLOAD --tasks $TASKS --limit $LIMIT | tee $OUT_FILE

# Llama2 topk-sparse
MODEL=meta-llama/Llama-2-13b-hf
WORKLOAD=topk
# for r in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
# do
#     OUT_FILE=$LOG_PATH/log_llama2-13b_topk_$r.log
#     echo $OUT_FILE
#     python benchmark_llm.py --model $MODEL:$WORKLOAD --tasks $TASKS --limit $LIMIT --model_args keep_ratio=$r | tee $OUT_FILE
# done


# Llama2 tqr-sparse
MODEL=meta-llama/Llama-2-7b-hf
WORKLOAD=svd
for METHOD in tqr
do
    for RANK in 512 1024
    do
        for r in 0.75 0.5
#         # for r in 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.3 
        do
            OUT_FILE=$LOG_PATH/log_llama2-7b_${METHOD}_${RANK}_$r_skip_3.log
            echo $OUT_FILE
            python benchmark_llm.py --model $MODEL:$WORKLOAD --tasks $TASKS --limit $LIMIT --model_args method=$METHOD,rank=$RANK,keep_ratio=$r | tee $OUT_FILE
        done
    done
done

