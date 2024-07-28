# Task settings
LIMIT=10
TASKS="winogrande piqa hellaswag lambada"

# Llama2-7b baseline
MODEL=meta-llama/Llama-2-7b-hf:sample
python benchmark_llm.py --model $MODEL --tasks $TASKS --limit $LIMIT --model_args out_file=sample_data_llama2_7b_mlp.pt
