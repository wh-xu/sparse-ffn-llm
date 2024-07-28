
python examples/eval_long_ppl.py --model_name_or_path facebook/opt-1.3b \
    --enable_sparse_ffn

# python examples/eval_long_ppl.py --enable_start_recent_kv_cache --enable_pos_shift \
    # --start_size 4 --recent_size 520