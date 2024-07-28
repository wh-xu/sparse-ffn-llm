# MODEL=meta-llama/Llama-2-7b-hf
MODEL=meta-llama/Llama-2-13b-hf
CALIB_SAMPLE=16

for METHOD in tqr
do
    for RANK in 1024 1728
    do
        OUT_FILE=logs/log_llama2-13b_train_${METHOD}_${RANK}_$CALIB_SAMPLE.log
        echo $OUT_FILE
        python train_lr_llm.py --model $MODEL --method $METHOD --rank $RANK --n_sample $CALIB_SAMPLE | tee $OUT_FILE
    done
done

# METHOD=tqr
# RANK=512
# # CALIB_SAMPLE=64
# OUT_FILE=logs/test.log

# python train_lr_llm.py --model $MODEL --method $METHOD --rank $RANK --n_sample $CALIB_SAMPLE | tee $OUT_FILE