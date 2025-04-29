
CURRENT_TIME=$(date +'%m-%d-%Y-%H:%M:%S')
WORKSPACE=/mnt/nlp/gaoqiang/training/eval
export PYTHONPATH=${WORKSPACE}

MODEL_PATH=/mnt/nlp/gaoqiang/training/output/01-08_retrieval_llama2-7b-Instruct__bs__maxlen_2048_pad_right_lr_2e-6_format_llama2/checkpoint-449
FILE_PATH=/mnt/nlp/gaoqiang/training/test_process.json


export CUDA_VISIBLE_DEVICES=0,3
# #  CUDA_VISIBLE_DEVICES=4,5,6,7
export CMD="python3 eval.py \
        --model_name_or_path_baseline ${MODEL_PATH} \
        --file_path ${FILE_PATH} \
        --output_path ./output \
        --use_vllm True"

echo $CMD
eval ${CMD} 2>&1 | tee -a $WORKSPACE/log/log_event_summarize_${CURRENT_TIME}.txt
set +x
