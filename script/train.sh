



# MASTER_ADDR=$CHIEF_IP
MASTER_PORT=60001
NUM_GPUS=$NODE_NUM


WORKSPACE=/mnt/nlp/gaoqiang/training
export PYTHONPATH=${WORKSPACE}



MODEL_PATH=/mnt/nlp/gaoqiang/ckpt/Llama-2-7b-chat-hf
# MODEL_PATH=/mnt/nlp/gaoqiang/ckpt/Qwen2-0.5B-Instruct
DATA_PATH=/mnt/nlp/gaoqiang/training/sft_train.json
# DATA_PATH=/mnt/nlp/gaoqiang/training/retrieval_train.json
EVAL_PATH=None

EVAL_OUTPUT_PATH=None
MODEL_OUTPUT_DIR=$WORKSPACE/output/
MODEL_TYPE=llama2-7b-Instruct
TASK=retrieval
# data config
FORMAT_MODE=llama2
MAX_RESPONSE=1


# training setups
#---------------------------------------------------------------------------------
# BATCH_SIZE=
MICRO_BATCH_SIZE=8
# NUM_GPUS=8
echo $NUM_GPUS
echo $MICRO_BATCH_SIZE

GRADIENT_ACCUMULATION_STEP=1


MAX_LENGTH=4096


PADDING_SIDE="right"
TRUNCATION_SIDE="left"
POOLING_TYPE="last"
EPOCH=1
LEARNING_RATE=2e-5




# deepspeed setups
#---------------------------------------------------------------------------------
# DS_ZERO=3
# if [[ $DS_ZERO = 2 ]]; then
#     DEEPSPEED=${WORKSPACE}/configs/default_zero2_config.json
# else
#     DEEPSPEED=${WORKSPACE}/configs/default_offload_opt_param.json
# fi

# TMP_DIR=${WORKSPACE}/tmp
# mkdir -p $TMP_DIR
# echo $NODE_IP_LIST > ${TMP_DIR}/env.txt
 
# # generate hostfile and pssh.hosts
# sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
# sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts

DEEPSPEED=${WORKSPACE}/configs/ds_config_zero3.json
# DEEPSPEED=${WORKSPACE}/configs/ds_config_zero2_no_offload.json
# DEEPSPEED=${WORKSPACE}/configs/ds_config_zero1.json


# output config
#----------------------------------------------------------------------------------
EXPERIMENT_NAME=$(date +'%m-%d')_${TASK}_${MODEL_TYPE}_${DATA_NAME}_bs_${BATCH_SIZE}_maxlen_${MAX_LENGTH}_pad_${PADDING_SIDE}_lr_${LEARNING_RATE}_format_${FORMAT_MODE}

OUTPUT_DIR=${MODEL_OUTPUT_DIR}/${EXPERIMENT_NAME}
LOGS_PATH=${OUTPUT_DIR}/logs

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH


echo "begin experiment ${EXPERIMENT_NAME}"


CURRENT_TIME=$(date +'%m-%d-%Y_%H:%M:%S')

# export CMD="deepspeed   --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} finetune.py \
export CMD="deepspeed  --include localhost:0,3 --master_port=${MASTER_PORT} train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --eval_output_path $EVAL_OUTPUT_PATH \
    --output_dir $OUTPUT_DIR\
    --do_train True \
    --do_eval False \
    --padding_side $PADDING_SIDE \
    --num_train_epochs 2 \
    --model_max_length $MAX_LENGTH \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size 1\
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEP} \
    --evaluation_strategy "no" \
    --eval_steps 1   \
    --eval_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_steps 0.1 \
    --save_total_limit 15 \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed $DEEPSPEED \
    --bf16 True \
    --stage sft \
    --use_lora False"

echo $CMD
eval ${CMD} 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt
set +x
