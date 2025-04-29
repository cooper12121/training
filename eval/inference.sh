# connection config
#---------------------------------------------------------------------------------
# NET_TYPE="low"
# export NCCL_IB_TIMEOUT=24
# if [[ "${NET_TYPE}" = "low" ]]; then
#     export NCCL_SOCKET_IFNAME=eth1
#     export NCCL_IB_GID_INDEX=3
#     export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
#     export NCCL_IB_SL=3
#     export NCCL_CHECK_DISABLE=1
#     export NCCL_P2P_DISABLE=0
#     export NCCL_LL_THRESHOLD=16384
#     export NCCL_IB_CUDA_SUPPORT=1
# else
#     export NCCL_IB_GID_INDEX=3
#     export NCCL_IB_SL=3
#     export NCCL_CHECK_DISABLE=1
#     export NCCL_P2P_DISABLE=0
#     export NCCL_IB_DISABLE=0
#     export NCCL_LL_THRESHOLD=16384
#     export NCCL_IB_CUDA_SUPPORT=1
#     export NCCL_SOCKET_IFNAME=bond1
#     export UCX_NET_DEVICES=bond1
#     export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
#     export NCCL_COLLNET_ENABLE=0
#     export SHARP_COLL_ENABLE_SAT=0
#     export NCCL_NET_GDR_LEVEL=2
#     export NCCL_IB_QPS_PER_CONNECTION=4
#     export NCCL_IB_TC=160
#     export NCCL_PXN_DISABLE=1
# fi

# DEBUG=false
# env config
#---------------------------------------------------------------------------------
# MASTER_ADDR=$CHIEF_IP
# MASTER_PORT=6000
# NUM_GPUS=$NODE_NUM
NUM_GPUS=2
# TMP_DIR=${WORKSPACE}/tmp

# mkdir -p $TMP_DIR
# echo $NODE_IP_LIST > ${TMP_DIR}/env.txt
# # generate hostfile and pssh.hosts
# sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
# sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts
MODEL_PATH=/mnt/nlp/gaoqiang/training/output/01-08_retrieval_llama2-7b-Instruct__bs__maxlen_2048_pad_right_lr_2e-6_format_llama2/checkpoint-449
DATA_PATH=/mnt/nlp/gaoqiang/training/test_process.json
OUTPUT_DIR=/mnt/nlp/gaoqiang/training/eval
DATASET=CLES
CURRENT_TIME=$(date +'%m-%d_%T')
LOGS_PATH=/mnt/nlp/gaoqiang/training/logs

deepspeed --include localhost:0,3 inference.py \
    --model ${MODEL_PATH} \
    --batch_size 1 \
    --data ${DATA_PATH}\
    --output_path ${OUTPUT_DIR}/${DATASET}_outputs.json \
    --output_dir ${OUTPUT_DIR} \
    --precision bf16 \
    --max_input_length 2048 \
    --max_output_length 512 \
    --temperature 0.0 \
    --top_p 0.0 \
    --repetition_penalty 1.0
 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt