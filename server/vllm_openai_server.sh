#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 读取环境变量 sh中不支持source,使用.
. ./.env

# 定义变量
# HOST="0.0.0.0"
# PORT=8000
TP_SIZE=1
PIPELINE_PARALLEL_SIZE=1

MAX_LEN=2048


CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")

LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="../logs/vllm_openai_server_$CURRENT_TIME.log"
exec > "$LOG_FILE" 2>&1

echo "Starting vLLM server with the following configuration:"
echo "MODEL: $MODEL_PATH"
echo "HOST: $HOST"
echo "PORT: $PORT"
echo "TP_SIZE: $TP_SIZE"
echo "MAX_LEN: $MAX_LEN"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

vllm serve $MODEL_PATH \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" 