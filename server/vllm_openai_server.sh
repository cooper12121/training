#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,2


source ./.env
# 定义变量
HOST="0.0.0.0"
PORT=8000
TP_SIZE=2
MAX_LEN=2048


CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_FILE="../logs/vllm_openai_server_$CURRENT_TIME.log"
exec > "$LOG_FILE" 2>&1

echo "Starting vLLM server with the following configuration:"
echo "MODEL: $MODEL_PATH"
echo "HOST: $HOST"
echo "PORT: $PORT"
echo "TP_SIZE: $TP_SIZE"
echo "MAX_LEN: $MAX_LEN"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

vllm serve \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --enable-reasoning True \
  --reasoning-parser deepseek_r1 
  # --tensor-parallel-size "$TP_SIZE" \