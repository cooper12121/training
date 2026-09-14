#!/usr/bin/env bash
# Run inside a Merlin worker or Arnold container, once per node by default.
set -euo pipefail
WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# ==================== 配置区：日常只修改这里 ====================
# 填写任务容器内可访问的路径；环境变量仍可覆盖这些默认值。
MODEL_PATH="${MODEL_PATH:-}"                 # 例如 /mnt/models/Qwen2.5-7B-Instruct
DATA_PATH="${DATA_PATH:-}"                   # 例如 /mnt/data/sft_train.json
OUTPUT_DIR="${OUTPUT_DIR:-}"                 # 持久化目录；调试和正式训练分开

TRAIN_MODE="${TRAIN_MODE:-train}"            # debug / train；--debug 可临时切换
DEBUG_STEPS="${DEBUG_STEPS:-5}"
EPOCHS="${EPOCHS:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SEED="${SEED:-42}"
BF16="${BF16:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
REPORT_TO="${REPORT_TO:-none}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"      # 留空使用 DDP
LAUNCH_MODE="${LAUNCH_MODE:-node}"            # 平台按每 GPU 启动进程时改为 process
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
NNODES="${NNODES:-1}"
# 多节点的 NODE_RANK / MASTER_ADDR / MASTER_PORT 由任务环境提供。
# 不要将所有节点的 NODE_RANK 固定为同一个值。
EXTRA_TRAIN_ARGS=()                          # 例如 (--weight_decay 0.01)
# ==================== 配置区结束 ====================
MODE="$TRAIN_MODE"
if [[ "${1:-}" == "--debug" ]]; then
    MODE=debug
    shift
fi
if [[ "$MODE" != train && "$MODE" != debug ]]; then
    echo 'TRAIN_MODE must be train or debug' >&2
    exit 2
fi
: "${MODEL_PATH:?Set MODEL_PATH to a model visible inside the worker/job}"
: "${DATA_PATH:?Set DATA_PATH to training data visible inside the worker/job}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to persistent storage; separate debug/train runs}"
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
args=(
    --model_name_or_path "$MODEL_PATH" --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR" --do_train True --do_eval False
    --num_train_epochs "$EPOCHS" --model_max_length "$MAX_LENGTH"
    --per_device_train_batch_size "$MICRO_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE" --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" --logging_steps "$LOGGING_STEPS" --report_to "$REPORT_TO"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" --ddp_find_unused_parameters False --bf16 "$BF16"
    --eval_strategy no --save_strategy epoch --save_total_limit "$SAVE_TOTAL_LIMIT" --seed "$SEED"
    --stage sft --use_lora False
)
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
    args+=(--deepspeed "$DEEPSPEED_CONFIG")
fi
args+=("${EXTRA_TRAIN_ARGS[@]}" "$@")
if [[ "$MODE" == debug ]]; then
    if [[ ! "$DEBUG_STEPS" =~ ^[1-9][0-9]*$ ]]; then
        echo 'DEBUG_STEPS must be a positive integer' >&2
        exit 2
    fi
    args+=(--overwrite_output_dir True --max_steps "$DEBUG_STEPS" --save_strategy no --skip_final_save True)
fi
case "$LAUNCH_MODE" in
    node)
        # Preserve the scheduler's CUDA_VISIBLE_DEVICES allocation.
        launch=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NPROC_PER_NODE")
        if [[ "$NNODES" == 1 ]]; then
            launch+=(--standalone --nnodes 1)
        else
            : "${NODE_RANK:?Set NODE_RANK for multi-node training}"
            : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
            : "${MASTER_PORT:?Set MASTER_PORT for multi-node training}"
            launch+=(--nnodes "$NNODES" --node_rank "$NODE_RANK"
                     --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
        fi
        ;;
    process)
        # Use only when the platform already starts one process per GPU.
        launch=("$PYTHON_BIN")
        ;;
    *) echo 'LAUNCH_MODE must be node or process' >&2; exit 2 ;;
esac
exec "${launch[@]}" "$WORKSPACE/train/train.py" "${args[@]}"
