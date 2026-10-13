#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/qwen3_5_config.sh"
PROJECT_ROOT="$QWEN_PROJECT_ROOT"
cd "$PROJECT_ROOT"
case "$TASK" in
    mmlu) data_prefix=mmlu-chat ;;
    bbh) data_prefix=bbh-icl ;;
    tydiqa) data_prefix=tydiqa ;;
    *) printf 'Unknown TASK: %s\n' "$TASK" >&2; exit 2 ;;
esac
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/selected_data/${data_prefix}_adam_sim_trainp0.05_seed${SELECTION_SEED}_p0.05.jsonl}"
JOB_NAME="${JOB_NAME:-qwen3.5-4b-${TASK}-transfer-s${SELECTION_SEED}-t${TRAIN_SEED}-${TRAIN_NPROC}gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/out/$JOB_NAME}"
launcher=("$PYTHON_BIN")
if (( TRAIN_NPROC > 1 )); then
    launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$TRAIN_NPROC")
fi

command=(
    "${launcher[@]}" -m less.train.train
    --model_name_or_path "$MODEL_PATH"
    --local_files_only True
    --torch_dtype bfloat16
    --attn_implementation sdpa
    --use_fast_tokenizer True
    --train_files "$TRAIN_FILE"
    --chat_format qwen
    --percentage "${TRAIN_PERCENTAGE:-1.0}"
    --sample_data_seed "$SELECTION_SEED"
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS"
    --max_seq_length "${MAX_SEQ_LENGTH:-1024}"
    --output_dir "$OUTPUT_DIR"
    --do_train True
    --eval_strategy no
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
    --max_steps "${MAX_STEPS:--1}"
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --ddp_find_unused_parameters False
    --ddp_broadcast_buffers False
    --learning_rate "${LEARNING_RATE:-1e-5}"
    --lr_scheduler_type linear
    --warmup_steps 0.03
    --weight_decay 0.0
    --optim adamw_torch
    --bf16 True
    --fp16 False
    --tf32 False
    --lora True
    --lora_r "${LORA_R:-16}"
    --lora_alpha "${LORA_ALPHA:-64}"
    --lora_dropout 0.1
    --lora_target_modules q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj
    --save_strategy epoch
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
    --export_optimizer_state "${EXPORT_OPTIMIZER_STATE:-False}"
    --logging_steps 1
    --report_to none
    --seed "$TRAIN_SEED"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --dataloader_pin_memory True
)

if [[ "${1:-}" == "--dry-run" && $# == 1 ]]; then
    printf 'Project: %s\nCUDA_VISIBLE_DEVICES=%s\n' "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if [[ $# != 0 ]]; then
    printf 'Usage: bash %s [--dry-run]\nOverride settings with MODEL_PATH, TRAIN_FILE, OUTPUT_DIR, MAX_STEPS, etc.\n' "$0" >&2
    exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/tokenizer_config.json" ]]; then
    printf 'Model config/tokenizer not found: %s\n' "$MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
    printf 'Training data not found: %s\n' "$TRAIN_FILE" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" && ( ! -d "$OUTPUT_DIR" || -n "$(ls -A "$OUTPUT_DIR")" ) ]]; then
    printf 'Output is not an empty directory: %s\nSet a new JOB_NAME or OUTPUT_DIR to avoid overwriting a run.\n' "$OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
