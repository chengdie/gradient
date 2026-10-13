#!/usr/bin/env bash
set -euo pipefail

EXPLICIT_ACCUMULATION="${GRADIENT_ACCUMULATION_STEPS:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen3_5_config.sh"
cd "$QWEN_PROJECT_ROOT"
ACTION="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
extra_args=()
if [[ -n "$EXPLICIT_ACCUMULATION" ]]; then
    extra_args+=(--gradient-accumulation "$EXPLICIT_ACCUMULATION")
fi
if [[ -n "${EVAL_MAX_PER_SUBTASK:-}" ]]; then
    extra_args+=(--eval-max-per-subtask "$EVAL_MAX_PER_SUBTASK")
fi

exec "$PYTHON_BIN" -m less.experiments "$ACTION" \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$EXPERIMENT_OUTPUT_DIR" \
    --tasks "$TASK" \
    --selection-seeds "$SELECTION_SEED" \
    --training-seeds "$TRAIN_SEED" \
    --epochs "$NUM_TRAIN_EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --max-length "$MAX_SEQ_LENGTH" \
    --learning-rate "$LEARNING_RATE" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --train-nproc "$TRAIN_NPROC" \
    --eval-nproc "$EVAL_NPROC" \
    --gradient-nproc "$GRADIENT_NPROC" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --gradient-checkpointing "$GRADIENT_CHECKPOINTING" \
    --preprocessing-workers "$PREPROCESSING_NUM_WORKERS" \
    --dataloader-workers "$DATALOADER_NUM_WORKERS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --warmup-max-steps "$WARMUP_MAX_STEPS" \
    --warmup-percentage "$WARMUP_PERCENTAGE" \
    --gradient-max-length "$GRADIENT_MAX_LENGTH" \
    --projection-dim "$PROJECTION_DIM" \
    --eval-seed "$EVAL_SEED" \
    --eval-max-length "$EVAL_MAX_LENGTH" \
    --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    "${extra_args[@]}" \
    "$@"
