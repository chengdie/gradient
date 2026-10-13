#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/less/scripts/qwen3_5_config.sh"
cd "$QWEN_PROJECT_ROOT"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-$MODEL_PATH}"
EVAL_NAME="${EVAL_NAME:-$(basename "$EVAL_MODEL_PATH")-${EVAL_NPROC}gpu}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EXPERIMENT_OUTPUT_DIR/standalone-eval/$EVAL_NAME/$TASK}"
launcher=("$PYTHON_BIN")
if (( EVAL_NPROC > 1 )); then
    launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$EVAL_NPROC")
fi

command=(
    "${launcher[@]}" -m evaluation.eval.qwen
    --model_path "$EVAL_MODEL_PATH"
    --base_model_path "$MODEL_PATH"
    --task "$TASK"
    --data_dir "$DATA_DIR"
    --output_dir "$EVAL_OUTPUT_DIR"
    --seed "$EVAL_SEED"
    --max_length "$EVAL_MAX_LENGTH"
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS"
    --batch_size "$EVAL_BATCH_SIZE"
    --device cuda:0
    --dtype bfloat16
)
if [[ -n "${EVAL_MAX_PER_SUBTASK:-}" ]]; then
    command+=(--max_per_subtask "$EVAL_MAX_PER_SUBTASK")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${command[@]}" "$@"
