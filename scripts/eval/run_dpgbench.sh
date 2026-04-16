#!/bin/bash

# ── User configuration ────────────────────────────────────────────────────
CKPT_PATH="./checkpoints/FiMR"
SAVE_PATH="./outputs"
EXP_NAME="fimr"
WORLD_SIZE=8
BATCH_SIZE=4
# ─────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../inference"

python dpgbench.py \
    --cfg_path ../configs \
    --overrides "ckpt_path=${CKPT_PATH}" \
    --overrides "save_path=${SAVE_PATH}" \
    --overrides "exp_name=${EXP_NAME}" \
    --overrides "world_size=${WORLD_SIZE}" \
    --overrides "batch_size=${BATCH_SIZE}"
