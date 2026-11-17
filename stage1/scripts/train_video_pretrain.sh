#!/bin/bash
# Launch Motus2 Stage 1 video pretraining with torchrun + DeepSpeed.
#
#   CONFIG=path/to/video_pretrain.yaml bash scripts/train_video_pretrain.sh
#
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT on every node.
set -euo pipefail

CONFIG="${CONFIG:?set CONFIG to a training config yaml}"
RUN_NAME="${RUN_NAME:-video_pretrain}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "$(dirname "$0")/.."
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train_video_pretrain.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    "$@"
