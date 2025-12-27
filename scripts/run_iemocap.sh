#!/bin/bash
# Training script for IEMOCAP dataset
# Usage: ./run_iemocap.sh [gpu_id]

set -e
GPU_ID=${1:-0}
DATA_DIR=${DATA_DIR:-./data}

cd "$(dirname "$0")/../.."

echo "Training DecAlign on IEMOCAP dataset (GPU: ${GPU_ID})"

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --dataset iemocap \
    --data_dir ${DATA_DIR} \
    --mode train \
    --seeds 1111 1112 1113 1114 1115 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log
