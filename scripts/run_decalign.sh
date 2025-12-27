#!/bin/bash
# Training script for DecAlign model
# Usage: ./run_decalign.sh [dataset] [gpu_id]
#   dataset: mosi, mosei, or iemocap (default: mosi)
#   gpu_id: GPU device ID (default: 0)

set -e

# Default values
DATASET=${1:-mosi}
GPU_ID=${2:-0}
DATA_DIR=${DATA_DIR:-./data}

echo "Training DecAlign on ${DATASET} dataset using GPU ${GPU_ID}"
echo "Data directory: ${DATA_DIR}"

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --mode train \
    --seeds 1111 1112 1113 1114 1115 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log

echo "Training completed!"
