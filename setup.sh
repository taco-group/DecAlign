#!/bin/bash
# Setup script for ProtoAlign project
# This script creates a conda environment and installs dependencies

set -e

ENV_NAME=${1:-protoalign}
PYTHON_VERSION=${2:-3.9}

echo "Creating conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}"

# Create conda environment
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo "Installing ProtoAlign in editable mode..."
pip install -e .

echo "Setup completed!"
echo "Activate the environment with: conda activate ${ENV_NAME}"
