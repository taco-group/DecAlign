#!/bin/bash
# Data preparation script for multimodal sentiment analysis datasets
# Supports: MOSI, MOSEI, IEMOCAP

set -e

DATA_DIR=${1:-./data}

echo "============================================"
echo "  ProtoAlign Data Preparation Script"
echo "============================================"
echo ""
echo "This script helps you organize datasets for the ProtoAlign model."
echo "You need to manually download the following datasets:"
echo ""
echo "1. CMU-MOSI Dataset:"
echo "   - Source: http://immortal.multicomp.cs.cmu.edu/raw_datasets/"
echo "   - Paper: https://arxiv.org/abs/1606.06259"
echo "   - Place preprocessed file as: ${DATA_DIR}/MOSI/mosi_data.pkl"
echo ""
echo "2. CMU-MOSEI Dataset:"
echo "   - Source: http://immortal.multicomp.cs.cmu.edu/raw_datasets/"
echo "   - Paper: https://www.aclweb.org/anthology/P18-1208/"
echo "   - Place preprocessed file as: ${DATA_DIR}/MOSEI/mosei_data.pkl"
echo ""
echo "3. IEMOCAP Dataset:"
echo "   - Source: https://sail.usc.edu/iemocap/"
echo "   - Requires license agreement"
echo "   - Place preprocessed file as: ${DATA_DIR}/IEMOCAP/iemocap_data.pkl"
echo ""
echo "============================================"

# Create directory structure
echo "Creating directory structure..."
mkdir -p ${DATA_DIR}/MOSI
mkdir -p ${DATA_DIR}/MOSEI
mkdir -p ${DATA_DIR}/IEMOCAP

echo ""
echo "Directory structure created at: ${DATA_DIR}"
echo ""
echo "Expected file format (pickle files):"
echo "  Each .pkl file should contain a dictionary with keys:"
echo "    - 'train', 'valid', 'test'"
echo "  Each split should contain:"
echo "    - 'text' or 'text_bert': text features"
echo "    - 'audio': audio features"
echo "    - 'vision': video features"
echo "    - 'regression_labels' (MOSI/MOSEI) or 'classification_labels' (IEMOCAP)"
echo "    - 'raw_text': original text"
echo "    - 'id': sample identifiers"
echo ""
echo "For preprocessed datasets, check:"
echo "  - https://github.com/thuiar/MMSA (official preprocessing tools)"
echo "  - https://github.com/A2Zadeh/CMU-MultimodalSDK"
echo ""
echo "After placing the data files, run training with:"
echo "  python main.py --dataset mosi --data_dir ${DATA_DIR}"
echo ""
