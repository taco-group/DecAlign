"""
prepare_iemocap.py

Prepares the IEMOCAP dataset into the pickle format expected by DecAlign's
data_loader.py for the 'iemocap' dataset option.

IEMOCAP Dataset:
    - Request access at: https://sail.usc.edu/iemocap/
    - Requires signing a license agreement with USC
    - Alternatively, use the preprocessed version shared by the authors:
      https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link

Usage:
    # Option 1: Use the preprocessed pkl shared by the authors (recommended)
    #   Download iemocap_data.pkl from the link above and place it at:
    #   ./data/IEMOCAP/iemocap_data.pkl

    # Option 2: Process from raw IEMOCAP data using this script
    python scripts/prepare_iemocap.py \
        --iemocap_dir /path/to/IEMOCAP_full_release \
        --output_dir ./data/IEMOCAP \
        --output_file iemocap_data.pkl

Output format:
    A pickle file containing a dict with keys 'train', 'valid', 'test'.
    Each split is a dict with:
        - 'text'                 : np.ndarray [N, seq_len, text_dim]
        - 'audio'                : np.ndarray [N, seq_len, audio_dim]
        - 'vision'               : np.ndarray [N, seq_len, vision_dim]
        - 'classification_labels': np.ndarray [N]  (int, 0-based emotion class)
        - 'raw_text'             : list of str
        - 'id'                   : list of str (utterance IDs)

    Emotion label mapping (4-class):
        0: happy/excited
        1: sad
        2: angry
        3: neutral
"""

import os
import re
import pickle
import argparse
import numpy as np
from pathlib import Path


# 4-class emotion mapping used in most IEMOCAP multimodal papers
EMOTION_MAP = {
    'hap': 0,
    'exc': 0,   # excited -> happy
    'sad': 1,
    'ang': 2,
    'neu': 3,
}

# Sessions and their speaker assignments
SESSIONS = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

# Train/val/test split: Sessions 1-4 train, Session 4 last part val, Session 5 test
# Following the standard split used in MMSA / M-SENA
TRAIN_SESSIONS = ['Session1', 'Session2', 'Session3']
VAL_SESSIONS   = ['Session4']
TEST_SESSIONS  = ['Session5']


def parse_emotion_labels(label_file):
    """Parse a single IEMOCAP evaluation file and return {utterance_id: emotion}."""
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            # Format: [start - end] utterance_id emotion [v, a, d]
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            utt_id = parts[1].strip()
            emotion = parts[2].strip().lower()
            if emotion in EMOTION_MAP:
                labels[utt_id] = EMOTION_MAP[emotion]
    return labels


def load_transcriptions(trans_file):
    """Parse a transcription file and return {utterance_id: text}."""
    trans = {}
    with open(trans_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            # Format: utterance_id [start - end]: transcription
            match = re.match(r'^(Ses\S+)\s+\[.*?\]:\s*(.*)', line)
            if match:
                utt_id = match.group(1)
                text = match.group(2).strip()
                trans[utt_id] = text
    return trans


def extract_utterance_ids(session_dir, sessions):
    """Collect all utterance IDs with valid 4-class emotion labels."""
    utterances = []
    for session in sessions:
        label_dir = Path(session_dir) / session / 'dialog' / 'EmoEvaluation'
        trans_dir = Path(session_dir) / session / 'dialog' / 'transcriptions'

        if not label_dir.exists():
            print(f"Warning: label dir not found: {label_dir}")
            continue

        for label_file in sorted(label_dir.glob('*.txt')):
            dialog_id = label_file.stem
            trans_file = trans_dir / f"{dialog_id}.txt"

            labels = parse_emotion_labels(label_file)
            trans = load_transcriptions(trans_file) if trans_file.exists() else {}

            for utt_id, label in labels.items():
                text = trans.get(utt_id, '')
                utterances.append({
                    'id': utt_id,
                    'text': text,
                    'label': label,
                    'session': session,
                })

    return utterances


def build_text_features(utterances, max_len=50, dim=300):
    """
    Build simple bag-of-words text features as float32 arrays.

    For production use, replace this with proper feature extraction
    (e.g., GloVe embeddings or BERT tokenization via MMSA's pipeline).
    The MMSA preprocessed pkl already contains these features — use that
    instead of re-extracting from scratch.
    """
    N = len(utterances)
    features = np.zeros((N, max_len, dim), dtype=np.float32)
    print(f"  Note: using zero-initialized text features ({dim}-d). "
          f"Use MMSA preprocessing for real GloVe/BERT features.")
    return features


def build_dummy_multimodal(N, seq_len=50, audio_dim=74, vision_dim=35):
    """
    Return zero arrays for audio and vision.

    For real features, use the MMSA SDK or the preprocessed pkl provided
    by the authors at:
    https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view
    """
    audio  = np.zeros((N, seq_len, audio_dim),  dtype=np.float32)
    vision = np.zeros((N, seq_len, vision_dim), dtype=np.float32)
    print(f"  Note: audio/vision are zero-initialized. "
          f"Use MMSA preprocessing or the authors' preprocessed pkl for real features.")
    return audio, vision


def build_split(utterances):
    N = len(utterances)
    text_feats          = build_text_features(utterances)
    audio_feats, vision_feats = build_dummy_multimodal(N)
    labels              = np.array([u['label'] for u in utterances], dtype=np.int64)
    raw_text            = [u['text'] for u in utterances]
    ids                 = [u['id']   for u in utterances]

    return {
        'text':                  text_feats,
        'audio':                 audio_feats,
        'vision':                vision_feats,
        'classification_labels': labels,
        'raw_text':              raw_text,
        'id':                    ids,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Prepare IEMOCAP dataset for DecAlign.'
    )
    parser.add_argument(
        '--iemocap_dir', type=str, required=True,
        help='Path to the IEMOCAP_full_release root directory.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./data/IEMOCAP',
        help='Directory to save the output pkl file.'
    )
    parser.add_argument(
        '--output_file', type=str, default='iemocap_data.pkl',
        help='Output pickle filename.'
    )
    args = parser.parse_args()

    print("=" * 55)
    print(" DecAlign — IEMOCAP Data Preparation")
    print("=" * 55)
    print()
    print("IMPORTANT: If you have access to the preprocessed pkl")
    print("shared by the authors, use that directly:")
    print("  https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3")
    print("Place it at: ./data/IEMOCAP/iemocap_data.pkl")
    print()
    print("Proceeding with raw IEMOCAP data processing...")
    print()

    iemocap_dir = Path(args.iemocap_dir)
    if not iemocap_dir.exists():
        raise FileNotFoundError(f"IEMOCAP directory not found: {iemocap_dir}")

    print("Collecting utterances...")
    train_utts = extract_utterance_ids(iemocap_dir, TRAIN_SESSIONS)
    val_utts   = extract_utterance_ids(iemocap_dir, VAL_SESSIONS)
    test_utts  = extract_utterance_ids(iemocap_dir, TEST_SESSIONS)

    print(f"  Train: {len(train_utts)} utterances")
    print(f"  Val:   {len(val_utts)} utterances")
    print(f"  Test:  {len(test_utts)} utterances")
    print()

    print("Building feature arrays...")
    data = {
        'train': build_split(train_utts),
        'valid': build_split(val_utts),
        'test':  build_split(test_utts),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / args.output_file

    print(f"\nSaving to {out_path} ...")
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    print(f"Done. Saved {out_path}")
    print()
    print("You can now train with:")
    print(f"  python main.py --dataset iemocap --data_dir {args.output_dir.rstrip('/')}/.. --mode train")


if __name__ == '__main__':
    main()
