import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']
logger = logging.getLogger('DecAlign')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'iemocap': self.__init_iemocap,
        }
        DATASET_MAP[args.dataset_name]()

    def __init_mosi(self):
        with open(self.args.featurePath, 'rb') as f:
            data = pickle.load(f)

        if getattr(self.args, 'use_bert', False):
            text_bert = data[self.mode]['text_bert']
            # MMSA format: [N, 3, seq_len] int64 (input_ids, type_ids, attention_mask)
            # Keep as int64 so BertTextEncoder can detect and process it correctly
            if text_bert.ndim == 3 and text_bert.shape[1] == 3 and np.issubdtype(text_bert.dtype, np.integer):
                self.text = text_bert  # keep int64 for BERT tokenizer input
            else:
                self.text = text_bert.astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # Load custom features if specified
        if getattr(self.args, 'feature_T', '') != "":
            with open(self.args.feature_T, 'rb') as f:
                data_T = pickle.load(f)
            if getattr(self.args, 'use_bert', False):
                text_bert_T = data_T[self.mode]['text_bert']
                if text_bert_T.ndim == 3 and text_bert_T.shape[1] == 3 and np.issubdtype(text_bert_T.dtype, np.integer):
                    self.text = text_bert_T
                else:
                    self.text = text_bert_T.astype(np.float32)
                self.args.feature_dims[0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args.feature_dims[0] = self.text.shape[2]

        if getattr(self.args, 'feature_A', '') != "":
            with open(self.args.feature_A, 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args.feature_dims[1] = self.audio.shape[2]

        if getattr(self.args, 'feature_V', '') != "":
            with open(self.args.feature_V, 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args.feature_dims[2] = self.vision.shape[2]

        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not getattr(self.args, 'need_data_aligned', False):
            if getattr(self.args, 'feature_A', '') != "":
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            elif 'audio_lengths' in data[self.mode]:
                self.audio_lengths = data[self.mode]['audio_lengths']
            else:
                # Aligned data has no audio_lengths; use full sequence length
                self.audio_lengths = [self.audio.shape[1]] * len(self.audio)
            if getattr(self.args, 'feature_V', '') != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            elif 'vision_lengths' in data[self.mode]:
                self.vision_lengths = data[self.mode]['vision_lengths']
            else:
                # Aligned data has no vision_lengths; use full sequence length
                self.vision_lengths = [self.vision.shape[1]] * len(self.vision)

        # Clean up inf values
        self.audio[self.audio == -np.inf] = 0

        if getattr(self.args, 'need_normalized', False):
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_iemocap(self):
        with open(self.args.featurePath, 'rb') as f:
            data = pickle.load(f)

        if getattr(self.args, 'use_bert', False):
            text_bert = data[self.mode]['text_bert']
            if text_bert.ndim == 3 and text_bert.shape[1] == 3 and np.issubdtype(text_bert.dtype, np.integer):
                self.text = text_bert  # keep int64 for BERT tokenizer input
            else:
                self.text = text_bert.astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # For IEMOCAP, labels are emotion categories
        self.labels = {
            'M': np.array(data[self.mode]['classification_labels']).astype(np.float32)
        }

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # Clean up inf values
        self.audio[self.audio == -np.inf] = 0

    def __normalize(self):
        """Normalize features"""
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        """Get sequence lengths for each modality"""
        if getattr(self.args, 'use_bert', False):
            # MMSA format text_bert: [N, 3, seq_len] → seq_len = shape[2]
            # Pre-encoded text_bert: [N, seq_len, dim] → seq_len = shape[1]
            if self.text.ndim == 3 and self.text.shape[1] == 3 and np.issubdtype(self.text.dtype, np.integer):
                text_seq_len = self.text.shape[2]
            else:
                text_seq_len = self.text.shape[1]
        else:
            text_seq_len = self.text.shape[1]
        return (text_seq_len, self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        """Get feature dimensions for each modality"""
        if getattr(self.args, 'use_bert', False):
            # MMSA format text_bert: [N, 3, seq_len] → BERT output will be 768-d
            if self.text.ndim == 3 and self.text.shape[1] == 3 and np.issubdtype(self.text.dtype, np.integer):
                text_feat_dim = 768
            else:
                text_feat_dim = self.text.shape[2]
        else:
            text_feat_dim = self.text.shape[2]
        return text_feat_dim, self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        # For MMSA format text_bert (int64), use LongTensor to preserve integer type
        if np.issubdtype(self.text.dtype, np.integer):
            text_tensor = torch.LongTensor(self.text[index])
        else:
            text_tensor = torch.Tensor(self.text[index])
        sample = {
            'raw_text': self.raw_text[index],
            'text': text_tensor,
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }

        if not getattr(self.args, 'need_data_aligned', False):
            sample['audio_lengths'] = self.audio_lengths[index] if hasattr(self, 'audio_lengths') else self.audio.shape[1]
            sample['vision_lengths'] = self.vision_lengths[index] if hasattr(self, 'vision_lengths') else self.vision.shape[1]

        return sample

def MMDataLoader(args, num_workers=4):
    """
    Create data loaders for train, valid, and test sets
    """
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    # Update sequence lengths in args
    if hasattr(args, 'seq_lens'):
        args.seq_lens = datasets['train'].get_seq_len()

    # Update feature dimensions in args
    feature_dims = datasets['train'].get_feature_dim()
    args.feature_dims = list(feature_dims)
    logger.info(f"Feature dimensions: Text={feature_dims[0]}, Audio={feature_dims[1]}, Vision={feature_dims[2]}")

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=num_workers,
                       shuffle=(ds == 'train'),
                       drop_last=False)
        for ds in datasets.keys()
    }

    return dataLoader