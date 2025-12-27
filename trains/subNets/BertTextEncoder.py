import torch
import torch.nn as nn
from transformers import BertModel

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=True, transformers='bert', pretrained='bert-base-uncased'):
        super(BertTextEncoder, self).__init__()
        self.use_finetune = use_finetune
        self.model = BertModel.from_pretrained(pretrained)
        
        # Freeze parameters if not fine-tuning
        if not use_finetune:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, text):
        """
        text: [batch_size, sequence_length, bert_dim] if pre-encoded
              or [batch_size, sequence_length] if token ids
        """
        if len(text.shape) == 3:
            # Already encoded BERT features
            return text
        
        # Token ids input
        with torch.set_grad_enabled(self.use_finetune):
            outputs = self.model(input_ids=text)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
        return last_hidden_state