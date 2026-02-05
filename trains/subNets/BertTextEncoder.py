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
        text: [batch_size, seq_len, bert_dim] if pre-encoded float features
              or [batch_size, 3, seq_len] if MMSA format (input_ids, type_ids, attention_mask)
              or [batch_size, seq_len] if plain token ids
        """
        if len(text.shape) == 3:
            # Check if this is MMSA format: [N, 3, seq_len] with integer token IDs
            if text.shape[1] == 3 and text.dtype in (torch.long, torch.int, torch.int64):
                input_ids = text[:, 0, :].long()       # [N, seq_len]
                token_type_ids = text[:, 1, :].long()   # [N, seq_len]
                attention_mask = text[:, 2, :].long()    # [N, seq_len]
                with torch.set_grad_enabled(self.use_finetune):
                    outputs = self.model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask
                    )
                    return outputs.last_hidden_state  # [N, seq_len, 768]
            else:
                # Already encoded float features [N, seq_len, bert_dim]
                return text

        # Plain token ids [N, seq_len]
        with torch.set_grad_enabled(self.use_finetune):
            outputs = self.model(input_ids=text)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        return last_hidden_state