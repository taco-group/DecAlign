import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0, attn_mask=False):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.attn_mask = attn_mask
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, attn_dropout, relu_dropout, res_dropout)
            for _ in range(layers)
        ])
        
    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Args:
            x_in (FloatTensor): embedded inputs of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded inputs of shape `(src_len, batch, embed_dim)` for key
            x_in_v (FloatTensor): embedded inputs of shape `(src_len, batch, embed_dim)` for value
        """
        x = self.embed_scale * x_in
        x = self.embed_dropout(x)
        
        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
        else:
            x_k = x_v = x
        
        # Create attention mask if needed
        attn_mask = None
        if self.attn_mask:
            src_len = x.size(0)
            attn_mask = torch.triu(torch.ones(src_len, src_len), diagonal=1).bool()
            attn_mask = attn_mask.to(x.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, x_k, x_v, attn_mask)
        
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(embed_dim, num_heads, attn_dropout=attn_dropout)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self.relu_dropout = nn.Dropout(relu_dropout)
        self.res_dropout = nn.Dropout(res_dropout)
        
    def forward(self, x, x_k=None, x_v=None, attn_mask=None):
        """
        Args:
            x: input to the layer of shape `(seq_len, batch, embed_dim)`
            x_k: key input shape `(seq_len, batch, embed_dim)`
            x_v: value input shape `(seq_len, batch, embed_dim)`
            attn_mask: attention mask of shape `(seq_len, seq_len)`
        """
        residual = x
        x, _ = self.self_attn(query=x, key=x_k if x_k is not None else x, 
                             value=x_v if x_v is not None else x, attn_mask=attn_mask)
        x = self.res_dropout(x)
        x = residual + x
        x = self.attn_layer_norm(x)
        
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.relu_dropout(x)
        x = self.fc2(x)
        x = self.res_dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)
        
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, bias=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
            
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
            
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: `(L, N, E)` where L is the target sequence length, N is the batch size, E is embedding dim
            key: `(S, N, E)`, where S is the source sequence length
            value: `(S, N, E)`
            attn_mask: `(L, S)` where L is target length, S is source length
        """
        return self._multi_head_attention_forward(query, key, value, attn_mask)
        
    def _multi_head_attention_forward(self, query, key, value, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        # Linear projections
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).expand(bsz * self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn = torch.bmm(attn_weights, v)
        
        # Reshape and project
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        
        return attn, attn_weights