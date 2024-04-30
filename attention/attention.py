import torch
import torch.nn as nn
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

    # Attention


query = torch.rand(128, 32, 1, 256)
key = value = torch.rand(128, 16, 1, 256)
query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
multihead_attn = ScaledDotProductAttention(temperature=query.size(2))
attn_output, attn_weights = multihead_attn(query, key, value)
attn_output = attn_output.transpose(1, 2)
print(f'attn_output: {attn_output.size()}, attn_weights: {attn_weights.size()}')

# Self-attention
query = torch.rand(128, 32, 1, 256)
query = query.transpose(1, 2)
multihead_attn = ScaledDotProductAttention(temperature=query.size(2))
attn_output, attn_weights = multihead_attn(query, query, query)
attn_output = attn_output.transpose(1, 2)
print(f'attn_output: {attn_output.size()}, attn_weights: {attn_weights.size()}')


torch.nn.MultiheadAttention