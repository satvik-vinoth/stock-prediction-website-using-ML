import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class TransformerConfig:
    def __init__(self,
                 hidden_size=64,
                 num_hidden_layers=2,
                 num_attention_heads=4,
                 intermediate_size=128,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=100):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings

# Numeric Embedding: projects raw numeric input to hidden space and adds learned positional embeddings.
class NumericEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, max_seq_length, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        x = self.linear(x)  # [batch_size, seq_length, hidden_size]
        batch_size, seq_length, _ = x.size()
        position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_emb = self.position_embeddings(position_ids)
        x = x + pos_emb
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_length, seq_length]
    scores = scores / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    output = torch.bmm(weights, value)  # [batch_size, seq_length, head_dim]
    return output

# Attention Head
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        query = self.q(hidden_state)
        key = self.k(hidden_state)
        value = self.v(hidden_state)
        attn_output = scaled_dot_product_attention(query, key, value)
        return attn_output

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        head_outputs = [head(hidden_state) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.output_linear(concatenated)
        return output

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.layer_norm_1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))
        return x

# Simple Transformer Encoder (stack of encoder layers)
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Complete Transformer-based Stock Predictor
class TransformerPredictor(nn.Module):
    def __init__(self, config, input_dim, seq_length):
        super().__init__()
        self.embedding = NumericEmbedding(input_dim=input_dim,
                                          hidden_size=config.hidden_size,
                                          max_seq_length=seq_length,
                                          dropout=config.hidden_dropout_prob)
        self.encoder = SimpleTransformerEncoder(config)
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)    # [batch_size, seq_length, hidden_size]
        x = self.encoder(x)      # [batch_size, seq_length, hidden_size]
        last_token = x[:, -1, :] # Use the last time step's representation
        out = self.fc(last_token)
        return out
