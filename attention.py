import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head scaled dot-product attention.
    Input:  (batch_size, seq_len, embed_dim)
    Output: (batch_size, seq_len, embed_dim)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)   # (B, T, D)
        K = self.key(x)     # (B, T, D)
        V = self.value(x)   # (B, T, D)

        # scores[b, i, j] = how much token i should attend to token j
        scores = Q @ K.transpose(-2, -1) / (self.embed_dim ** 0.5)  # (B, T, T)
        weights = F.softmax(scores, dim=-1)                        # (B, T, T)
        out = weights @ V                                          # (B, T, D)
        return out


class MultiHeadAttention(nn.Module):
    """
    Splits embed_dim across num_heads, runs attention independently per head,
    then concatenates the results back together.

    Example: embed_dim=32, num_heads=4 -> each head works on 8 dims.
    Input:  (batch_size, seq_len, embed_dim)
    Output: (batch_size, seq_len, embed_dim)
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # One big linear layer for all heads at once (more efficient than looping)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Recombines the concatenated heads back into embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, B, T):
        # (B, T, D) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.split_heads(self.query(x), B, T)  # (B, heads, T, head_dim)
        K = self.split_heads(self.key(x), B, T)     # (B, heads, T, head_dim)
        V = self.split_heads(self.value(x), B, T)   # (B, heads, T, head_dim)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, heads, T, T)
        weights = F.softmax(scores, dim=-1)                       # (B, heads, T, T)
        out = weights @ V                                         # (B, heads, T, head_dim)

        # Merge heads back: (B, heads, T, head_dim) -> (B, T, heads, head_dim) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
