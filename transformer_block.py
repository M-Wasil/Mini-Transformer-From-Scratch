import torch.nn as nn
from attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    One Transformer block:
    x -> MultiHeadAttention -> residual add -> LayerNorm
      -> FeedForward         -> residual add -> LayerNorm

    Input:  (batch_size, seq_len, embed_dim)
    Output: (batch_size, seq_len, embed_dim)  (shape never changes -- blocks are stackable)
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Residual connection: add the input back so gradients have a direct path
        # through the network, even in very deep stacks.
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
