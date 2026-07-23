import torch
import torch.nn as nn
from transformer_block import TransformerBlock


class MiniTransformer(nn.Module):
    """
    Token embedding + positional embedding -> TransformerBlock -> output projection to vocab.

    Input:  (batch_size, seq_len)                token ids, dtype long
    Output: (batch_size, seq_len, vocab_size)     logits over the vocabulary at each position
    """
    def __init__(self, vocab_size, embed_dim, num_heads, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.block = TransformerBlock(embed_dim, num_heads)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        # Token identity + token position, added together
        x = self.embedding(x) + self.position_embedding(positions)
        x = self.block(x)
        logits = self.fc_out(x)
        return logits
