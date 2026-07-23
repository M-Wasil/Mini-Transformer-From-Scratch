import torch
from attention import SelfAttention, MultiHeadAttention
from transformer_block import TransformerBlock
from model import MiniTransformer

torch.manual_seed(0)

print("=" * 50)
print("STEP 1: Single-head self-attention")
print("=" * 50)
x = torch.randn(2, 8, 32)  # (batch_size, seq_len, embed_dim)
attn = SelfAttention(embed_dim=32)
out = attn(x)
print("Input: ", x.shape)
print("Output:", out.shape)
assert out.shape == (2, 8, 32)
print("OK -- shape unchanged, each token's vector is now a blend of every other token's value vector.\n")

print("=" * 50)
print("STEP 2: Multi-head attention")
print("=" * 50)
mha = MultiHeadAttention(embed_dim=32, num_heads=4)
out = mha(x)
print("Input: ", x.shape)
print("Output:", out.shape)
assert out.shape == (2, 8, 32)
print("OK -- same output shape as single-head, but internally 4 heads of dim 8 each ran in parallel.\n")

print("=" * 50)
print("STEP 3: Full Transformer block (attention + FFN + residuals + norm)")
print("=" * 50)
block = TransformerBlock(embed_dim=32, num_heads=4)
out = block(x)
print("Input: ", x.shape)
print("Output:", out.shape)
assert out.shape == (2, 8, 32)
print("OK -- this is the exact structure you'd stack N times to build GPT.\n")

print("=" * 50)
print("STEP 4: Full mini model (embeddings -> block -> vocab logits)")
print("=" * 50)
vocab_size = 100
embed_dim = 32
num_heads = 4
max_len = 8

model = MiniTransformer(vocab_size, embed_dim, num_heads, max_len)
token_ids = torch.randint(0, vocab_size, (2, 8))  # (batch_size, seq_len)
logits = model(token_ids)
print("Input token ids:", token_ids.shape)
print("Output logits:  ", logits.shape)
assert logits.shape == (2, 8, vocab_size)
print("OK -- for every one of the 8 positions, we get a probability distribution over 100 vocab tokens.\n")

print("All checks passed. Every stage matches the expected shape.")
