# Mini Transformer from Scratch

A minimal implementation of a Transformer built from scratch in PyTorch to understand the core architecture behind modern Large Language Models (LLMs).

## Features
- Token & positional embeddings
- Single-head self-attention
- Multi-head self-attention
- Transformer block
- Feed-forward network
- Residual connections
- Layer normalization
- Output projection to vocabulary logits

## Project Structure

```
.
├── attention.py          # Self & Multi-Head Attention
├── transformer_block.py  # Transformer block
├── model.py              # Mini Transformer model
└── main.py               # Shape verification and testing
```

## Current Status
- ✅ Forward pass implemented
- ✅ Shape verification completed
- ⏳ Causal masking (next)
- ⏳ Multiple transformer blocks
- ⏳ Training on Tiny Shakespeare
- ⏳ Text generation

## Tech Stack
- Python
- PyTorch

## Goal
To build a GPT-style Transformer from first principles while understanding every component instead of relying on high-level libraries.
