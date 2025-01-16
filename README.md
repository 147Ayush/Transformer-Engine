
# Transformer Model Implementation

## Overview
This project implements a custom Transformer model inspired by the original "Attention is All You Need" paper. It includes the Encoder, Decoder, Multi-Head Attention, and other components necessary to perform sequence-to-sequence tasks, such as translation between English and Kannada.

The model is built using PyTorch and is designed to be modular, flexible, and extensible. It can be applied to various NLP tasks, including language translation, text summarization, and more.

---

## Features
1. **Scalable Transformer Architecture:**
   - Implements Encoder and Decoder stacks with multiple layers.
   - Includes Multi-Head Self-Attention and Cross-Attention mechanisms.

2. **Custom Components:**
   - Positional Encoding.
   - Sentence Embedding for tokenization and vocabulary management.
   - Layer Normalization for stable training.
   - Positionwise Feedforward Networks for added expressiveness.

3. **Tokenization Support:**
   - Handles vocabulary lookup, padding, and special tokens (START, END, and PADDING).

4. **Device Compatibility:**
   - Automatically detects and uses GPU (CUDA) if available.

---

## File Structure
- **`transformer.py`:** The main file containing all model components.
- **Model Components:**
  - `Encoder`
  - `Decoder`
  - `MultiHeadAttention`
  - `LayerNormalization`
  - `PositionwiseFeedForward`
- **Utilities:**
  - `scaled_dot_product`
  - `PositionalEncoding`
  - `SentenceEmbedding`

---

## Installation
To run this project, ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install torch numpy
```

---

## Usage
### 1. Model Initialization
```python
from transformer import Transformer

# Model Initialization
model = Transformer(
    d_model=d_model,
    ffn_hidden=ffn_hidden,
    num_heads=num_heads,
    drop_prob=drop_prob,
    num_layers=num_layers,
    max_sequence_length=max_sequence_length,
    kn_vocab_size=kn_vocab_size,
    english_to_index=english_to_index,
    kannada_to_index=kannada_to_index,
    START_TOKEN=START_TOKEN,
    END_TOKEN=END_TOKEN,
    PADDING_TOKEN=PADDING_TOKEN
)
```
## Key Functions and Classes

### `scaled_dot_product(q, k, v, mask=None)`
Performs scaled dot-product attention, computing the attention-weighted values.

### `PositionalEncoding`
Generates positional encodings for sequence embeddings, adding positional information to word embeddings.

### `SentenceEmbedding`
Handles tokenization, embedding lookup, and positional encoding for input sentences.

### `MultiHeadAttention`
Implements multi-head self-attention and cross-attention mechanisms.

### `LayerNormalization`
Applies layer normalization for stable and efficient training.

### `Encoder`
Processes input sequences using a stack of Encoder layers, combining self-attention and feedforward networks.

### `Decoder`
Processes target sequences using a stack of Decoder layers, combining self-attention, cross-attention, and feedforward networks.

### `Transformer`
Combines the Encoder and Decoder into a single Transformer model with linear projection for output logits.

---

## Example Workflow
1. Define English-to-Kannada vocabularies and special tokens.
2. Initialize the Transformer model with appropriate hyperparameters.
3. Tokenize input and target sentences.
4. Pass the tokenized data through the model.
5. Use the output logits for training or inference.

---

## Acknowledgments
This implementation is inspired by the paper *"Attention is All You Need"* by Vaswani et al. and aims to provide a clear and modular implementation for educational and research purposes.
