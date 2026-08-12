"""Tiny continuous numeric Transformer classifier.

The model treats each numeric coordinate as one token-like element:
x1 -> linear projection -> embedding, x2 -> linear projection -> embedding.
Learned positional embeddings tell the Transformer which coordinate is which.
Output: binary class logit (class 0 / class 1) after mean pooling.

This module is deliberately isolated from the training loop so that Phase 2
(HRM controller) can wrap it without rewriting the experiment framework.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TinyNumericTransformer(nn.Module):
    """A very small Transformer encoder classifier over numeric sequences."""

    def __init__(
        self,
        input_dim: int = 2,
        seq_len: int = 2,
        embedding_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        # Per-coordinate numeric projection: each scalar becomes a vector.
        self.input_proj = nn.Linear(1, embedding_dim)

        # Learned positional embedding so the model knows which coordinate
        # occupies which sequence position.
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head: mean-pooled representation -> binary logit.
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len) numeric input in [-1, 1].
        Returns:
            (batch,) binary logits.
        """
        # (B, S, 1) -> linear -> (B, S, D)
        tokens = self.input_proj(x.unsqueeze(-1))

        positions = torch.arange(self.seq_len, device=x.device)
        tokens = tokens + self.pos_embedding(positions)

        encoded = self.encoder(tokens)

        # Mean pooling over the sequence dimension.
        pooled = encoded.mean(dim=1)  # (B, D)

        return self.head(pooled).squeeze(-1)  # (B,)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)