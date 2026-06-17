# -*- coding: utf-8 -*-
"""
Sensory pipeline: entity embedder + LiDAR embedder + transformer encoder stack.

The LiDAR signal (30 rays + 2 time features) is injected as an additional
Transformer token so the attention mechanism can learn wall-awareness
(e.g. "entity X is behind a wall").
"""
import torch
import torch.nn as nn

from constants import ENTITY_FEAT_DIM, LIDAR_DIM


class EntityEmbedder(nn.Module):
    def __init__(self, input_dim: int = ENTITY_FEAT_DIM, hidden_dim: int = 64, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LidarEmbedder(nn.Module):
    """Projects the LiDAR/time observation vector into the Transformer's
    embedding space so it can participate in multi-head attention."""

    def __init__(self, lidar_dim: int = LIDAR_DIM, hidden_dim: int = 64, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.ln1       = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln2       = nn.LayerNorm(embed_dim)
        self.ffn       = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=padding_mask)
        x = x + attn_out
        x_norm2 = self.ln2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        return x


class SensoryPipeline(nn.Module):
    """
    Entity embeddings + LiDAR token → Transformer → mean-pooled context.

    Sequence layout:
        [entity_0, entity_1, ..., entity_{MAX-1}, lidar_token]

    The LiDAR token is always unmasked so the Transformer can attend to
    wall geometry when processing entity-relative positions.
    """

    def __init__(self, input_dim: int = ENTITY_FEAT_DIM, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, lidar_dim: int = LIDAR_DIM):
        super().__init__()
        self.embedder       = EntityEmbedder(input_dim=input_dim, embed_dim=embed_dim)
        self.lidar_embedder = LidarEmbedder(lidar_dim=lidar_dim, embed_dim=embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, entities: torch.Tensor, mask: torch.Tensor,
                lidar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entities: (B, MAX_ENTITIES, FEAT_DIM)
            mask:     (B, MAX_ENTITIES)  — 1.0 for valid, 0.0 for padding
            lidar:    (B, LIDAR_DIM)

        Returns:
            context:  (B, embed_dim)  — mean-pooled over valid tokens + lidar
        """
        B = entities.shape[0]

        # Embed entities and LiDAR into the same space
        x = self.embedder(entities)                              # (B, MAX_ENTITIES, D)
        lidar_token = self.lidar_embedder(lidar).unsqueeze(1)    # (B, 1, D)

        # Concatenate: entities + lidar token
        x = torch.cat([x, lidar_token], dim=1)                  # (B, MAX_ENTITIES+1, D)

        # Extend mask — LiDAR token is always valid
        lidar_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
        extended_mask = torch.cat([mask, lidar_mask], dim=1)     # (B, MAX_ENTITIES+1)

        # Transformer expects True = ignore, so invert
        padding_mask = (extended_mask == 0.0)
        for layer in self.layers:
            x = layer(x, padding_mask)

        # Mean-pooling over the entity dimension, ignoring padded elements.
        expanded_mask = extended_mask.unsqueeze(-1)
        sum_emb = (x * expanded_mask).sum(dim=1)
        valid_counts = expanded_mask.sum(dim=1).clamp(min=1.0)
        context = sum_emb / valid_counts

        return context
