"""
Simple GPT Model for Testing (without mup)
"""

import torch
import torch.nn as nn


class SimpleGPT2(nn.Module):
    """Simple GPT model without mup for testing."""

    def __init__(
        self,
        embedding_dim: int = 256,
        dim_heads: int = 128,
        nb_layers: int = 4,
        mlp_dim_mult: int = 4,
        vocabulary_size: int = 1000,
        nb_timesteps: int = 8,
        nb_tokens_per_timestep: int = 576,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.nb_timesteps = nb_timesteps

        # Embedding
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.position_embedding = nn.Embedding(nb_timesteps * nb_tokens_per_timestep, embedding_dim)

        # Layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=dim_heads // 32,  # head dim = 32
                dim_feedforward=embedding_dim * mlp_dim_mult,
                batch_first=True,
            )
            for _ in range(nb_layers)
        ])

        # Output
        self.lm_head = nn.Linear(embedding_dim, vocabulary_size)

        # Store init_std for use in _init_weights
        self._init_std = init_std

        # Init weights
        self.apply(self._init_weights)

    def forward(self, token_sequence, spatial_positions=None, temporal_positions=None):
        # token_sequence: (B, T*H*W) -> (B, T*H*W)
        B = token_sequence.shape[0]
        T = self.nb_timesteps
        H = W = int((token_sequence.shape[1] // T) ** 0.5) if token_sequence.shape[1] > 0 else 1

        # Token embeddings
        emb = self.token_embedding(token_sequence)  # (B, T*H*W, D)

        # Position embeddings
        if spatial_positions is None or temporal_positions is None:
            # Generate default position indices: use combined position embedding
            num_tokens = emb.shape[1]
            position_ids = torch.arange(num_tokens, device=emb.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.position_embedding(position_ids)
        else:
            # Combine spatial and temporal positions into a single position index
            # spatial_positions: (B, T*H*W), temporal_positions: (B, T*H*W)
            # Combined position: temporal * (H*W) + spatial
            max_spatial = H * W
            combined_positions = temporal_positions * max_spatial + spatial_positions
            pos_emb = self.position_embedding(combined_positions % self.position_embedding.num_embeddings)

        emb = emb + pos_emb

        # Transformer layers
        for layer in self.layers:
            emb = layer(emb)

        # Output logits
        logits = self.lm_head(emb)  # (B, T*H*W, V)

        # Reshape back to (B, V, T, H, W)
        logits = logits.view(B, self.vocabulary_size, T, H, W)

        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0, 0.02)


def load_pretrained_gpt(path):
    """Load pretrained model."""
    return torch.load(path, map_location="cpu")