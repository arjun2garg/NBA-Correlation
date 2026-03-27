"""
Track A: Attention-based player decoder.

Self-attention over all 16 player slots so each player's prediction
directly interacts with every other player's representation. z modulates
the shared player context via a learned projection, injected as an additive
bias before the attention block.
"""

import torch
import torch.nn as nn


class PlayerDecoderAttention(nn.Module):
    def __init__(self, latent_dim, player_dim, h_dim=64, n_heads=2, output_dim=3, dropout=0.3):
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, h_dim)
        self.p_proj = nn.Linear(player_dim, h_dim)
        self.attn = nn.MultiheadAttention(h_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(h_dim)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, z, player_feats, mask=None):
        """
        z:           (batch, latent_dim)
        player_feats:(batch, n_players, player_dim)
        mask:        (batch, n_players) — 1 for valid players, 0 for padded

        Returns mu_pred, logvar_pred: (batch, n_players, output_dim)
        """
        # Project z and player features into shared h_dim space
        z_exp = self.z_proj(z).unsqueeze(1).expand(-1, player_feats.size(1), -1)
        h = self.p_proj(player_feats) + z_exp  # z biases each player token

        # Self-attention: padded slots are ignored
        if mask is not None:
            key_pad = ~mask.bool()  # True = ignore this key
        else:
            key_pad = None

        h_attn, _ = self.attn(h, h, h, key_padding_mask=key_pad)
        h = self.norm(h + h_attn)          # residual connection + layer norm
        h = self.mlp(h)

        return self.mu_head(h), self.logvar_head(h).clamp(-6.0, 2.0)
