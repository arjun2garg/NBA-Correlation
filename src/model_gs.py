"""
Game State Anchored VAE (Phase 2).

Key addition: a GameStateHead that maps z → G_pred (predicted game state).
Auxiliary loss: MSE(G_pred, actual_G) forces z to encode game state information.

Architecture:
  GameEncoder:   X_team → (mu_z, logvar_z)            [unchanged]
  PlayerDecoder: (z, X_players) → (mu_pred, logvar_pred)  [unchanged]
  GameStateHead: z → G_pred                             [NEW]

The GameStateHead is a lightweight linear projection so that z is the
bottleneck — the decoder must use z to predict both player outcomes
and game state. This prevents z from collapsing.
"""

import torch
import torch.nn as nn


class GameEncoder(nn.Module):
    """Game-level encoder: X_team → (mu_z, logvar_z)."""

    def __init__(self, input_dim, h_dim=128, latent_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(h_dim, latent_dim)
        self.logvar = nn.Linear(h_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class GameStateHead(nn.Module):
    """
    Lightweight head: z → G_pred (predicted game state vector).

    A single linear layer is intentional — if we add nonlinearity here,
    the head can approximate G independently of z's semantic meaning.
    A linear projection forces z to linearly encode G information.
    """

    def __init__(self, latent_dim, g_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, g_dim)

    def forward(self, z):
        return self.proj(z)


class PlayerDecoder(nn.Module):
    """Player-level decoder: (z, X_players) → (mu_pred, logvar_pred) per player per stat."""

    def __init__(self, latent_dim, player_dim, h_dim=64, output_dim=3, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + player_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, z, player_feats):
        # z: (batch, latent_dim) → broadcast over players
        z_exp = z.unsqueeze(1).expand(-1, player_feats.size(1), -1)
        h = self.trunk(torch.cat([z_exp, player_feats], dim=-1))
        return self.mu_head(h), self.logvar_head(h).clamp(-6.0, 2.0)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)
