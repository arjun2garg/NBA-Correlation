"""
Track B: Two-stage VAE.

Stage 1: z → GameOutcomeDecoder → predicted game outcome (6-dim)
Stage 2: concat(z, game_outcome, player_feats) → TwoStagePlayerDecoder → (mu, logvar)

Anchoring z to real game outcomes forces it to carry game-state signal.
Using raw targets (not residuals) preserves game-context correlation.
"""

import torch
import torch.nn as nn


class GameOutcomeDecoder(nn.Module):
    """z (batch, latent_dim) → game outcome prediction (batch, outcome_dim)."""

    def __init__(self, latent_dim, outcome_dim=6, h_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, outcome_dim),
        )

    def forward(self, z):
        return self.net(z)  # (batch, outcome_dim)


class TwoStagePlayerDecoder(nn.Module):
    """
    Wider PlayerDecoder that takes z + predicted game outcome + player features.

    input dim = latent_dim + outcome_dim + player_dim
    """

    def __init__(self, latent_dim, player_dim, outcome_dim=6, h_dim=64, output_dim=3, dropout=0.3):
        super().__init__()
        in_dim = latent_dim + outcome_dim + player_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, z, game_outcome, player_feats):
        """
        z:            (batch, latent_dim)
        game_outcome: (batch, outcome_dim) — predicted game outcome from GameOutcomeDecoder
        player_feats: (batch, n_players, player_dim)

        Returns mu_pred, logvar_pred: (batch, n_players, output_dim)
        """
        n_players = player_feats.size(1)
        z_exp = z.unsqueeze(1).expand(-1, n_players, -1)
        g_exp = game_outcome.unsqueeze(1).expand(-1, n_players, -1)
        h = self.trunk(torch.cat([z_exp, g_exp, player_feats], dim=-1))
        return self.mu_head(h), self.logvar_head(h).clamp(-6.0, 2.0)
