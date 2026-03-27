"""
Two-Stage Game State VAE.

Stage 1 — Supervised decoder (train first, independently):
  Input:  G (12-dim actual game state, normalized) + player_feats (24-dim)
  Output: (mu_pred, logvar_pred) per player per stat
  Loss:   Gaussian NLL (minutes-weighted)
  Result: decoder that IS sensitive to G by design — G has real signal,
          decoder must use it. sigma_pred reflects true residual after G.

Stage 2 — Probabilistic encoder (train after Stage 1 is frozen):
  Input:  X_team (256-dim pre-game features)
  Output: mu_G, logvar_G  (12-dim distribution over G)
  Loss:   NLL of actual G under predicted distribution + beta * KL(q || N(0,1))
  Result: encoder that outputs uncertain G predictions. Uncertainty = "what
          kind of game will this be?"

Simulation:
  sample G ~ N(mu_G, sigma_G)       # game uncertainty
  (mu_pred, logvar_pred) = decoder(G, player_feats)   # all 16 players get same G
  P(over|G) = Phi(mu_pred / sigma_pred)
  → shared G sample = correlation source across all players in the game
"""

import torch
import torch.nn as nn

G_DIM = 12   # 6 TS features × 2 teams (home + away)


class GCondDecoder(nn.Module):
    """
    Stage 1: G-conditioned player decoder.

    Takes ACTUAL game state G (not z) as game-level context.
    This guarantees the decoder uses G — no shortcut possible since
    G is the only game-level signal. Player features give player identity;
    G gives game context.
    """

    def __init__(self, g_dim=G_DIM, player_dim=24, h_dim=64, output_dim=3, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(g_dim + player_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, G, player_feats):
        """
        G:            (batch, g_dim) — normalized game state
        player_feats: (batch, max_players, player_dim)
        Returns:      mu_pred, logvar_pred  each (batch, max_players, output_dim)
        """
        G_exp = G.unsqueeze(1).expand(-1, player_feats.size(1), -1)
        h = self.trunk(torch.cat([G_exp, player_feats], dim=-1))
        return self.mu_head(h), self.logvar_head(h).clamp(-6.0, 2.0)


class GameEncoder(nn.Module):
    """
    Stage 2: Probabilistic encoder over game state G.

    X_team → (mu_G, logvar_G)  where mu_G ≈ actual G (normalized)

    The posterior N(mu_G, sigma_G) captures uncertainty about what game
    state will actually occur given pre-game context. Sampling from this
    posterior gives different game scenarios, driving correlated player
    outcome variation at simulation time.
    """

    def __init__(self, input_dim=256, h_dim=128, g_dim=G_DIM, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(h_dim, g_dim)
        self.logvar = nn.Linear(h_dim, g_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)
