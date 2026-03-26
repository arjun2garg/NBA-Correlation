import torch
import torch.nn as nn


class PlayerDecoderFiLM(nn.Module):
    """
    Player decoder with FiLM (Feature-wise Linear Modulation) conditioning on z.

    Instead of concatenating z with player features, z produces per-dimension
    scale (gamma) and bias (beta) that modulate the intermediate representation.
    This gives z a multiplicative pathway that is easier to learn than pure
    concatenation, since the gradient flows through the modulation gate directly.

    Architecture:
      player_feats → Linear → ReLU → h  (base feature processing)
      z → Linear → (gamma, beta) of same dim as h
      h_mod = h * (1 + gamma) + beta    (residual FiLM: gamma initialised ~0 so identity at start)
      h_mod → Linear → ReLU → Dropout  (second hidden layer)
      → mu_head + logvar_head
    """

    def __init__(self, latent_dim, player_dim, h_dim=64, output_dim=3, dropout=0.3):
        super().__init__()
        self.h_dim = h_dim

        # Base player feature network (first hidden layer)
        self.player_net = nn.Sequential(
            nn.Linear(player_dim, h_dim),
            nn.ReLU(),
        )

        # FiLM generator: z → (gamma, beta) each of dim h_dim
        # Initialise small so it starts close to identity modulation
        self.film_gen = nn.Linear(latent_dim, 2 * h_dim)
        nn.init.normal_(self.film_gen.weight, std=0.01)
        nn.init.zeros_(self.film_gen.bias)

        # Second hidden layer after FiLM modulation
        self.post_film = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mu_head     = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, z, player_feats):
        # z: (batch, latent_dim)
        # player_feats: (batch, n_players, player_dim)
        batch_size, n_players, _ = player_feats.shape

        # Base player representation: (batch, n_players, h_dim)
        h = self.player_net(player_feats)

        # FiLM modulation from z
        # z: (batch, latent_dim) → (batch, 1, 2*h_dim) → broadcast over players
        film_params = self.film_gen(z).unsqueeze(1)       # (batch, 1, 2*h_dim)
        gamma = film_params[:, :, :self.h_dim]             # (batch, 1, h_dim)
        beta  = film_params[:, :, self.h_dim:]             # (batch, 1, h_dim)

        # Residual FiLM: identity at init (gamma≈0), learns to modulate
        h_mod = h * (1.0 + gamma) + beta                   # (batch, n_players, h_dim)

        # Second hidden layer
        h_out = self.post_film(h_mod)                      # (batch, n_players, h_dim)

        mu_pred     = self.mu_head(h_out)
        logvar_pred = self.logvar_head(h_out).clamp(-6.0, 2.0)
        return mu_pred, logvar_pred
