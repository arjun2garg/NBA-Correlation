import torch
import torch.nn as nn


class GameEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=48, latent_dim=16, dropout=0.3):
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


class PlayerDecoder(nn.Module):
    def __init__(self, latent_dim, player_dim, h_dim=24, output_dim=3, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + player_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head     = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, z, player_feats):
        # z: (batch, latent_dim) → broadcast over players
        z_exp = z.unsqueeze(1).expand(-1, player_feats.size(1), -1)
        h = self.trunk(torch.cat([z_exp, player_feats], dim=-1))
        return self.mu_head(h), self.logvar_head(h).clamp(-6.0, 2.0)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)
