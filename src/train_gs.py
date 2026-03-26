"""
Training loop for game state anchored VAE (Phase 2).

Adds auxiliary loss: lambda_gs * MSE(G_pred, actual_G)
where G_pred = game_state_head(z).

This forces z to encode actual game state information, preventing
the decoder from ignoring z (the root cause of phi=0 in all previous experiments).
"""

import torch
from src.model_gs import reparameterize


def masked_nll(mu_pred, logvar_pred, target, weights):
    """Minutes-weighted Gaussian NLL."""
    nll = 0.5 * (logvar_pred + (target - mu_pred).pow(2) / logvar_pred.exp())
    nll = nll * weights.unsqueeze(-1)
    return nll.sum() / (weights.sum() * mu_pred.size(-1) + 1e-8)


def kl_divergence(mu, logvar, free_bits=0.0):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim.mean(dim=0), min=free_bits)
        return kl_per_dim.sum()
    return kl_per_dim.mean(dim=0).sum()


def gs_aux_loss(G_pred, G_actual, G_mask):
    """
    MSE between predicted and actual game state, masked to valid games.

    G_pred:   (batch, G_DIM)
    G_actual: (batch, G_DIM) — normalized
    G_mask:   (batch,) bool
    """
    if G_mask.sum() == 0:
        return torch.tensor(0.0, device=G_pred.device)
    diff = (G_pred[G_mask] - G_actual[G_mask]).pow(2)
    return diff.mean()


def train_epoch_gs(
    encoder, gs_head, decoder, optimizer, loader,
    beta=0.001, lambda_gs=1.0, free_bits=0.0, device="cpu"
):
    encoder.train()
    gs_head.train()
    decoder.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "gs": 0.0}

    for X_t, X_p, Y, weights, _, G, G_mask in loader:
        X_t = X_t.to(device)
        X_p = X_p.to(device)
        Y = Y.to(device)
        weights = weights.to(device)
        G = G.to(device)
        G_mask = G_mask.to(device)

        optimizer.zero_grad()
        mu, logvar = encoder(X_t)
        z = reparameterize(mu, logvar)

        # Player reconstruction loss
        mu_pred, logvar_pred = decoder(z, X_p)
        recon = masked_nll(mu_pred, logvar_pred, Y, weights)

        # KL divergence
        kl = kl_divergence(mu, logvar, free_bits=free_bits)

        # Game state auxiliary loss
        G_pred = gs_head(z)
        gs_loss = gs_aux_loss(G_pred, G, G_mask)

        loss = recon + beta * kl + lambda_gs * gs_loss
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(gs_head.parameters()) + list(decoder.parameters()),
            max_norm=5.0,
        )
        optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()
        totals["gs"] += gs_loss.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def evaluate_gs(
    encoder, gs_head, decoder, loader,
    beta=0.001, lambda_gs=1.0, num_samples=1, device="cpu"
):
    encoder.eval()
    gs_head.eval()
    decoder.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "gs": 0.0}

    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G, G_mask in loader:
            X_t = X_t.to(device)
            X_p = X_p.to(device)
            Y = Y.to(device)
            weights = weights.to(device)
            G = G.to(device)
            G_mask = G_mask.to(device)

            mu, logvar = encoder(X_t)
            kl = kl_divergence(mu, logvar)

            recon_samples = []
            gs_samples = []
            for _ in range(num_samples):
                z = reparameterize(mu, logvar)
                mu_pred, logvar_pred = decoder(z, X_p)
                recon_samples.append(masked_nll(mu_pred, logvar_pred, Y, weights))
                G_pred = gs_head(z)
                gs_samples.append(gs_aux_loss(G_pred, G, G_mask))

            recon = torch.stack(recon_samples).mean()
            gs_loss = torch.stack(gs_samples).mean()
            loss = recon + beta * kl + lambda_gs * gs_loss

            totals["loss"] += loss.item()
            totals["recon"] += recon.item()
            totals["kl"] += kl.item()
            totals["gs"] += gs_loss.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def compute_phi_sensitivity(encoder, gs_head, decoder, loader, n_z_samples=100, device="cpu"):
    """
    Measure P(over|z) std across z samples — the key diagnostic metric.

    Higher std → z meaningfully modulates player outcomes → phi > 0 possible.
    Target: std > 0.05 (current baseline: 0.02-0.04).

    Returns dict with:
      p_over_std_mean: mean P(over|z) std across all players in val set
      gs_r2: R² of G_pred vs actual G (how well z encodes game state)
    """
    import torch
    from scipy import stats as sp_stats
    import numpy as np

    encoder.eval()
    gs_head.eval()
    decoder.eval()

    all_p_over_stds = []
    gs_preds, gs_actuals = [], []

    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G, G_mask in loader:
            X_t = X_t.to(device)
            X_p = X_p.to(device)
            G = G.to(device)
            G_mask = G_mask.to(device)

            mu, logvar = encoder(X_t)

            # Sample n_z_samples z vectors per game
            # P(over|z) = Phi(mu_pred / sigma_pred)
            batch_size = mu.size(0)
            p_over_samples = []  # (n_z_samples, batch, n_players, n_stats)

            for _ in range(n_z_samples):
                z = reparameterize(mu, logvar)
                mu_pred, logvar_pred = decoder(z, X_p)
                sigma_pred = (0.5 * logvar_pred).exp()
                p_over = torch.distributions.Normal(0, 1).cdf(mu_pred / (sigma_pred + 1e-8))
                p_over_samples.append(p_over.cpu())

            p_over_stack = torch.stack(p_over_samples)  # (n_z, batch, players, stats)
            # Std across z samples per (player, stat)
            std_per_player = p_over_stack.std(dim=0)  # (batch, players, stats)
            active_mask = X_p.cpu().abs().sum(dim=-1) > 0  # non-padded players
            stds = std_per_player[active_mask.unsqueeze(-1).expand_as(std_per_player)]
            all_p_over_stds.extend(stds.tolist())

            # G prediction quality
            z_mu = mu  # use posterior mean for G prediction assessment
            G_pred = gs_head(z_mu).cpu().numpy()
            G_actual = G.cpu().numpy()
            mask = G_mask.cpu().numpy()
            if mask.sum() > 0:
                gs_preds.append(G_pred[mask])
                gs_actuals.append(G_actual[mask])

    p_over_std_mean = float(np.mean(all_p_over_stds)) if all_p_over_stds else 0.0

    gs_r2 = 0.0
    if gs_preds:
        gp = np.vstack(gs_preds)
        ga = np.vstack(gs_actuals)
        ss_res = ((gp - ga) ** 2).sum(axis=0)
        ss_tot = ((ga - ga.mean(axis=0)) ** 2).sum(axis=0)
        gs_r2 = float(1 - ss_res.mean() / (ss_tot.mean() + 1e-8))

    return {
        "p_over_std_mean": p_over_std_mean,
        "gs_r2": gs_r2,
    }
