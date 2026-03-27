"""
Training with an explicit Mutual Information (MI) term.

The MI term penalizes low variance of mu_pred across z samples.
We want the decoder's mu_pred to VARY with z (sensitivity).
Adding -lambda_mi * var_across_z(mu_pred) encourages this directly.

This is distinct from the KL term: KL pushes encoder posterior closer to prior,
but says nothing about whether the decoder uses z. The MI term acts on the
decoder side, directly incentivizing z-sensitivity of outputs.
"""

import torch
from src.model import reparameterize


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


def mi_variance_term(encoder, decoder, X_t, X_p, weights, n_z_samples=8):
    """
    Compute -E_game[Var_z(mu_pred)] summed over valid players and stats.

    Higher variance = decoder is more z-sensitive = better.
    We negate it so minimizing the total loss maximizes this variance.

    Returns a scalar loss term (positive = penalizes low variance).
    """
    mu_enc, logvar_enc = encoder(X_t)
    mask = (weights > 0).float().unsqueeze(-1)  # (batch, 16, 1)

    # Sample n_z_samples z values and get mu_pred for each
    mu_preds = []
    for _ in range(n_z_samples):
        z = reparameterize(mu_enc, logvar_enc)
        mu_pred, _ = decoder(z, X_p)  # (batch, 16, 3)
        mu_preds.append(mu_pred)

    # Stack: (n_z_samples, batch, 16, 3)
    mu_stack = torch.stack(mu_preds, dim=0)

    # Variance across z samples: (batch, 16, 3)
    var_z = mu_stack.var(dim=0)

    # Mask out padded players
    var_masked = var_z * mask

    # Mean variance over valid players and stats
    denom = mask.sum() * var_z.size(-1) + 1e-8
    mean_var = var_masked.sum() / denom

    # Return negative variance (minimizing loss = maximizing variance)
    return -mean_var


def train_epoch_mi(encoder, decoder, optimizer, loader,
                   beta=0.001, free_bits=0.0, lambda_mi=0.1,
                   n_z_samples=8, device="cpu", grad_clip=0.3):
    encoder.train()
    decoder.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "mi": 0.0}

    for X_t, X_p, Y, weights, _ in loader:
        X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)

        optimizer.zero_grad()

        # Standard ELBO terms
        mu, logvar = encoder(X_t)
        z = reparameterize(mu, logvar)
        mu_pred, logvar_pred = decoder(z, X_p)

        recon = masked_nll(mu_pred, logvar_pred, Y, weights)
        kl = kl_divergence(mu, logvar, free_bits=free_bits)

        # MI term: encourage decoder to be z-sensitive
        mi_term = mi_variance_term(encoder, decoder, X_t, X_p, weights, n_z_samples)

        loss = recon + beta * kl + lambda_mi * mi_term
        loss.backward()

        # Gradient clipping prevents numerical explosion with large lambda_mi
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=grad_clip,
            )

        optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()
        totals["mi"] += mi_term.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def evaluate_mi(encoder, decoder, loader, beta=0.001, num_samples=1, device="cpu"):
    encoder.eval()
    decoder.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    with torch.no_grad():
        for X_t, X_p, Y, weights, _ in loader:
            X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
            mu, logvar = encoder(X_t)
            kl = kl_divergence(mu, logvar)
            recon = torch.stack([
                masked_nll(*decoder(reparameterize(mu, logvar), X_p), Y, weights)
                for _ in range(num_samples)
            ]).mean()
            loss = recon + beta * kl
            totals["loss"] += loss.item()
            totals["recon"] += recon.item()
            totals["kl"] += kl.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}
