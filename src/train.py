import torch
from src.model import reparameterize


def masked_mse(pred, target, weights):
    """Weighted MSE. weights: (batch, players) — h_numMinutes, 0 for padded slots."""
    diff = (pred - target) ** 2
    diff = diff * weights.unsqueeze(-1)
    return diff.sum() / (weights.sum() * pred.size(-1) + 1e-8)


def kl_divergence(mu, logvar, free_bits=0.0):
    # per-dimension KL: (batch, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        # clamp per-dim mean — don't penalize dimensions already below threshold
        kl_per_dim = torch.clamp(kl_per_dim.mean(dim=0), min=free_bits)
        return kl_per_dim.sum()
    return kl_per_dim.mean(dim=0).sum()


def train_epoch(encoder, decoder, optimizer, loader, beta=0.001, free_bits=0.0, device="cpu"):
    encoder.train()
    decoder.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    for X_t, X_p, Y, weights, _ in loader:
        X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)

        optimizer.zero_grad()
        mu, logvar = encoder(X_t)
        z = reparameterize(mu, logvar)
        preds = decoder(z, X_p)

        recon = masked_mse(preds, Y, weights)
        kl = kl_divergence(mu, logvar, free_bits=free_bits)
        loss = recon + beta * kl
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def evaluate(encoder, decoder, loader, beta=0.001, num_samples=1, device="cpu"):
    encoder.eval()
    decoder.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    with torch.no_grad():
        for X_t, X_p, Y, weights, _ in loader:
            X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
            mu, logvar = encoder(X_t)
            kl = kl_divergence(mu, logvar)
            recon = torch.stack([
                masked_mse(decoder(reparameterize(mu, logvar), X_p), Y, weights)
                for _ in range(num_samples)
            ]).mean()
            loss = recon + beta * kl
            totals["loss"] += loss.item()
            totals["recon"] += recon.item()
            totals["kl"] += kl.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}
