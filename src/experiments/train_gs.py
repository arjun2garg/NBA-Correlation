"""
Two-stage training for Game State VAE.

Stage 1 — train_decoder_stage1():
  Supervised: actual G + player_feats → residuals.
  No encoder, no KL, no reparameterization. Pure NLL minimization.

Stage 2 — train_encoder_stage2():
  Decoder is FROZEN. Train encoder: X_team → distribution over G.
  Loss: Gaussian NLL of actual G + beta * KL(q || N(0,1))
  KL prevents posterior collapse and keeps z samples meaningful at inference.
"""

import torch
from src.model_gs import reparameterize


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def masked_nll(mu_pred, logvar_pred, target, weights):
    """Minutes-weighted Gaussian NLL over player-game residuals."""
    nll = 0.5 * (logvar_pred + (target - mu_pred).pow(2) / logvar_pred.exp())
    nll = nll * weights.unsqueeze(-1)
    return nll.sum() / (weights.sum() * mu_pred.size(-1) + 1e-8)


def kl_divergence(mu, logvar, free_bits=0.0):
    """KL(N(mu, exp(logvar)) || N(0,1)) summed over dimensions."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim.mean(dim=0), min=free_bits)
        return kl_per_dim.sum()
    return kl_per_dim.mean(dim=0).sum()


def g_nll(mu_G, logvar_G, actual_G, G_mask):
    """
    Gaussian NLL of actual game state under encoder's predicted distribution.

    mu_G, logvar_G: (batch, G_DIM)  — encoder output
    actual_G:       (batch, G_DIM)  — normalized actual game state
    G_mask:         (batch,) bool   — True if game has valid G
    """
    if G_mask.sum() == 0:
        return torch.tensor(0.0, device=mu_G.device)
    nll = 0.5 * (logvar_G + (actual_G - mu_G).pow(2) / logvar_G.exp())
    return nll[G_mask].mean()


# ---------------------------------------------------------------------------
# Stage 1: Train decoder on actual G
# ---------------------------------------------------------------------------

def train_decoder_epoch(decoder, optimizer, loader, device="cpu"):
    """One epoch of supervised decoder training on actual G."""
    decoder.train()
    totals = {"loss": 0.0}

    for X_t, X_p, Y, weights, _, G, G_mask in loader:
        X_p = X_p.to(device)
        Y = Y.to(device)
        weights = weights.to(device)
        G = G.to(device)
        G_mask = G_mask.to(device)

        # Only train on games with valid G
        if G_mask.sum() == 0:
            continue

        optimizer.zero_grad()
        mu_pred, logvar_pred = decoder(G, X_p)
        loss = masked_nll(mu_pred, logvar_pred, Y, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        totals["loss"] += loss.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


def eval_decoder(decoder, loader, device="cpu"):
    """Evaluate decoder NLL on actual G."""
    decoder.eval()
    totals = {"loss": 0.0}

    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G, G_mask in loader:
            X_p = X_p.to(device)
            Y = Y.to(device)
            weights = weights.to(device)
            G = G.to(device)
            G_mask = G_mask.to(device)

            if G_mask.sum() == 0:
                continue

            mu_pred, logvar_pred = decoder(G, X_p)
            loss = masked_nll(mu_pred, logvar_pred, Y, weights)
            totals["loss"] += loss.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Stage 2: Train encoder to predict G (decoder frozen)
# ---------------------------------------------------------------------------

def train_encoder_epoch(encoder, decoder, optimizer, loader,
                         beta=0.01, free_bits=0.5, device="cpu"):
    """
    One epoch of encoder training. Decoder is frozen.

    Loss = NLL(actual G | mu_G, logvar_G) + beta * KL(q || N(0,1))

    The NLL term pushes mu_G toward actual G.
    The KL term prevents sigma_G from collapsing to 0, which would make
    z samples identical at inference and kill the correlation mechanism.
    """
    encoder.train()
    decoder.eval()  # frozen
    totals = {"loss": 0.0, "g_nll": 0.0, "kl": 0.0}

    for X_t, X_p, Y, weights, _, G, G_mask in loader:
        X_t = X_t.to(device)
        G = G.to(device)
        G_mask = G_mask.to(device)

        if G_mask.sum() == 0:
            continue

        optimizer.zero_grad()
        mu_G, logvar_G = encoder(X_t)

        gnll = g_nll(mu_G, logvar_G, G, G_mask)
        kl = kl_divergence(mu_G, logvar_G, free_bits=free_bits)
        loss = gnll + beta * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["g_nll"] += gnll.item()
        totals["kl"] += kl.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


def eval_encoder(encoder, loader, beta=0.01, device="cpu"):
    """Evaluate encoder's G prediction quality."""
    encoder.eval()
    totals = {"loss": 0.0, "g_nll": 0.0, "kl": 0.0}
    gs_preds, gs_actuals = [], []

    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G, G_mask in loader:
            X_t = X_t.to(device)
            G = G.to(device)
            G_mask = G_mask.to(device)

            if G_mask.sum() == 0:
                continue

            mu_G, logvar_G = encoder(X_t)
            gnll = g_nll(mu_G, logvar_G, G, G_mask)
            kl = kl_divergence(mu_G, logvar_G)
            loss = gnll + beta * kl

            totals["loss"] += loss.item()
            totals["g_nll"] += gnll.item()
            totals["kl"] += kl.item()

            gs_preds.append(mu_G[G_mask].cpu())
            gs_actuals.append(G[G_mask].cpu())

    n = max(len(loader), 1)
    metrics = {k: v / n for k, v in totals.items()}

    # R² of encoder's mean prediction vs actual G
    if gs_preds:
        import numpy as np
        gp = torch.cat(gs_preds).numpy()
        ga = torch.cat(gs_actuals).numpy()
        ss_res = ((gp - ga) ** 2).sum(axis=0)
        ss_tot = ((ga - ga.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
        metrics["g_r2"] = float((1 - ss_res / ss_tot).mean())
    else:
        metrics["g_r2"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Simulation diagnostic
# ---------------------------------------------------------------------------

def compute_p_over_std(encoder, decoder, loader, n_samples=200, device="cpu"):
    """
    Key diagnostic: std of P(over|G_sample) across G samples per game.

    High std → encoder uncertainty drives player outcome variation → phi possible.
    Target: std > 0.05.

    With two-stage design:
      - G samples come from encoder's posterior N(mu_G, sigma_G)
      - Decoder is G-sensitive (trained on actual G)
      - Expected std ≈ phi(0) * |dmu_pred/dG| * sigma_G / sigma_pred
    """
    import numpy as np

    encoder.eval()
    decoder.eval()
    all_stds = []
    sigma_preds = []

    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G, G_mask in loader:
            X_t = X_t.to(device)
            X_p = X_p.to(device)

            mu_G, logvar_G = encoder(X_t)

            p_overs = []
            for _ in range(n_samples):
                G_sample = reparameterize(mu_G, logvar_G)
                mu_pred, logvar_pred = decoder(G_sample, X_p)
                sigma_pred = (0.5 * logvar_pred).exp()
                p_over = torch.distributions.Normal(0, 1).cdf(mu_pred / (sigma_pred + 1e-8))
                p_overs.append(p_over.cpu())

                if len(sigma_preds) < 10000:
                    active = X_p.cpu().abs().sum(-1) > 0
                    sigma_preds.extend(sigma_pred.cpu()[active.unsqueeze(-1).expand_as(sigma_pred)].tolist())

            p_stack = torch.stack(p_overs)  # (n_samples, batch, players, stats)
            std_per = p_stack.std(dim=0)    # (batch, players, stats)
            active = X_p.cpu().abs().sum(-1) > 0
            stds = std_per[active.unsqueeze(-1).expand_as(std_per)]
            all_stds.extend(stds.tolist())

            if len(all_stds) > 100000:
                break

    return {
        "p_over_std_mean": float(np.mean(all_stds)) if all_stds else 0.0,
        "sigma_pred_mean": float(np.mean(sigma_preds)) if sigma_preds else 0.0,
    }
