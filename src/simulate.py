import torch
from src.model import reparameterize


def simulate(encoder, decoder, loader, Y_mean, Y_std, num_samples=100, device="cpu"):
    """
    Run Monte Carlo simulation over a DataLoader.

    Returns per-batch joint outcome counts: OO, OU, UO, UU
    each of shape (batch, n_vars, n_vars) where n_vars = n_players * n_stats.
    """
    encoder.eval()
    decoder.eval()

    results = []

    with torch.no_grad():
        for X_t, X_p, Y, weights, lines in loader:
            X_t, X_p, Y, weights, lines = (
                X_t.to(device), X_p.to(device), Y.to(device), weights.to(device), lines.to(device)
            )
            mask = (weights > 0).float()
            mu, logvar = encoder(X_t)

            # lines in normalized space: (batch, 16, 3)
            lines_norm = (lines - Y_mean.to(device)) / Y_std.to(device)

            # sample num_samples z values; compute P(over | z) analytically via normal CDF
            over_samples = []
            for _ in range(num_samples):
                z = reparameterize(mu, logvar)
                mu_pred, logvar_pred = decoder(z, X_p)          # (batch, 16, 3)
                sigma_pred = (0.5 * logvar_pred).exp()          # (batch, 16, 3)
                z_score = (mu_pred - lines_norm) / sigma_pred
                p_over = 0.5 * (1.0 + torch.erf(z_score / (2.0 ** 0.5)))
                over_samples.append(p_over)                     # continuous [0, 1]

            # over: (num_samples, batch, n_players, n_stats)
            over = torch.stack(over_samples, dim=0)
            batch_size, n_players, n_stats = over.shape[1], over.shape[2], over.shape[3]

            # flatten players*stats, put samples last → (batch, n_vars, num_samples)
            A = over.permute(1, 2, 3, 0).reshape(batch_size, n_players * n_stats, num_samples).float()

            OO, OU, UO, UU = compute_joint_outcomes(A, num_samples)
            over_mean = A.mean(dim=2)       # (batch, n_vars) — marginal P(over) per variable

            # actual outcomes: denormalize Y, compare to lines, flatten, mask padded slots
            Y_actual = Y * Y_std.to(device) + Y_mean.to(device)
            actual_over = (Y_actual > lines).int()
            n_vars = n_players * n_stats
            mask_flat = mask.unsqueeze(-1).expand(-1, -1, n_stats).reshape(batch_size, n_vars)
            actual_over_flat = actual_over.reshape(batch_size, n_vars) * mask_flat.int()

            results.append({
                "OO": OO, "OU": OU, "UO": UO, "UU": UU,
                "over_mean": over_mean,
                "actual_over": actual_over_flat,
                "mask_flat": mask_flat,
            })

    return results


def compute_joint_outcomes(A, num_samples):
    """
    A: (batch, n_vars, num_samples) — P(over | z_s) for each sample s; continuous [0, 1].

    Returns unnormalized joint outcome matrices (batch, n_vars, n_vars).
    Dividing by num_samples gives joint probabilities:
      OO ≈ E_z[P(over_i|z) * P(over_j|z)]
      OU ≈ E_z[P(over_i|z) * P(under_j|z)]  etc.
    """
    S = A.sum(dim=2)                         # (batch, n_vars) — over count per var
    OO = A @ A.transpose(1, 2)               # (batch, n_vars, n_vars)
    OU = S.unsqueeze(2) - OO                 # i over, j under
    UO = S.unsqueeze(1) - OO                 # i under, j over
    UU = num_samples - OO - OU - UO          # both under
    return OO, OU, UO, UU


def joint_probabilities(OO, OU, UO, UU, num_samples):
    """Normalize joint outcome counts to probabilities."""
    return OO / num_samples, OU / num_samples, UO / num_samples, UU / num_samples


def var_label(k, n_stats, stat_names):
    """Map flat variable index k → (player_slot, stat_name). Slot 0 = highest-minutes player."""
    return k // n_stats, stat_names[k % n_stats]
