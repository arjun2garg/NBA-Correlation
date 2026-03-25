import torch
from src.simulate import var_label


def phi_coefficient(OO, OU, UO, UU, mask_flat, n_stats):
    """
    Compute phi (Matthews) correlation coefficient for all variable pairs.

    Args:
        OO, OU, UO, UU : (batch, n_vars, n_vars) — joint outcome counts
        mask_flat       : (batch, n_vars) — 1 for real player-stat slots, 0 for padding
        n_stats         : int — number of stats per player (used to identify same-player pairs)

    Returns:
        phi : (batch, n_vars, n_vars) — NaN for padded, diagonal, and same-player entries
    """
    num = OO * UU - OU * UO
    denom = torch.sqrt((OO + OU) * (OO + UO) * (UU + OU) * (UU + UO) + 1e-8)
    phi = (num / denom).clamp(-1, 1)

    nan = torch.tensor(float("nan"))
    n_vars = phi.shape[1]

    # mask out padded pairs
    pair_mask = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)   # (batch, n_vars, n_vars)
    phi = torch.where(pair_mask.bool(), phi, nan)

    # mask out same-player pairs (different stats, same player slot)
    player_ids = torch.arange(n_vars, device=phi.device) // n_stats
    same_player = (player_ids.unsqueeze(1) == player_ids.unsqueeze(0)).unsqueeze(0)  # (1, n_vars, n_vars)
    phi = torch.where(same_player.expand_as(phi), nan, phi)

    return phi


def extract_pairs(phi, OO, OU, UO, UU, num_samples, mask_flat, stat_names, threshold=0.15, top_k=10):
    """
    For each game in the batch, find the top-k pairs with |phi| above threshold.
    Covers all four directions: OO, UU (positive phi) and OU, UO (negative phi).

    Args:
        phi             : (batch, n_vars, n_vars)
        OO, OU, UO, UU  : (batch, n_vars, n_vars) — joint outcome counts
        num_samples     : int
        mask_flat       : (batch, n_vars)
        stat_names      : list[str]
        threshold       : minimum |phi| to include
        top_k           : max pairs per game

    Returns:
        list[list[dict]] — outer: batch dim, inner: pairs for that game
    """
    batch_size, n_vars, _ = phi.shape
    n_stats = len(stat_names)

    # upper triangle mask to avoid duplicate (i,j)/(j,i) pairs
    upper = torch.triu(torch.ones(n_vars, n_vars, dtype=torch.bool, device=phi.device), diagonal=1)

    all_pairs = []
    for b in range(batch_size):
        phi_b = phi[b].clone()
        phi_b[~upper] = float("nan")

        flat = phi_b.reshape(-1)
        abs_flat = flat.abs()
        valid = ~torch.isnan(flat) & (abs_flat >= threshold)
        indices = valid.nonzero(as_tuple=False).squeeze(1)

        if len(indices) == 0:
            all_pairs.append([])
            continue

        # sort by |phi| descending
        vals = abs_flat[indices]
        sorted_idx = vals.argsort(descending=True)[:top_k]
        top_indices = indices[sorted_idx]

        pairs = []
        for idx in top_indices:
            i, j = idx.item() // n_vars, idx.item() % n_vars
            phi_val = phi_b[i, j].item()
            oo_prob = OO[b, i, j].item() / num_samples
            ou_prob = OU[b, i, j].item() / num_samples
            uo_prob = UO[b, i, j].item() / num_samples
            uu_prob = UU[b, i, j].item() / num_samples

            if phi_val >= 0:
                predicted_dir = "OO" if oo_prob >= uu_prob else "UU"
            else:
                predicted_dir = "OU" if ou_prob >= uo_prob else "UO"

            pairs.append({
                "i": i,
                "j": j,
                "phi": phi_val,
                "i_label": var_label(i, n_stats, stat_names),
                "j_label": var_label(j, n_stats, stat_names),
                "predicted_dir": predicted_dir,
                "oo_prob": oo_prob,
                "ou_prob": ou_prob,
                "uo_prob": uo_prob,
                "uu_prob": uu_prob,
            })
        all_pairs.append(pairs)

    return all_pairs


def backtest(pairs_per_game, actual_over, parlay_odds=-110):
    """
    Simulate flat-stake 2-leg parlay bets on each identified pair.

    Args:
        pairs_per_game : list[list[dict]] — from extract_pairs
        actual_over    : (batch, n_vars) int tensor
        parlay_odds    : American odds per leg (both legs same)

    Returns:
        dict: {bets, wins, losses, win_rate, net_pnl, roi, breakeven_wr, records}
    """
    decimal = 1 + 100 / abs(parlay_odds)
    multiplier = decimal ** 2
    breakeven_wr = 1 / multiplier

    records = []
    for b, pairs in enumerate(pairs_per_game):
        for pair in pairs:
            i, j = pair["i"], pair["j"]
            i_actual = actual_over[b, i].item()
            j_actual = actual_over[b, j].item()

            direction = pair["predicted_dir"]
            if direction == "OO":
                won = (i_actual == 1 and j_actual == 1)
            elif direction == "UU":
                won = (i_actual == 0 and j_actual == 0)
            elif direction == "OU":
                won = (i_actual == 1 and j_actual == 0)
            else:  # UO
                won = (i_actual == 0 and j_actual == 1)

            records.append({
                "game_idx": b,
                "i": i, "j": j,
                "i_label": pair["i_label"],
                "j_label": pair["j_label"],
                "phi": pair["phi"],
                "predicted_dir": pair["predicted_dir"],
                "i_actual": i_actual,
                "j_actual": j_actual,
                "won": won,
                "pnl": multiplier - 1 if won else -1.0,
            })

    bets = len(records)
    wins = sum(r["won"] for r in records)
    net_pnl = sum(r["pnl"] for r in records)

    return {
        "bets": bets,
        "wins": wins,
        "losses": bets - wins,
        "win_rate": wins / bets if bets > 0 else 0.0,
        "net_pnl": net_pnl,
        "roi": net_pnl / bets if bets > 0 else 0.0,
        "breakeven_wr": breakeven_wr,
        "records": records,
    }
