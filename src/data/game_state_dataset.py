"""
Dataset extension for game state anchored training (Phase 2).

Adds actual game state G to each batch so the training loop can compute
the auxiliary loss: MSE(G_pred, actual_G_normalized).

G is normalized using training-set statistics (same as other features).
Games without G data are assigned zeros and masked out from the GS loss.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.data.dataset import (
    load_processed, temporal_split, build_tensors, STAT_COLS, TARGET_COLS,
    MINUTES_COL, N_PLAYERS_PER_TEAM, N_STARTERS, PLAYER_EXTRA_COLS,
)
from src.data.game_state import build_game_state_df, GAME_STATE_COLS

ROOT = Path(__file__).resolve().parents[2]

# Only TS-based features (always available, no PBP complexity)
GS_TARGET_COLS = [
    "home_actual_pace", "home_actual_3pa_rate", "home_actual_ast_rate",
    "home_actual_to_rate", "home_actual_margin", "home_actual_oreb_rate",
    "away_actual_pace", "away_actual_3pa_rate", "away_actual_ast_rate",
    "away_actual_to_rate", "away_actual_margin", "away_actual_oreb_rate",
]
G_DIM = len(GS_TARGET_COLS)


class NBADatasetGS(Dataset):
    """NBADataset extended with game state G tensor."""

    def __init__(self, X_team, X_players, Y, weights, lines, G, G_mask):
        self.X_team = X_team
        self.X_players = X_players
        self.Y = Y
        self.weights = weights
        self.lines = lines
        self.G = G          # (n_games, G_DIM) normalized game state
        self.G_mask = G_mask  # (n_games,) bool — True if G is valid

    def __len__(self):
        return len(self.X_team)

    def __getitem__(self, idx):
        return (
            self.X_team[idx], self.X_players[idx], self.Y[idx],
            self.weights[idx], self.lines[idx],
            self.G[idx], self.G_mask[idx],
        )


def _build_G_tensor(game_ids: list, game_state_df: pd.DataFrame) -> tuple:
    """
    Build G tensor and mask for a list of game IDs.

    Returns:
        G_arr: np.ndarray (n_games, G_DIM)
        mask:  np.ndarray (n_games,) bool
    """
    gs_map = game_state_df.set_index("gameId")
    G_arr = np.zeros((len(game_ids), G_DIM), dtype=np.float32)
    mask = np.zeros(len(game_ids), dtype=bool)

    for i, gid in enumerate(game_ids):
        if gid in gs_map.index:
            row = gs_map.loc[gid]
            vals = []
            valid = True
            for col in GS_TARGET_COLS:
                v = row.get(col, np.nan) if isinstance(row, pd.Series) else np.nan
                if np.isnan(v):
                    valid = False
                    break
                vals.append(v)
            if valid:
                G_arr[i] = vals
                mask[i] = True

    return G_arr, mask


def make_loaders_gs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 64,
    include_pbp: bool = False,
    **tensor_kwargs,
):
    """
    Build DataLoaders with game state G tensors included.

    Returns everything make_loaders returns, plus:
        G_mean, G_std   — normalization stats for game state
        G_dim           — dimensionality of G
        train_loader, val_loader with G and G_mask in each batch
    """
    # Build standard tensors
    X_team_tr, X_pl_tr, weights_tr, Y_tr, lines_tr = build_tensors(train_df, **tensor_kwargs)
    X_team_val, X_pl_val, weights_val, Y_val, lines_val = build_tensors(val_df, **tensor_kwargs)

    # Get ordered game IDs matching the tensor order
    train_game_ids = _get_ordered_game_ids(train_df)
    val_game_ids = _get_ordered_game_ids(val_df)

    # Load game state
    print("Loading game state G...")
    game_state_df = build_game_state_df(include_pbp=include_pbp)

    # Build raw G tensors
    G_tr_raw, mask_tr = _build_G_tensor(train_game_ids, game_state_df)
    G_val_raw, mask_val = _build_G_tensor(val_game_ids, game_state_df)

    print(f"  G coverage: train {mask_tr.sum()}/{len(mask_tr)}, val {mask_val.sum()}/{len(mask_val)}")

    # Normalize G using train-set stats (only rows where mask is True)
    G_mean = np.zeros(G_DIM, dtype=np.float32)
    G_std = np.ones(G_DIM, dtype=np.float32)
    if mask_tr.sum() > 0:
        G_train_valid = G_tr_raw[mask_tr]
        G_mean = G_train_valid.mean(axis=0).astype(np.float32)
        G_std = (G_train_valid.std(axis=0) + 1e-6).astype(np.float32)
        G_tr_raw[mask_tr] = (G_tr_raw[mask_tr] - G_mean) / G_std
        G_val_raw[mask_val] = (G_val_raw[mask_val] - G_mean) / G_std

    # Normalize standard features (same as make_loaders)
    Y_mean = Y_tr.mean(dim=(0, 1))
    Y_std = Y_tr.std(dim=(0, 1)) + 1e-6
    Y_tr = (Y_tr - Y_mean) / Y_std
    Y_val = (Y_val - Y_mean) / Y_std

    Xt_mean = X_team_tr.mean(dim=0)
    Xt_std = X_team_tr.std(dim=0) + 1e-6
    X_team_tr = (X_team_tr - Xt_mean) / Xt_std
    X_team_val = (X_team_val - Xt_mean) / Xt_std

    Xp_mean = X_pl_tr.mean(dim=(0, 1))
    Xp_std = X_pl_tr.std(dim=(0, 1)) + 1e-6
    X_pl_tr = (X_pl_tr - Xp_mean) / Xp_std
    X_pl_val = (X_pl_val - Xp_mean) / Xp_std

    G_tr = torch.tensor(G_tr_raw, dtype=torch.float32)
    G_val = torch.tensor(G_val_raw, dtype=torch.float32)
    mask_tr_t = torch.tensor(mask_tr, dtype=torch.bool)
    mask_val_t = torch.tensor(mask_val, dtype=torch.bool)

    train_loader = DataLoader(
        NBADatasetGS(X_team_tr, X_pl_tr, Y_tr, weights_tr, lines_tr, G_tr, mask_tr_t),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        NBADatasetGS(X_team_val, X_pl_val, Y_val, weights_val, lines_val, G_val, mask_val_t),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    norm_stats = {
        "Y_mean": Y_mean, "Y_std": Y_std,
        "Xt_mean": Xt_mean, "Xt_std": Xt_std,
        "Xp_mean": Xp_mean, "Xp_std": Xp_std,
        "G_mean": torch.tensor(G_mean), "G_std": torch.tensor(G_std),
    }
    return train_loader, val_loader, norm_stats


def _get_ordered_game_ids(df: pd.DataFrame) -> list:
    """Return game IDs in the order they appear in build_tensors output."""
    # build_tensors iterates groupby("gameId") which sorts game IDs
    game_ids = sorted(df["gameId"].unique())
    # build_game uses all player rows — filter to games with enough players
    valid = []
    for gid, gdf in df.groupby("gameId"):
        home = gdf[gdf["home"] == 1]
        away = gdf[gdf["home"] == 0]
        if len(home) >= N_STARTERS + 1 and len(away) >= N_STARTERS + 1:
            valid.append(gid)
    return sorted(valid)
