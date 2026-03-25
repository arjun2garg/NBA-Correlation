import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

N_STARTERS = 5
N_PLAYERS_PER_TEAM = 8
MINUTES_COL = "h_numMinutes"

STAT_COLS = [
    "h_points", "h_assists", "h_reboundsTotal",
    "h_steals",
    "h_fieldGoalsAttempted", "h_fieldGoalsMade",
    "h_threePointersAttempted", "h_threePointersMade",
    "h_freeThrowsAttempted", "h_freeThrowsMade",
    "h_reboundsDefensive", "h_reboundsOffensive",
    "h_foulsPersonal", "h_turnovers", "h_numMinutes",
    # Advanced stats (Layer 2) — present after running ingest/fetch_advanced_stats.py
    "h_usage_rate", "h_usage_share",
    "h_pace", "h_off_rating", "h_def_rating", "h_implied_total",
]

# Columns added to X_players but NOT pooled into X_team
PLAYER_EXTRA_COLS = ["home", "cov_pts_ast", "cov_pts_reb"]

# Game-level team scalars appended to X_team after the pooled stat block
GAME_TEAM_COLS = ["days_rest", "is_b2b"]

TARGET_COLS = ["points", "assists", "reboundsTotal"]


def load_processed(season_suffix="2024-25", processed_dir=None):
    d = Path(processed_dir) if processed_dir else PROCESSED_DIR
    input_df = pd.read_csv(d / f"input_data_{season_suffix}.csv")
    target_df = pd.read_csv(d / f"target_data_{season_suffix}.csv")
    df = pd.merge(input_df, target_df, how="inner", on=["personId", "gameId", "home"])
    df = df.dropna()
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    return df


def temporal_split(df, train_frac=0.8):
    df = df.sort_values("gameDateTimeEst")
    unique_dates = df["gameDateTimeEst"].unique()
    cutoff = unique_dates[int(len(unique_dates) * train_frac)]
    return df[df["gameDateTimeEst"] <= cutoff], df[df["gameDateTimeEst"] > cutoff]


def _pool_team(team_df, n_starters, minutes_col, stat_cols):
    """Top n_starters + bench pooled into one row. Returns None if too few players."""
    if len(team_df) < n_starters + 1:
        return None
    sorted_df = team_df.sort_values(minutes_col, ascending=False)
    starters = sorted_df.head(n_starters)
    bench = sorted_df.iloc[n_starters:]
    weights = bench[minutes_col].values
    bench_stats = {col: np.average(bench[col], weights=weights) for col in stat_cols}
    bench_row = {**{c: bench.iloc[0][c] for c in ["gameId", "home"]}, **bench_stats}
    return pd.concat([starters, pd.DataFrame([bench_row])], ignore_index=True)


def _safe_team_scalars(players_df, cols):
    """Extract game-level team scalars from the first player row; default to 0 if missing."""
    row = players_df.iloc[0]
    return np.array(
        [float(row[c]) if c in players_df.columns else 0.0 for c in cols],
        dtype=np.float32,
    )


def _safe_player_feats(player_df, stat_cols, extra_cols):
    """Build player feature matrix, filling any missing extra columns with 0."""
    base = player_df[stat_cols].values.astype(np.float32)
    parts = [base]
    for c in extra_cols:
        if c in player_df.columns:
            parts.append(player_df[[c]].values.astype(np.float32))
        else:
            parts.append(np.zeros((len(player_df), 1), dtype=np.float32))
    return np.hstack(parts)


def build_game(game_df, max_players, n_starters=N_STARTERS, n_players_per_team=N_PLAYERS_PER_TEAM,
               stat_cols=None, target_cols=None, minutes_col=MINUTES_COL):
    stat_cols = stat_cols or STAT_COLS
    target_cols = target_cols or TARGET_COLS

    home_players_all = game_df[game_df["home"] == 1].sort_values(minutes_col, ascending=False)
    away_players_all = game_df[game_df["home"] == 0].sort_values(minutes_col, ascending=False)

    home_pooled = _pool_team(home_players_all, n_starters, minutes_col, stat_cols)
    away_pooled = _pool_team(away_players_all, n_starters, minutes_col, stat_cols)

    if home_pooled is None or away_pooled is None:
        return None

    # X_team: pooled stat block for both teams + game-level scalars
    home_scalars = _safe_team_scalars(home_players_all, GAME_TEAM_COLS)
    away_scalars = _safe_team_scalars(away_players_all, GAME_TEAM_COLS)
    team_vec = np.concatenate([
        home_pooled[stat_cols].values.ravel(),
        away_pooled[stat_cols].values.ravel(),
        home_scalars,
        away_scalars,
    ])

    # Top n_players_per_team by minutes per team — avoids deep bench noise
    home_players = home_players_all.head(n_players_per_team)
    away_players = away_players_all.head(n_players_per_team)
    player_df = pd.concat([home_players, away_players], ignore_index=True)

    player_feats = _safe_player_feats(player_df, stat_cols, PLAYER_EXTRA_COLS)
    line_cols = ["h_" + c for c in target_cols]
    player_lines = player_df[line_cols].values
    # Targets are residuals vs player's own historical average.
    # The over/under threshold is therefore 0 for every player.
    player_targets = player_df[target_cols].values - player_lines
    player_lines = np.zeros((player_feats.shape[0], len(target_cols)))

    # Minutes-based weights: higher-minute players contribute more to the loss
    player_weights = player_df[minutes_col].values.astype(np.float32)

    num_players = player_feats.shape[0]
    pad = max_players - num_players
    if pad > 0:
        player_feats    = np.vstack([player_feats,    np.zeros((pad, player_feats.shape[1]))])
        player_targets  = np.vstack([player_targets,  np.zeros((pad, len(target_cols)))])
        player_lines    = np.vstack([player_lines,    np.zeros((pad, len(target_cols)))])
        player_weights  = np.concatenate([player_weights, np.zeros(pad)])

    return team_vec, player_feats, player_targets, player_weights, player_lines


def build_tensors(df, n_starters=N_STARTERS, n_players_per_team=N_PLAYERS_PER_TEAM,
                  stat_cols=None, target_cols=None, minutes_col=MINUTES_COL):
    stat_cols = stat_cols or STAT_COLS
    target_cols = target_cols or TARGET_COLS

    max_players = n_players_per_team * 2

    X_team, X_players, Y_list, weights_list, lines = [], [], [], [], []
    for _, game in df.groupby("gameId"):
        result = build_game(game, max_players, n_starters, n_players_per_team,
                            stat_cols, target_cols, minutes_col)
        if result is None:
            continue
        team_vec, player_feats, player_targets, player_weights, player_lines = result
        X_team.append(team_vec)
        X_players.append(player_feats)
        Y_list.append(player_targets)
        weights_list.append(player_weights)
        lines.append(player_lines)

    return (
        torch.tensor(np.stack(X_team), dtype=torch.float32),
        torch.tensor(np.stack(X_players), dtype=torch.float32),
        torch.tensor(np.stack(weights_list), dtype=torch.float32),
        torch.tensor(np.stack(Y_list), dtype=torch.float32),
        torch.tensor(np.stack(lines), dtype=torch.float32),
    )


class NBADataset(Dataset):
    def __init__(self, X_team, X_players, Y, weights, lines):
        self.X_team = X_team
        self.X_players = X_players
        self.Y = Y
        self.weights = weights
        self.lines = lines

    def __len__(self):
        return len(self.X_team)

    def __getitem__(self, idx):
        return self.X_team[idx], self.X_players[idx], self.Y[idx], self.weights[idx], self.lines[idx]


def make_loaders(train_df, val_df, batch_size=32, **tensor_kwargs):
    X_team_tr, X_pl_tr, weights_tr, Y_tr, lines_tr = build_tensors(train_df, **tensor_kwargs)
    X_team_val, X_pl_val, weights_val, Y_val, lines_val = build_tensors(val_df, **tensor_kwargs)

    # normalize targets using train statistics
    Y_mean = Y_tr.mean(dim=(0, 1))
    Y_std = Y_tr.std(dim=(0, 1)) + 1e-6
    Y_tr  = (Y_tr  - Y_mean) / Y_std
    Y_val = (Y_val - Y_mean) / Y_std

    # normalize team features
    Xt_mean = X_team_tr.mean(dim=0)
    Xt_std  = X_team_tr.std(dim=0) + 1e-6
    X_team_tr  = (X_team_tr  - Xt_mean) / Xt_std
    X_team_val = (X_team_val - Xt_mean) / Xt_std

    # normalize player features — mean/std over (games, players) per feature
    Xp_mean = X_pl_tr.mean(dim=(0, 1))
    Xp_std  = X_pl_tr.std(dim=(0, 1))  + 1e-6
    X_pl_tr  = (X_pl_tr  - Xp_mean) / Xp_std
    X_pl_val = (X_pl_val - Xp_mean) / Xp_std

    train_loader = DataLoader(
        NBADataset(X_team_tr, X_pl_tr, Y_tr, weights_tr, lines_tr),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        NBADataset(X_team_val, X_pl_val, Y_val, weights_val, lines_val),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader, Y_mean, Y_std, Xt_mean, Xt_std, Xp_mean, Xp_std
