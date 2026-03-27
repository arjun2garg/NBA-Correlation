"""
Player-specific G experiment (v7_player_mins_fga).

Tests how much sigma_pred drops when G includes per-player actual minutes and FGA.
This is the key experiment for answering: "can player-level G unlock phi > 0.15?"

Architecture:
  G = (G_game, G_player) where:
    G_game:   (batch, g_game_dim) — team-level state (pts, ast, reb per team)
    G_player: (batch, 16, g_player_dim) — per-player actual mins + FGA

  Decoder: G_game (broadcast) + G_player (player-specific) + X_players → mu, logvar

At simulation time:
  Encoder predicts G_game ~ N(mu_game, sigma_game) and G_player ~ N(mu_player, sigma_player)
  Sampling G drives correlation: high G_game → all players benefit (positive)
  G_player captures player-specific opportunity (negative within-team for minutes,
  but positive across all stats for a given player)

Results tell us:
  1. How low sigma_pred can go with player-level G
  2. Whether phi > 0.15 is achievable with this encoding
  3. Which G features matter most (mins? FGA? both?)

Usage:
  python scripts/train_player_g.py --epochs 60 --ckpt-dir checkpoints_v7
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_num_threads(1)

import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import (
    load_processed, temporal_split, build_tensors,
    N_PLAYERS_PER_TEAM, N_STARTERS, STAT_COLS, TARGET_COLS, MINUTES_COL,
    PLAYER_EXTRA_COLS, GAME_TEAM_COLS,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Game-level G features (team totals × 2 teams)
G_GAME_COLS = ["actual_pts", "actual_ast", "actual_reb"]  # per team → ×2 = 6-dim
G_GAME_DIM = 6

# Per-player G: actual minutes + actual FGA → 2-dim per slot
G_PLAYER_DIM = 2  # [actual_mins_norm, actual_fga_norm]

MAX_PLAYERS = N_PLAYERS_PER_TEAM * 2  # 16
PLAYER_DIM = 24  # len(STAT_COLS) + len(PLAYER_EXTRA_COLS)
OUTPUT_DIM = len(TARGET_COLS)  # 3


# ---------------------------------------------------------------------------
# Player-specific G decoder
# ---------------------------------------------------------------------------

class PlayerGDecoder(nn.Module):
    """
    G-conditioned player decoder with player-specific G.

    G_game (batch, g_game_dim) — same for all players in a game (team totals)
    G_player (batch, max_players, g_player_dim) — per-player actual mins + FGA
    player_feats (batch, max_players, player_dim) — historical features

    Input per player: G_game + G_player_i + player_feats_i
    """

    def __init__(self, g_game_dim=G_GAME_DIM, g_player_dim=G_PLAYER_DIM,
                 player_dim=PLAYER_DIM, h_dim=64, output_dim=OUTPUT_DIM, dropout=0.3):
        super().__init__()
        total_input = g_game_dim + g_player_dim + player_dim
        self.trunk = nn.Sequential(
            nn.Linear(total_input, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, G_game, G_player, player_feats):
        """
        G_game:       (batch, g_game_dim)
        G_player:     (batch, max_players, g_player_dim)
        player_feats: (batch, max_players, player_dim)
        Returns:      mu_pred, logvar_pred  each (batch, max_players, output_dim)
        """
        G_game_exp = G_game.unsqueeze(1).expand(-1, player_feats.size(1), -1)
        x = torch.cat([G_game_exp, G_player, player_feats], dim=-1)
        h = self.trunk(x)
        return self.mu_head(h), self.logvar_head(h).clamp(-6.0, 2.0)


# ---------------------------------------------------------------------------
# Dataset building: player-specific G from actual stats
# ---------------------------------------------------------------------------

def _load_player_actuals(raw_dir: Path) -> pd.DataFrame:
    """Load actual player minutes and FGA from PlayerStatistics."""
    print("Loading PlayerStatistics for actual mins + FGA...")
    ps = pd.read_csv(raw_dir / "PlayerStatistics.csv", low_memory=False,
                     usecols=["gameId", "personId", "numMinutes", "fieldGoalsAttempted"])
    ps["gameId"] = pd.to_numeric(ps["gameId"], errors="coerce")
    ps["personId"] = pd.to_numeric(ps["personId"], errors="coerce")
    ps["actual_mins"] = pd.to_numeric(ps["numMinutes"], errors="coerce").fillna(0)
    ps["actual_fga"] = pd.to_numeric(ps["fieldGoalsAttempted"], errors="coerce").fillna(0)
    ps = ps[["gameId", "personId", "actual_mins", "actual_fga"]].dropna()
    print(f"  Loaded {len(ps):,} player-game rows")
    return ps


def _load_team_actuals(raw_dir: Path) -> pd.DataFrame:
    """Load actual team stats (pts/ast/reb) from TeamStatistics."""
    ts = pd.read_csv(raw_dir / "TeamStatistics.csv",
                     usecols=["gameId", "teamId", "home", "teamScore", "assists",
                               "reboundsTotal", "numMinutes"])
    ts["gameId"] = pd.to_numeric(ts["gameId"], errors="coerce")
    ts["teamId"] = pd.to_numeric(ts["teamId"], errors="coerce")
    ts["actual_pts"] = pd.to_numeric(ts["teamScore"], errors="coerce")
    ts["actual_ast"] = pd.to_numeric(ts["assists"], errors="coerce")
    ts["actual_reb"] = pd.to_numeric(ts["reboundsTotal"], errors="coerce")
    ts = ts.dropna(subset=["gameId", "actual_pts", "actual_ast", "actual_reb"])
    return ts[["gameId", "home", "actual_pts", "actual_ast", "actual_reb"]]


def build_player_g_tensors(df: pd.DataFrame, player_actuals: pd.DataFrame) -> tuple:
    """
    Build per-game G tensors aligned with build_tensors() player slot ordering.

    Players in each game are sorted by h_numMinutes (descending) — same as build_game().
    For each slot, fetch actual mins + FGA from player_actuals.

    Returns:
        game_ids: list of game IDs (in same order as build_tensors outputs)
        G_player_arr: np.ndarray (n_games, 16, 2) — [actual_mins, actual_fga] per slot
        G_player_mask: np.ndarray (n_games,) bool — True if all player actuals available
    """
    # Build O(1) lookup dict: (gameId, personId) → (mins, fga)
    pa_dict = {
        (row.gameId, row.personId): (row.actual_mins, row.actual_fga)
        for row in player_actuals.itertuples(index=False)
    }

    game_ids = []
    G_player_list = []
    G_player_mask = []

    for game_id, game_df in df.groupby("gameId"):
        home_players = game_df[game_df["home"] == 1].sort_values(MINUTES_COL, ascending=False)
        away_players = game_df[game_df["home"] == 0].sort_values(MINUTES_COL, ascending=False)

        if len(home_players) < N_STARTERS + 1 or len(away_players) < N_STARTERS + 1:
            continue

        home_top8 = home_players.head(N_PLAYERS_PER_TEAM)
        away_top8 = away_players.head(N_PLAYERS_PER_TEAM)
        ordered = pd.concat([home_top8, away_top8], ignore_index=True)

        # Fetch actual stats for each slot
        g_player = np.zeros((MAX_PLAYERS, G_PLAYER_DIM), dtype=np.float32)
        valid = True

        for slot_idx, row in enumerate(ordered.itertuples()):
            key = (game_id, row.personId)
            if key in pa_dict:
                mins, fga = pa_dict[key]
                g_player[slot_idx, 0] = float(mins)
                g_player[slot_idx, 1] = float(fga)
            else:
                valid = False
                break

        game_ids.append(game_id)
        G_player_list.append(g_player)
        G_player_mask.append(valid)

    return (
        game_ids,
        np.array(G_player_list, dtype=np.float32),
        np.array(G_player_mask, dtype=bool),
    )


def build_game_g_tensors(df: pd.DataFrame, team_actuals: pd.DataFrame) -> tuple:
    """
    Build game-level G tensors: [home_pts, home_ast, home_reb, away_pts, away_ast, away_reb].

    Returns dict: game_id → G_game (6-dim np array)
    """
    ts_home = team_actuals[team_actuals["home"] == 1].set_index("gameId")
    ts_away = team_actuals[team_actuals["home"] == 0].set_index("gameId")
    return ts_home, ts_away


class NBADatasetPlayerG(Dataset):
    """Dataset with game-level G + player-specific G."""

    def __init__(self, X_team, X_players, Y, weights, lines, G_game, G_player, G_mask):
        self.X_team = X_team
        self.X_players = X_players
        self.Y = Y
        self.weights = weights
        self.lines = lines
        self.G_game = G_game    # (n, G_GAME_DIM)
        self.G_player = G_player  # (n, 16, G_PLAYER_DIM)
        self.G_mask = G_mask    # (n,) bool

    def __len__(self):
        return len(self.X_team)

    def __getitem__(self, idx):
        return (
            self.X_team[idx], self.X_players[idx], self.Y[idx],
            self.weights[idx], self.lines[idx],
            self.G_game[idx], self.G_player[idx], self.G_mask[idx],
        )


def make_loaders_player_g(train_df, val_df, batch_size=64, raw_dir=RAW_DIR):
    """Build DataLoaders with game-level G + player-specific G tensors."""
    # Standard tensors
    print("Building standard tensors...")
    X_team_tr, X_pl_tr, weights_tr, Y_tr, lines_tr = build_tensors(train_df)
    X_team_val, X_pl_val, weights_val, Y_val, lines_val = build_tensors(val_df)

    # Load actual stats
    player_actuals = _load_player_actuals(raw_dir)
    team_actuals = _load_team_actuals(raw_dir)

    ts_home, ts_away = build_game_g_tensors(train_df, team_actuals)

    # Build player-specific G tensors
    print("Building player-specific G tensors (train)...")
    tr_game_ids, G_pl_tr_raw, G_pl_mask_tr = build_player_g_tensors(train_df, player_actuals)
    print("Building player-specific G tensors (val)...")
    va_game_ids, G_pl_va_raw, G_pl_mask_va = build_player_g_tensors(val_df, player_actuals)

    print(f"  Player G coverage: train {G_pl_mask_tr.sum()}/{len(G_pl_mask_tr)}, "
          f"val {G_pl_mask_va.sum()}/{len(G_pl_mask_va)}")

    # Build game-level G arrays aligned to game_ids
    def get_game_g(game_ids, ts_home, ts_away):
        G_game = np.zeros((len(game_ids), G_GAME_DIM), dtype=np.float32)
        G_game_mask = np.zeros(len(game_ids), dtype=bool)
        for i, gid in enumerate(game_ids):
            if gid in ts_home.index and gid in ts_away.index:
                h = ts_home.loc[gid]
                a = ts_away.loc[gid]
                if isinstance(h, pd.DataFrame): h = h.iloc[0]
                if isinstance(a, pd.DataFrame): a = a.iloc[0]
                G_game[i] = [
                    h["actual_pts"], h["actual_ast"], h["actual_reb"],
                    a["actual_pts"], a["actual_ast"], a["actual_reb"],
                ]
                G_game_mask[i] = True
        return G_game, G_game_mask

    # Also need all-team actuals for val
    ts_home_all = team_actuals[team_actuals["home"] == 1].set_index("gameId")
    ts_away_all = team_actuals[team_actuals["home"] == 0].set_index("gameId")

    G_game_tr_raw, G_game_mask_tr = get_game_g(tr_game_ids, ts_home_all, ts_away_all)
    G_game_va_raw, G_game_mask_va = get_game_g(va_game_ids, ts_home_all, ts_away_all)

    # Combined mask: both game G and player G must be available
    combined_mask_tr = G_pl_mask_tr & G_game_mask_tr
    combined_mask_va = G_pl_mask_va & G_game_mask_va
    print(f"  Combined G mask: train {combined_mask_tr.sum()}/{len(combined_mask_tr)}, "
          f"val {combined_mask_va.sum()}/{len(combined_mask_va)}")

    # Normalize game G using train stats
    G_game_mean = np.zeros(G_GAME_DIM, dtype=np.float32)
    G_game_std = np.ones(G_GAME_DIM, dtype=np.float32)
    if combined_mask_tr.sum() > 0:
        valid = G_game_tr_raw[combined_mask_tr]
        G_game_mean = valid.mean(axis=0).astype(np.float32)
        G_game_std = (valid.std(axis=0) + 1e-6).astype(np.float32)
        G_game_tr_raw[combined_mask_tr] = (G_game_tr_raw[combined_mask_tr] - G_game_mean) / G_game_std
        G_game_va_raw[combined_mask_va] = (G_game_va_raw[combined_mask_va] - G_game_mean) / G_game_std

    # Normalize player G using train stats
    G_player_mean = np.zeros(G_PLAYER_DIM, dtype=np.float32)
    G_player_std = np.ones(G_PLAYER_DIM, dtype=np.float32)
    if combined_mask_tr.sum() > 0:
        valid_p = G_pl_tr_raw[combined_mask_tr].reshape(-1, G_PLAYER_DIM)
        G_player_mean = valid_p.mean(axis=0).astype(np.float32)
        G_player_std = (valid_p.std(axis=0) + 1e-6).astype(np.float32)
        G_pl_tr_raw[combined_mask_tr] = (G_pl_tr_raw[combined_mask_tr] - G_player_mean) / G_player_std
        G_pl_va_raw[combined_mask_va] = (G_pl_va_raw[combined_mask_va] - G_player_mean) / G_player_std

    # Normalize standard features
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

    G_game_tr = torch.tensor(G_game_tr_raw, dtype=torch.float32)
    G_game_va = torch.tensor(G_game_va_raw, dtype=torch.float32)
    G_pl_tr = torch.tensor(G_pl_tr_raw, dtype=torch.float32)
    G_pl_va = torch.tensor(G_pl_va_raw, dtype=torch.float32)
    mask_tr_t = torch.tensor(combined_mask_tr, dtype=torch.bool)
    mask_va_t = torch.tensor(combined_mask_va, dtype=torch.bool)

    train_loader = DataLoader(
        NBADatasetPlayerG(X_team_tr, X_pl_tr, Y_tr, weights_tr, lines_tr,
                          G_game_tr, G_pl_tr, mask_tr_t),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        NBADatasetPlayerG(X_team_val, X_pl_val, Y_val, weights_val, lines_val,
                          G_game_va, G_pl_va, mask_va_t),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    norm_stats = {
        "Y_mean": Y_mean, "Y_std": Y_std,
        "Xt_mean": Xt_mean, "Xt_std": Xt_std,
        "Xp_mean": Xp_mean, "Xp_std": Xp_std,
        "G_game_mean": torch.tensor(G_game_mean), "G_game_std": torch.tensor(G_game_std),
        "G_player_mean": torch.tensor(G_player_mean), "G_player_std": torch.tensor(G_player_std),
    }
    return train_loader, val_loader, norm_stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def masked_nll(mu_pred, logvar_pred, target, weights):
    nll = 0.5 * (logvar_pred + (target - mu_pred).pow(2) / logvar_pred.exp())
    nll = nll * weights.unsqueeze(-1)
    return nll.sum() / (weights.sum() * mu_pred.size(-1) + 1e-8)


def train_epoch(decoder, optimizer, loader, device="cpu"):
    decoder.train()
    total_loss = 0.0; n = 0
    for X_t, X_p, Y, weights, _, G_game, G_player, G_mask in loader:
        X_p = X_p.to(device); Y = Y.to(device)
        weights = weights.to(device)
        G_game = G_game.to(device); G_player = G_player.to(device)
        G_mask = G_mask.to(device)
        if G_mask.sum() == 0: continue
        # Only train on games with valid G
        X_p_m = X_p[G_mask]; Y_m = Y[G_mask]; w_m = weights[G_mask]
        G_g_m = G_game[G_mask]; G_p_m = G_player[G_mask]
        optimizer.zero_grad()
        mu_pred, logvar_pred = decoder(G_g_m, G_p_m, X_p_m)
        loss = masked_nll(mu_pred, logvar_pred, Y_m, w_m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


def eval_epoch(decoder, loader, device="cpu"):
    decoder.eval()
    total_loss = 0.0; n = 0
    sigma_all = []; mu_all = []
    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G_game, G_player, G_mask in loader:
            X_p = X_p.to(device); Y = Y.to(device)
            weights = weights.to(device)
            G_game = G_game.to(device); G_player = G_player.to(device)
            G_mask = G_mask.to(device)
            if G_mask.sum() == 0: continue
            X_p_m = X_p[G_mask]; Y_m = Y[G_mask]; w_m = weights[G_mask]
            G_g_m = G_game[G_mask]; G_p_m = G_player[G_mask]
            mu_pred, logvar_pred = decoder(G_g_m, G_p_m, X_p_m)
            loss = masked_nll(mu_pred, logvar_pred, Y_m, w_m)
            total_loss += loss.item(); n += 1
            active = X_p_m.abs().sum(-1) > 0
            sigma_all.extend((0.5*logvar_pred).exp()[active].reshape(-1).cpu().tolist())
            mu_all.extend(mu_pred[active].reshape(-1).cpu().tolist())
    return {
        "nll": total_loss / max(n, 1),
        "sigma_pred": np.mean(sigma_all) if sigma_all else 0.0,
        "mu_std": np.std(mu_all) if mu_all else 0.0,
    }


def compute_p_over_std(decoder, loader, sigma_G_game=0.6, sigma_G_player=0.5,
                        n_samples=100, device="cpu"):
    """Simulate P(over|G) std by perturbing G with Gaussian noise."""
    decoder.eval()
    all_stds = []
    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G_game, G_player, G_mask in loader:
            X_p = X_p.to(device)
            G_game = G_game.to(device); G_player = G_player.to(device)
            G_mask = G_mask.to(device)
            if G_mask.sum() < 2: continue
            X_p_m = X_p[G_mask]; G_g_m = G_game[G_mask]; G_p_m = G_player[G_mask]
            p_list = []
            for _ in range(n_samples):
                G_g_n = G_g_m + torch.randn_like(G_g_m) * sigma_G_game
                G_p_n = G_p_m + torch.randn_like(G_p_m) * sigma_G_player
                mu_p, lv_p = decoder(G_g_n, G_p_n, X_p_m)
                sp = (0.5*lv_p).exp()
                p_over = torch.distributions.Normal(0,1).cdf(mu_p/(sp+1e-8))
                p_list.append(p_over.cpu())
            p_stack = torch.stack(p_list).std(dim=0)
            active = X_p_m.abs().sum(-1) > 0
            all_stds.extend(p_stack[active.unsqueeze(-1).expand_as(p_stack)].tolist())
            if len(all_stds) > 50000: break
    return float(np.mean(all_stds)) if all_stds else 0.0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_v7")
    p.add_argument("--season", type=str, default="2019-26")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--diag-every", type=int, default=10)
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)
    log_file = ckpt_dir / "train_log.txt"
    ckpt_path = ckpt_dir / "decoder_player_g.pt"
    DEVICE = "cpu"

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log(f"Player-specific G decoder | season={args.season} | epochs={args.epochs}")
    log(f"G_GAME_DIM={G_GAME_DIM} | G_PLAYER_DIM={G_PLAYER_DIM} | MAX_PLAYERS={MAX_PLAYERS}")

    log("Loading data...")
    df = load_processed(season_suffix=args.season)
    train_df, val_df = temporal_split(df)
    log(f"  train={len(train_df)}  val={len(val_df)}")

    log("Building loaders...")
    train_loader, val_loader, norm_stats = make_loaders_player_g(
        train_df, val_df, batch_size=64, raw_dir=RAW_DIR,
    )

    decoder = PlayerGDecoder(
        g_game_dim=G_GAME_DIM, g_player_dim=G_PLAYER_DIM,
        player_dim=PLAYER_DIM, h_dim=64, output_dim=OUTPUT_DIM, dropout=0.3,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    start_epoch = 0
    if not args.no_resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"Resumed from epoch {start_epoch - 1}")

    log(f"Training {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        tr_nll = train_epoch(decoder, optimizer, train_loader, DEVICE)
        va_metrics = eval_epoch(decoder, val_loader, DEVICE)

        log(f"  Epoch {epoch:03d}/{args.epochs-1} | "
            f"train NLL {tr_nll:.4f} | val NLL {va_metrics['nll']:.4f} | "
            f"sigma_pred={va_metrics['sigma_pred']:.4f} | mu_std={va_metrics['mu_std']:.4f}")

        if args.diag_every > 0 and (epoch + 1) % args.diag_every == 0:
            p_std = compute_p_over_std(decoder, val_loader, n_samples=50, device=DEVICE)
            log(f"    [DIAG] P(over|G) std={p_std:.4f}  (target > 0.097 for phi > 0.15)")

        torch.save({
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "norm_stats": norm_stats,
        }, ckpt_path)

    log("Training complete.")
    log(f"Checkpoint: {ckpt_path}")

    # Final diagnostics
    log("\n=== Final Diagnostics ===")
    va = eval_epoch(decoder, val_loader, DEVICE)
    log(f"Val NLL={va['nll']:.4f}  sigma_pred={va['sigma_pred']:.4f}  mu_std={va['mu_std']:.4f}")

    p_std_game = compute_p_over_std(decoder, val_loader, sigma_G_game=0.6, sigma_G_player=0.0, n_samples=100)
    p_std_player = compute_p_over_std(decoder, val_loader, sigma_G_game=0.0, sigma_G_player=0.5, n_samples=100)
    p_std_both = compute_p_over_std(decoder, val_loader, sigma_G_game=0.6, sigma_G_player=0.5, n_samples=200)
    log(f"P(over|G) std:")
    log(f"  G_game noise only (sigma=0.6):        {p_std_game:.4f}")
    log(f"  G_player noise only (sigma=0.5):      {p_std_player:.4f}")
    log(f"  Both (sigma_game=0.6, sigma_pl=0.5):  {p_std_both:.4f}")
    log(f"  Target for phi > 0.15: > 0.097")

    phi_max = 4 * p_std_both**2 / 0.25
    log(f"  Theoretical phi_max ≈ {phi_max:.4f}")
