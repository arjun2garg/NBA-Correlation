import pandas as pd
import numpy as np

stats = pd.read_csv('archive/PlayerStatistics.csv')
season_mask_24 = (stats['gameDateTimeEst'] >= '2024-10-22') & (stats['gameDateTimeEst'] <= '2025-04-14')
season_stats = stats[season_mask_24]
season_stats = season_stats[season_stats['numMinutes'].notna()]
season_stats = season_stats.drop(columns=['gameLabel', 'gameSubLabel', 'seriesGameNumber'])

season_stats["gameDateTimeEst"] = pd.to_datetime(
    season_stats["gameDateTimeEst"],
    utc=True,
    errors="coerce"
)

def exp_time_decay_feature(
    df,
    stat_col,
    time_col="gameDateTimeEst",
    player_col="personId",
    beta=0.99
):
    out = []

    for _, g in df.groupby(player_col, sort=False):
        times = g[time_col].values
        values = g[stat_col].values

        hist_vals = []

        for i in range(len(g)):
            if i == 0:
                hist_vals.append(np.nan)
                continue

            # days since each past game
            days_ago = (
                (times[i] - times[:i])
                / np.timedelta64(1, "D")
            )

            weights = beta ** days_ago
            weighted_avg = np.sum(values[:i] * weights) / np.sum(weights)

            hist_vals.append(weighted_avg)

        out.extend(hist_vals)

    return out

season_stats = season_stats.sort_values(['personId', 'gameDateTimeEst'])
stats_to_decay = ['points', 'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted', 'threePointersMade',
                  'freeThrowsAttempted', 'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal', 'foulsPersonal', 'turnovers', 
                  'numMinutes']
for stat in stats_to_decay:
    season_stats['h_' + stat] = exp_time_decay_feature(season_stats, stat_col=stat, beta=0.99)

season_stat_target = season_stats[['personId', 'gameId', 'home', 'points', 'assists', 'reboundsTotal']]
season_stat_input = season_stats[['personId', 'gameId', 'home', 'gameDateTimeEst', 'h_points', 'h_assists', 'h_blocks', 'h_steals', 'h_fieldGoalsAttempted', 'h_fieldGoalsMade', 
                                  'h_threePointersAttempted', 'h_threePointersMade', 'h_freeThrowsAttempted', 'h_freeThrowsMade', 'h_reboundsDefensive', 
                                  'h_reboundsOffensive', 'h_reboundsTotal', 'h_foulsPersonal', 'h_turnovers', 'h_numMinutes']]

season_stat_target.to_csv('cleaned_data/target_data_2024-25.csv', index=False)
season_stat_input.to_csv('cleaned_data/input_data_2024-25.csv', index=False)