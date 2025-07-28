import pandas as pd
import os

# --- Configuration ---
RAW_DIR = "raw_data"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_aggregate(raw_df, weight_col, stats_to_keep):
    """Aggregates player-level data to the team level."""
    raw_df = raw_df.rename(columns={'Name': 'player_name', 'Season': 'year'})
    agg_dict = {}
    for stat, method in stats_to_keep.items():
        if method == 'weighted_avg':
            raw_df[f'{stat}_x_{weight_col}'] = raw_df[stat] * raw_df[weight_col]
            agg_dict[f'{stat}_x_{weight_col}'] = 'sum'
        elif method == 'sum':
            agg_dict[stat] = 'sum'
    agg_dict[weight_col] = 'sum'

    # Use the more compatible .agg(agg_dict) syntax
    team_df = raw_df.groupby(['year', 'Team']).agg(agg_dict).reset_index()

    for stat, method in stats_to_keep.items():
        if method == 'weighted_avg':
            team_df[stat] = team_df[f'{stat}_x_{weight_col}'] / team_df[weight_col]
            team_df.drop(columns=[f'{stat}_x_{weight_col}'], inplace=True)
            
    return team_df

# --- Main Execution ---
if __name__ == "__main__":
    # --- Process Hitting Data ---
    print("--- Aggregating Hitting Data ---")
    raw_hitting_df = pd.read_csv(os.path.join(RAW_DIR, 'batting_data.csv'))
    hitting_stats_to_keep = {
        'wOBA': 'weighted_avg', 'K%': 'weighted_avg', 'BB%': 'weighted_avg',
        'HR': 'sum', 'Barrel%': 'weighted_avg', 'HardHit%': 'weighted_avg'
    }
    team_hitting_stats = process_and_aggregate(
        raw_df=raw_hitting_df, weight_col='PA', stats_to_keep=hitting_stats_to_keep
    )
    hitting_output_path = os.path.join(OUTPUT_DIR, 'team_hitting_stats.csv')
    team_hitting_stats.to_csv(hitting_output_path, index=False)
    print(f"✅ Team hitting stats saved to '{hitting_output_path}'")

    # --- Process Pitching Data ---
    print("\n--- Aggregating Pitching Data ---")
    raw_pitching_df = pd.read_csv(os.path.join(RAW_DIR, 'pitching_data.csv'))
    pitching_stats_to_keep = {
        'FIP': 'weighted_avg', 'xFIP': 'weighted_avg', 'K/9': 'weighted_avg',
        'BB/9': 'weighted_avg', 'K-BB%': 'weighted_avg', 'HR/9': 'weighted_avg',
        'Barrel%': 'weighted_avg', 'HardHit%': 'weighted_avg', 'HR': 'sum'
    }
    team_pitching_stats = process_and_aggregate(
        raw_df=raw_pitching_df, weight_col='TBF', stats_to_keep=pitching_stats_to_keep
    )
    pitching_output_path = os.path.join(OUTPUT_DIR, 'team_pitching_stats.csv')
    team_pitching_stats.to_csv(pitching_output_path, index=False)
    print(f"✅ Team pitching stats saved to '{pitching_output_path}'")