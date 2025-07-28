import pandas as pd
import time
import os
import glob
from pybaseball import batting_stats, pitching_stats

# --- Configuration ---
YEARS = [2022, 2023, 2024, 2025] 
SAVANT_DIR = "savant_data"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def combine_savant_files(file_pattern):
    """Loads and combines all Savant CSVs matching a pattern."""
    full_pattern = os.path.join(SAVANT_DIR, file_pattern)
    file_list = glob.glob(full_pattern)
    if not file_list: return None
    df_list = [pd.read_csv(file) for file in file_list]
    return pd.concat(df_list, ignore_index=True)

def process_and_aggregate(savant_df, mapping_df, weight_col, stats_to_keep):
    """
    Merges Savant data with team mappings and aggregates to the team level.
    """
    merged_df = pd.merge(savant_df, mapping_df, on=['player_name', 'year'], how='left')
    
    merged_df.dropna(subset=['Team'], inplace=True)
    merged_df = merged_df[merged_df['Team'] != '- - -']

    agg_dict = {}
    for stat, method in stats_to_keep.items():
        if method == 'weighted_avg':
            merged_df[f'{stat}_x_{weight_col}'] = merged_df[stat] * merged_df[weight_col]
            # Ensure the value is a string 'sum'
            agg_dict[f'{stat}_x_{weight_col}'] = 'sum'
        elif method == 'sum':
            # Ensure the value is a string 'sum'
            agg_dict[stat] = 'sum'
    
    # Ensure the value is a string 'sum'
    agg_dict[weight_col] = 'sum'

    team_df = merged_df.groupby(['year', 'Team']).agg(**agg_dict).reset_index()

    for stat, method in stats_to_keep.items():
        if method == 'weighted_avg':
            team_df[stat] = team_df[f'{stat}_x_{weight_col}'].divide(team_df[weight_col]).fillna(0)
            team_df.drop(columns=[f'{stat}_x_{weight_col}'], inplace=True)
            
    return team_df

# --- Main Execution ---
if __name__ == "__main__":
    # --- Build Mapping DataFrames using pybaseball ---
    print("--- Getting Player-to-Team Mappings from pybaseball ---")
    start_year, end_year = YEARS[0], YEARS[-1]

    # Get hitting data for all years in one call
    hitting_map = batting_stats(start_year, end_year, qual=200)
    hitting_map.reset_index(inplace=True)
    hitting_map.rename(columns={'Name': 'player_name', 'Season': 'year'}, inplace=True)

    # Get pitching data for all years in one call
    pitching_map = pitching_stats(start_year, end_year, qual=40)
    pitching_map.reset_index(inplace=True)
    pitching_map.rename(columns={'Name': 'player_name', 'Season': 'year'}, inplace=True)
    
    print("✅ Successfully fetched mapping data.")

    # --- Process Hitting Data ---
    print("\n--- Processing Hitting Data ---")
    combined_hitting_df = combine_savant_files("savant_batter_*.csv")
    
    if combined_hitting_df is not None and not hitting_map.empty:
        hitting_stats_to_keep = {
            'woba': 'weighted_avg', 'xwoba': 'weighted_avg', 'k_percent': 'weighted_avg', 
            'bb_percent': 'weighted_avg', 'barrel_batted_rate': 'weighted_avg', 
            'hard_hit_percent': 'weighted_avg', 'home_run': 'sum', 'barrel': 'sum'
        }
        
        team_hitting_stats = process_and_aggregate(
            savant_df=combined_hitting_df, mapping_df=hitting_map,
            weight_col='PA', stats_to_keep=hitting_stats_to_keep
        )
        
        hitting_output_path = os.path.join(OUTPUT_DIR, 'team_hitting_stats.csv')
        team_hitting_stats.to_csv(hitting_output_path, index=False)
        print(f"✅ Team hitting stats saved to '{hitting_output_path}'")
        print(team_hitting_stats.head())

    # --- Process Pitching Data ---
    print("\n--- Processing Pitching Data ---")
    combined_pitching_df = combine_savant_files("savant_pitcher_*.csv")
    
    if combined_pitching_df is not None and not pitching_map.empty:
        pitching_stats_to_keep = {
            'woba': 'weighted_avg', 'xwoba': 'weighted_avg', 'k_percent': 'weighted_avg', 
            'bb_percent': 'weighted_avg', 'barrel_batted_rate': 'weighted_avg', 
            'hard_hit_percent': 'weighted_avg', 'p_total_home_run': 'sum', 'barrel': 'sum'
        }
        
        team_pitching_stats = process_and_aggregate(
            savant_df=combined_pitching_df, mapping_df=pitching_map,
            weight_col='BF', stats_to_keep=pitching_stats_to_keep
        )
        
        pitching_output_path = os.path.join(OUTPUT_DIR, 'team_pitching_stats.csv')
        team_pitching_stats.to_csv(pitching_output_path, index=False)
        print(f"✅ Team pitching stats saved to '{pitching_output_path}'")
        print(team_pitching_stats.head())