import pandas as pd
import os

# --- Configuration ---
RAW_DIR = "raw_data"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This list contains the standard abbreviations for all 30 MLB teams.
ALL_TEAMS = [
    'ATL', 'MIA', 'NYM', 'PHI', 'WSN', 'CHC', 'CIN', 'MIL', 'PIT', 'STL',
    'ARI', 'COL', 'LAD', 'SDP', 'SFG', 'BAL', 'BOS', 'NYY', 'TBR', 'TOR',
    'CHW', 'CLE', 'DET', 'KCR', 'MIN', 'HOU', 'LAA', 'ATH', 'SEA', 'TEX'
]

def validate_data_completeness(df, all_teams, stat_type):
    """
    Checks if the DataFrame has an entry for every team for every year present in the data.
    Reports any missing combinations.
    """
    print(f"\n--- Validating {stat_type} Data Completeness ---")
    is_complete = True
    
    # Find all unique years present in the data
    years_in_data = sorted(df['year'].unique())
    
    # Create a master DataFrame of all expected Year/Team combinations
    expected_index = pd.MultiIndex.from_product(
        [years_in_data, all_teams], 
        names=['year', 'Team']
    )
    expected_df = pd.DataFrame(index=expected_index).reset_index()

    # Get the actual combinations present in the data
    actual_combinations = df[['year', 'Team']].drop_duplicates()
    
    # Merge to find what's missing
    merged = pd.merge(expected_df, actual_combinations, on=['year', 'Team'], how='left', indicator=True)
    
    missing = merged[merged['_merge'] == 'left_only']
    
    if not missing.empty:
        is_complete = False
        print(f"⚠️  Warning: Missing {stat_type} data for the following Year/Team combinations:")
        missing_by_team = missing.groupby('Team')['year'].apply(list)
        for team, years in missing_by_team.items():
            print(f"  - Team: {team}, Missing Years: {years}")
    else:
        print(f"✅ Data is complete for all teams for all years found ({years_in_data}).")
        
    return is_complete

def process_and_aggregate(raw_df, weight_col, stats_to_keep):
    """Aggregates player-level data to the team level."""
    agg_dict = {}
    for stat, method in stats_to_keep.items():
        if method == 'weighted_avg':
            # Ensure columns are numeric before multiplying
            raw_df[stat] = pd.to_numeric(raw_df[stat], errors='coerce')
            raw_df[weight_col] = pd.to_numeric(raw_df[weight_col], errors='coerce')
            raw_df[f'{stat}_x_{weight_col}'] = raw_df[stat] * raw_df[weight_col]
            agg_dict[f'{stat}_x_{weight_col}'] = 'sum'
        elif method == 'sum':
            agg_dict[stat] = 'sum'
    agg_dict[weight_col] = 'sum'

    team_df = raw_df.groupby(['year', 'Team']).agg(agg_dict).reset_index()

    for stat, method in stats_to_keep.items():
        if method == 'weighted_avg':
            # Avoid division by zero
            team_df[stat] = team_df[f'{stat}_x_{weight_col}'].divide(team_df[weight_col]).fillna(0)
            team_df.drop(columns=[f'{stat}_x_{weight_col}'], inplace=True)
            
    return team_df

# --- Main Execution ---
if __name__ == "__main__":
    # --- Process Hitting Data ---
    print("--- Aggregating Hitting Data ---")
    raw_hitting_df = pd.read_csv(os.path.join(RAW_DIR, 'batting_data.csv'))
    
    # **FIX: Rename 'Season' to 'year' immediately after loading**
    if 'Season' in raw_hitting_df.columns:
        raw_hitting_df.rename(columns={'Season': 'year'}, inplace=True)

    validate_data_completeness(raw_hitting_df, ALL_TEAMS, "Hitting")
    
    hitting_stats_to_keep = {
        'wOBA': 'weighted_avg', 'K%': 'weighted_avg', 'BB%': 'weighted_avg',
        'HR': 'sum', 'Barrel%': 'weighted_avg', 'HardHit%': 'weighted_avg'
    }
    team_hitting_stats = process_and_aggregate(
        raw_df=raw_hitting_df, weight_col='PA', stats_to_keep=hitting_stats_to_keep
    )
    hitting_output_path = os.path.join(OUTPUT_DIR, 'team_hitting_stats.csv')
    team_hitting_stats.to_csv(hitting_output_path, index=False)
    print(f"\n✅ Team hitting stats saved to '{hitting_output_path}'")

    # --- Process Pitching Data ---
    print("\n--- Aggregating Pitching Data ---")
    raw_pitching_df = pd.read_csv(os.path.join(RAW_DIR, 'pitching_data.csv'))

    # **FIX: Rename 'Season' to 'year' immediately after loading**
    if 'Season' in raw_pitching_df.columns:
        raw_pitching_df.rename(columns={'Season': 'year'}, inplace=True)

    validate_data_completeness(raw_pitching_df, ALL_TEAMS, "Pitching")
    
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
    print(f"\n✅ Team pitching stats saved to '{pitching_output_path}'")