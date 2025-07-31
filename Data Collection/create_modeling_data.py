import pandas as pd
import os
from datetime import datetime
import numpy as np

def create_modeling_dataset():
    """
    Combines schedule, stats, and odds data. It separates completed games
    for training and today's upcoming games for testing.
    """
    # --- File Paths ---
    RAW_DATA_DIR = 'raw_data'
    PROCESSED_DATA_DIR = 'processed_data'
    OUTPUT_DIR = 'modeling_data'

    SCHEDULE_PATH = os.path.join(RAW_DATA_DIR, 'schedule_data.csv')
    HITTING_STATS_PATH = os.path.join(PROCESSED_DATA_DIR, 'team_hitting_stats.csv')
    PITCHING_STATS_PATH = os.path.join(PROCESSED_DATA_DIR, 'team_pitching_stats.csv')
    ODDS_PATH = os.path.join(PROCESSED_DATA_DIR, 'mlb_odds_2022_present.csv')

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # --- Team Name Mappings ---
    team_name_map = {
        'Atlanta Braves': 'ATL', 'Miami Marlins': 'MIA', 'New York Mets': 'NYM',
        'Philadelphia Phillies': 'PHI', 'Washington Nationals': 'WSN', 'Chicago Cubs': 'CHC',
        'Cincinnati Reds': 'CIN', 'Milwaukee Brewers': 'MIL', 'Pittsburgh Pirates': 'PIT',
        'St. Louis Cardinals': 'STL', 'Arizona Diamondbacks': 'ARI', 'Colorado Rockies': 'COL',
        'Los Angeles Dodgers': 'LAD', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
        'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS', 'New York Yankees': 'NYY',
        'Tampa Bay Rays': 'TBR', 'Toronto Blue Jays': 'TOR', 'Chicago White Sox': 'CHW',
        'Cleveland Guardians': 'CLE', 'Detroit Tigers': 'DET', 'Kansas City Royals': 'KCR',
        'Minnesota Twins': 'MIN', 'Houston Astros': 'HOU', 'Los Angeles Angels': 'LAA',
        'Oakland Athletics': 'ATH', 'Seattle Mariners': 'SEA', 'Texas Rangers': 'TEX'
    }
    odds_team_name_map = {
        'Baltimore': 'BAL', 'Toronto': 'TOR', 'Arizona': 'ARI', 'Detroit': 'DET',
        'Boston': 'BOS', 'Minnesota': 'MIN', 'Washington': 'WSN', 'Houston': 'HOU',
        'Atlanta': 'ATL', 'Kansas City': 'KCR', 'Chi. Cubs': 'CHC', 'Milwaukee': 'MIL',
        'Philadelphia': 'PHI', 'Chi. White Sox': 'CHW', 'Pittsburgh': 'PIT',
        'San Francisco': 'SFG', 'NY Mets': 'NYM', 'San Diego': 'SDP', 'Colorado': 'COL',
        'Cleveland': 'CLE', 'Tampa Bay': 'TBR', 'NY Yankees': 'NYY', 'LA Dodgers': 'LAD',
        'Cincinnati': 'CIN', 'Miami': 'MIA', 'St. Louis': 'STL', 'Texas': 'TEX',
        'LA Angels': 'LAA', 'Seattle': 'SEA', 'Athletics': 'ATH'
    }

    # --- 1. Load and Pre-Process All Data Sources ---
    print("Loading all data sources...")
    try:
        schedule_df = pd.read_csv(SCHEDULE_PATH, engine='python', sep=',')
    except FileNotFoundError:
        print(f"Error: Schedule file not found at {SCHEDULE_PATH}")
        return

    hitting_df = pd.read_csv(HITTING_STATS_PATH)
    pitching_df = pd.read_csv(PITCHING_STATS_PATH)
    odds_df = pd.read_csv(ODDS_PATH)

    # Prepare schedule data (The source of truth for home/away teams)
    schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'])
    schedule_df['year'] = schedule_df['game_date'].dt.year
    schedule_df['home_team'] = schedule_df['home_name'].str.strip().map(team_name_map)
    schedule_df['away_team'] = schedule_df['away_name'].str.strip().map(team_name_map)

    # <<< FIX: Prepare odds data with a temporary, order-independent merge key >>>
    odds_df['game_date'] = pd.to_datetime(pd.to_datetime(odds_df['Game Time']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.date)
    odds_df['odds_home_team'] = odds_df['Home Team'].str.strip().map(odds_team_name_map)
    odds_df['odds_away_team'] = odds_df['Away Team'].str.strip().map(odds_team_name_map)
    odds_df.dropna(subset=['odds_home_team', 'odds_away_team'], inplace=True)
    odds_df['merge_key'] = odds_df.apply(lambda row: '_'.join(sorted([row['odds_home_team'], row['odds_away_team']])) + '_' + row['game_date'].strftime('%Y-%m-%d'), axis=1)

    # --- 2. Split Data into Training (Completed) and Testing (Today's Games) ---
    print("Splitting data into training and testing sets...")
    
    training_schedule = schedule_df[schedule_df['winning_team'].notna()].copy()
    training_schedule['home_team_won'] = (training_schedule['home_score'] > training_schedule['away_score']).astype(int)
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    todays_games_mask = (schedule_df['game_date'].dt.strftime('%Y-%m-%d') == today_str)
    status_mask = schedule_df['status'].isin(['Scheduled', 'Pre-Game', 'Delayed Start: Rain'])
    testing_schedule = schedule_df[todays_games_mask & status_mask].copy()
    if not testing_schedule.empty:
        testing_schedule['home_team_won'] = np.nan

    print(f"Found {len(training_schedule)} completed games for the training set.")
    print(f"Found {len(testing_schedule)} upcoming games for today's testing set.")

    # --- 3. Define a Reusable Merging Function ---
    def merge_game_data(df, hitting_stats, pitching_stats, odds_data):
        if df.empty:
            return pd.DataFrame()
            
        base_cols = ['game_id', 'game_date', 'year', 'home_team', 'away_team', 'home_team_won']
        games = df[base_cols].copy()
        games.dropna(subset=['home_team', 'away_team'], inplace=True)

        # <<< FIX: Create the same temporary merge key on the schedule data >>>
        # This key is order-independent and does NOT change the original home/away columns.
        games['merge_key'] = games.apply(lambda row: '_'.join(sorted([row['home_team'], row['away_team']])) + '_' + row['game_date'].strftime('%Y-%m-%d'), axis=1)

        # Merge hitting and pitching stats
        merged = pd.merge(games, hitting_stats, left_on=['year', 'home_team'], right_on=['year', 'Team'], how='left')
        merged = pd.merge(merged, hitting_stats, left_on=['year', 'away_team'], right_on=['year', 'Team'], how='left', suffixes=('_home_hitting', '_away_hitting'))
        merged = pd.merge(merged, pitching_stats, left_on=['year', 'home_team'], right_on=['year', 'Team'], how='left')
        merged = pd.merge(merged, pitching_stats, left_on=['year', 'away_team'], right_on=['year', 'Team'], how='left', suffixes=('_home_pitching', '_away_pitching'))
        
        # <<< FIX: Merge odds using the temporary key >>>
        final = pd.merge(merged, odds_data, on='merge_key', how='left', suffixes=('', '_odds'))

        # <<< FIX: Assign odds correctly based on the true home team from the schedule >>>
        # This checks if the home team in the schedule matches the home team in the odds file and assigns odds accordingly.
        final['Home Opener Odds'] = np.where(final['home_team'] == final['odds_home_team'], final['Home Opener Odds'], final['Away Opener Odds'])
        final['Away Opener Odds'] = np.where(final['home_team'] == final['odds_home_team'], final['Away Opener Odds'], final['Home Opener Odds'])
        
        # Drop temporary and redundant columns for a clean final dataset
        final.drop(columns=[col for col in final.columns if 'Team_' in str(col) or '_odds' in str(col) or col == 'merge_key'], inplace=True)
        return final

    # --- 4. Process and Save Datasets ---
    for data_type, schedule_data in [('training', training_schedule), ('testing', testing_schedule)]:
        if schedule_data.empty:
            continue
        
        print(f"\nProcessing {data_type} data...")
        final_data = merge_game_data(schedule_data, hitting_df, pitching_df, odds_df)
        
        if not final_data.empty:
            if data_type == 'training':
                initial_rows = len(final_data)
                final_data.dropna(inplace=True) # Drop rows with any missing features for clean training
                print(f"   Dropped {initial_rows - len(final_data)} rows with missing values.")
            
            final_data.reset_index(drop=True, inplace=True)
            output_path = os.path.join(OUTPUT_DIR, f'{data_type}_dataset.csv')
            final_data.to_csv(output_path, index=False)
            print(f"✅ {data_type.capitalize()} data created successfully.")
            print(f"   Saved {len(final_data)} games to: {output_path}")
if __name__ == '__main__':
    create_modeling_dataset()