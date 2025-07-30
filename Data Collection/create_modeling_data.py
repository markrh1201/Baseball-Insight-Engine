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
        
    # --- Team Name Mappings (Corrected for consistency) ---
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

    # Prepare schedule data
    schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'])
    schedule_df['year'] = schedule_df['game_date'].dt.year
    schedule_df['home_team'] = schedule_df['home_name'].map(team_name_map)
    schedule_df['away_team'] = schedule_df['away_name'].map(team_name_map)

    # Prepare odds data
    odds_df['game_date'] = pd.to_datetime(odds_df['Game Time'].str.split(' ').str[0])
    odds_df['home_team'] = odds_df['Home Team'].map(odds_team_name_map)
    odds_df['away_team'] = odds_df['Away Team'].map(odds_team_name_map)
    odds_df = odds_df[['game_date', 'home_team', 'away_team', 'Home Opener Odds', 'Away Opener Odds']].copy()

    # --- 2. Split Data into Training (Completed) and Testing (Today's Games) ---
    print("Splitting data into training and testing sets...")
    
    # Training data: All games with a recorded winner
    training_schedule = schedule_df[schedule_df['winning_team'].notna()].copy()
    training_schedule['home_team_won'] = (training_schedule['home_score'] > training_schedule['away_score']).astype(int)
    
    # Testing data: Today's games that are not yet finished
    today_str = datetime.now().strftime('%Y-%m-%d')
    todays_games_mask = (schedule_df['game_date'].dt.strftime('%Y-%m-%d') == today_str)
    status_mask = schedule_df['status'].isin(['Scheduled', 'Pre-Game', 'Delayed Start: Rain'])
    testing_schedule = schedule_df[todays_games_mask & status_mask].copy()
    if not testing_schedule.empty:
        testing_schedule['home_team_won'] = np.nan # Target is unknown

    print(f"Found {len(training_schedule)} completed games for the training set.")
    print(f"Found {len(testing_schedule)} upcoming games for today's testing set.")

    # --- 3. Define a Reusable Merging Function ---
    def merge_game_data(df, hitting_stats, pitching_stats, odds_data):
        if df.empty:
            return pd.DataFrame() # Return empty df if no games
            
        base_cols = ['game_id', 'game_date', 'year', 'home_team', 'away_team', 'home_team_won']
        games = df[base_cols].copy()

        # Merge hitting stats
        merged = pd.merge(games, hitting_stats, left_on=['year', 'home_team'], right_on=['year', 'Team'], how='left')
        merged = pd.merge(merged, hitting_stats, left_on=['year', 'away_team'], right_on=['year', 'Team'], how='left', suffixes=('_home_hitting', '_away_hitting'))
        
        # Merge pitching stats
        merged = pd.merge(merged, pitching_stats, left_on=['year', 'home_team'], right_on=['year', 'Team'], how='left')
        merged = pd.merge(merged, pitching_stats, left_on=['year', 'away_team'], right_on=['year', 'Team'], how='left', suffixes=('_home_pitching', '_away_pitching'))
        
        # Merge odds data
        final = pd.merge(merged, odds_data, on=['game_date', 'home_team', 'away_team'], how='left')

        # Clean up redundant columns
        final = final.drop(columns=[col for col in final.columns if 'Team_' in str(col)])
        return final

    # --- 4. Process and Save Datasets ---
    # Process training data
    training_data = merge_game_data(training_schedule, hitting_df, pitching_df, odds_df)
    if not training_data.empty:
        initial_rows = len(training_data)
        training_data.dropna(inplace=True) # Drop any rows with missing features for clean training
        training_data.reset_index(drop=True, inplace=True)
        
        output_path = os.path.join(OUTPUT_DIR, 'training_dataset.csv')
        training_data.to_csv(output_path, index=False)
        print(f"\n✅ Training data created successfully. Dropped {initial_rows - len(training_data)} rows with missing values.")
        print(f"   Saved to: {output_path}")

    # Process testing data
    testing_data = merge_game_data(testing_schedule, hitting_df, pitching_df, odds_df)
    if not testing_data.empty:
        # For testing data, we keep all games, even if some features (like odds) are missing
        testing_data.reset_index(drop=True, inplace=True)
        
        output_path = os.path.join(OUTPUT_DIR, 'testing_dataset.csv')
        testing_data.to_csv(output_path, index=False)
        print(f"\n✅ Testing data for today's games created successfully.")
        print(f"   Saved to: {output_path}")
        if testing_data.isnull().any().any():
            print("   (Note: Testing data may contain NaNs for features that are not yet available, like odds).")

if __name__ == '__main__':
    create_modeling_dataset()