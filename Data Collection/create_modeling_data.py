import pandas as pd
import statsapi
import time
import os

# --- Configuration ---
YEARS = [2022, 2023, 2024, 2025]
PROCESSED_DIR = "processed_data"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_schedules(years):
    """
    Gets game schedules and results using the mlb-statsapi library.
    Now includes completed games and games scheduled for the current day.
    """
    print("Fetching game schedules from MLB Stats API via mlb-statsapi...")
    all_games_data = []
    
    for year in years:
        try:
            print(f"  Getting {year} schedule...")
            games_for_year = statsapi.schedule(season=year)
            all_games_data.extend(games_for_year)
            time.sleep(1) # Be polite to the server
        except Exception as e:
            print(f"  Could not get schedule for {year}. Reason: {e}")

    if not all_games_data:
        print("Error: No schedules were downloaded.")
        return pd.DataFrame()

    full_schedule = pd.DataFrame(all_games_data)

    # --- Data Cleaning ---
    today_str = pd.to_datetime('today').strftime('%Y-%m-%d')
    final_games_mask = full_schedule['status'] == 'Final'
    todays_games_mask = full_schedule['game_date'] == today_str
    
    full_schedule = full_schedule[final_games_mask | todays_games_mask].copy()
    
    full_schedule['Year'] = pd.to_datetime(full_schedule['game_date']).dt.year
    
    full_schedule.rename(columns={
        'home_name': 'HomeTeam_Name', 'away_name': 'AwayTeam_Name', 
        'home_score': 'HomeRuns', 'away_score': 'AwayRuns'
    }, inplace=True)
    
    full_schedule['HomeRuns'] = pd.to_numeric(full_schedule['HomeRuns'], errors='coerce')
    full_schedule['AwayRuns'] = pd.to_numeric(full_schedule['AwayRuns'], errors='coerce')

    full_schedule['RunDifferential'] = full_schedule['HomeRuns'] - full_schedule['AwayRuns']
    
    games_df = full_schedule[['Year', 'game_date', 'HomeTeam_Name', 'AwayTeam_Name', 'HomeRuns', 'AwayRuns', 'RunDifferential']].reset_index(drop=True)
    
    print(f"✅ Successfully processed {len(games_df)} games (final and scheduled for today).")
    return games_df

# --- Main Execution ---
if __name__ == "__main__":

    schedule_df = get_all_schedules(YEARS)

    if not schedule_df.empty:
        try:
            team_hitting = pd.read_csv(os.path.join(PROCESSED_DIR, 'team_hitting_stats.csv'))
            team_pitching = pd.read_csv(os.path.join(PROCESSED_DIR, 'team_pitching_stats.csv'))
            mlb_odds = pd.read_csv(os.path.join(PROCESSED_DIR, 'mlb_odds.csv'))
            
        except FileNotFoundError as e:
            print(f"ERROR: Make sure your aggregated stat files are in the '{PROCESSED_DIR}' folder.")
            print(f"Missing file: {e.filename}")
            exit()

        # --- Team Name Mappings ---
        team_name_map = {
            'Atlanta Braves': 'ATL', 'Miami Marlins': 'MIA', 'New York Mets': 'NYM', 'Philadelphia Phillies': 'PHI',
            'Washington Nationals': 'WSH', 'Chicago Cubs': 'CHC', 'Cincinnati Reds': 'CIN', 'Milwaukee Brewers': 'MIL',
            'Pittsburgh Pirates': 'PIT', 'St. Louis Cardinals': 'STL', 
            'Arizona Diamondbacks': 'ARI', # **FIX: Corrected 'Arizona D-backs' to 'Arizona Diamondbacks'**
            'Colorado Rockies': 'COL',
            'Los Angeles Dodgers': 'LAD', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF', 'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS', 'New York Yankees': 'NYY', 'Tampa Bay Rays': 'TB', 'Toronto Blue Jays': 'TOR',
            'Chicago White Sox': 'CWS', 'Cleveland Guardians': 'CLE', 'Detroit Tigers': 'DET', 'Kansas City Royals': 'KC',
            'Minnesota Twins': 'MIN', 'Houston Astros': 'HOU', 'Los Angeles Angels': 'LAA', 'Oakland Athletics': 'OAK',
            'Seattle Mariners': 'SEA', 'Texas Rangers': 'TEX'
        }
        
        odds_team_name_map = {
            'Baltimore': 'BAL', 'Toronto': 'TOR', 'Arizona': 'ARI', 'Detroit': 'DET', 'Boston': 'BOS',
            'Minnesota': 'MIN', 'Washington': 'WSH', 'Houston': 'HOU', 'Atlanta': 'ATL', 'Kansas City': 'KC',
            'Chi. Cubs': 'CHC', 'Milwaukee': 'MIL', 'Philadelphia': 'PHI', 'Chi. White Sox': 'CWS',
            'Pittsburgh': 'PIT', 'San Francisco': 'SF', 'NY Mets': 'NYM', 'San Diego': 'SD', 'Colorado': 'COL',
            'Cleveland': 'CLE', 'Tampa Bay': 'TB', 'NY Yankees': 'NYY', 'LA Dodgers': 'LAD', 'Cincinnati': 'CIN',
            'Miami': 'MIA', 'St. Louis': 'STL', 'Texas': 'TEX', 'LA Angels': 'LAA', 'Seattle': 'SEA', 
            'Oakland': 'OAK' # **FIX: Changed 'Oakland' to 'Athletics' to match scraped data**
        }

        # --- Prepare DataFrames ---
        print("Preparing and mapping team names...")
        schedule_df['HomeTeam'] = schedule_df['HomeTeam_Name'].map(team_name_map)
        schedule_df['AwayTeam'] = schedule_df['AwayTeam_Name'].map(team_name_map)
        
        mlb_odds['game_date'] = pd.to_datetime(mlb_odds['Game Time'].str.replace(' ET', '', regex=False)).dt.strftime('%Y-%m-%d')
        mlb_odds['HomeTeam'] = mlb_odds['Home Team'].map(odds_team_name_map)
        mlb_odds['AwayTeam'] = mlb_odds['Away Team'].map(odds_team_name_map)

        # Prepare team stats data
        team_hitting.columns = ['year', 'Team'] + [f'off_{col}' for col in team_hitting.columns[2:]]
        team_pitching.columns = ['year', 'Team'] + [f'pch_{col}' for col in team_pitching.columns[2:]]
        team_stats = pd.merge(team_hitting, team_pitching, on=['year', 'Team'])
        
        # --- Merge All DataFrames ---
        print("Merging all data sources...")
        final_df = pd.merge(schedule_df, team_stats, left_on=['Year', 'HomeTeam'], right_on=['year', 'Team'], how='left')
        final_df = pd.merge(final_df, team_stats, left_on=['Year', 'AwayTeam'], right_on=['year', 'Team'], how='left', suffixes=('_home', '_away'))
        
        odds_to_merge = mlb_odds[['game_date', 'HomeTeam', 'AwayTeam', 'Home Wager %', 'Away Wager %', 'Home Opener Odds', 'Away Opener Odds']]
        final_df = pd.merge(final_df, odds_to_merge, on=['game_date', 'HomeTeam', 'AwayTeam'], how='left')

        final_df.drop_duplicates(subset=['game_date', 'HomeTeam_Name', 'AwayTeam_Name'], inplace=True)

        # --- Final Cleanup ---
        final_df.drop(columns=['year_home', 'Team_home', 'year_away', 'Team_away'], inplace=True, errors='ignore')
        final_df.dropna(subset=['off_HR_home', 'pch_HR_away'], inplace=True)

        output_path = os.path.join(OUTPUT_DIR, 'mlb_modeling_dataset.csv')
        final_df.to_csv(output_path, index=False)
        
        print("\n--- Final Dataset Created! ---")
        print(f"✅ Modeling dataset with {len(final_df)} games saved to '{output_path}'")
        pd.set_option('display.max_columns', None)
        print("\n--- Sample of the Final DataFrame ---")
        print(final_df.head())