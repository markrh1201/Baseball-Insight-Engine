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
    full_schedule = full_schedule[full_schedule['status'] == 'Final'].copy()
    full_schedule['Year'] = pd.to_datetime(full_schedule['game_date']).dt.year
    
    full_schedule.rename(columns={
        'home_name': 'HomeTeam_Name', 'away_name': 'AwayTeam_Name', 
        'home_score': 'HomeRuns', 'away_score': 'AwayRuns'
    }, inplace=True)
    
    full_schedule['RunDifferential'] = full_schedule['HomeRuns'] - full_schedule['AwayRuns']
    
    games_df = full_schedule[['Year', 'HomeTeam_Name', 'AwayTeam_Name', 'HomeRuns', 'AwayRuns', 'RunDifferential']].reset_index(drop=True)
    
    print(f"✅ Successfully processed {len(games_df)} games.")
    return games_df

# --- Main Execution ---
if __name__ == "__main__":
    schedule_df = get_all_schedules(YEARS)

    if not schedule_df.empty:
        try:
            team_hitting = pd.read_csv(os.path.join(PROCESSED_DIR, 'team_hitting_stats.csv'))
            team_pitching = pd.read_csv(os.path.join(PROCESSED_DIR, 'team_pitching_stats.csv'))
        except FileNotFoundError as e:
            print(f"ERROR: Make sure your aggregated stat files are in the '{PROCESSED_DIR}' folder.")
            print(f"Missing file: {e.filename}")
            exit()

        team_name_map = {
            'Atlanta Braves': 'ATL', 'Miami Marlins': 'MIA', 'New York Mets': 'NYM', 'Philadelphia Phillies': 'PHI',
            'Washington Nationals': 'WSH', 'Chicago Cubs': 'CHC', 'Cincinnati Reds': 'CIN', 'Milwaukee Brewers': 'MIL',
            'Pittsburgh Pirates': 'PIT', 'St. Louis Cardinals': 'STL', 'Arizona D-backs': 'ARI', 'Colorado Rockies': 'COL',
            'Los Angeles Dodgers': 'LAD', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF', 'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS', 'New York Yankees': 'NYY', 'Tampa Bay Rays': 'TB', 'Toronto Blue Jays': 'TOR',
            'Chicago White Sox': 'CWS', 'Cleveland Guardians': 'CLE', 'Detroit Tigers': 'DET', 'Kansas City Royals': 'KC',
            'Minnesota Twins': 'MIN', 'Houston Astros': 'HOU', 'Los Angeles Angels': 'LAA', 'Oakland Athletics': 'OAK',
            'Seattle Mariners': 'SEA', 'Texas Rangers': 'TEX'
        }
        
        schedule_df['HomeTeam'] = schedule_df['HomeTeam_Name'].map(team_name_map)
        schedule_df['AwayTeam'] = schedule_df['AwayTeam_Name'].map(team_name_map)
        schedule_df.dropna(subset=['HomeTeam', 'AwayTeam'], inplace=True)

        team_hitting.columns = ['year', 'Team'] + [f'off_{col}' for col in team_hitting.columns[2:]]
        team_pitching.columns = ['year', 'Team'] + [f'pch_{col}' for col in team_pitching.columns[2:]]
        team_stats = pd.merge(team_hitting, team_pitching, on=['year', 'Team'])

        final_df = pd.merge(schedule_df, team_stats, left_on=['Year', 'HomeTeam'], right_on=['year', 'Team'], how='left')
        final_df = pd.merge(final_df, team_stats, left_on=['Year', 'AwayTeam'], right_on=['year', 'Team'], how='left', suffixes=('_home', '_away'))

        final_df.drop(columns=['year_home', 'Team_home', 'year_away', 'Team_away'], inplace=True)
        final_df.dropna(inplace=True)
        
        output_path = os.path.join(OUTPUT_DIR, 'mlb_modeling_dataset.csv')
        final_df.to_csv(output_path, index=False)
        
        print("\n--- Final Dataset Created! ---")
        print(f"✅ Modeling dataset with {len(final_df)} games saved to '{output_path}'")
        pd.set_option('display.max_columns', None)
        print(final_df.head())