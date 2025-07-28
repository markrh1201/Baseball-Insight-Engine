import pandas as pd
from pybaseball import schedule_and_record, team_batting, team_pitching, team_fielding
from datetime import datetime

# Define the seasons and teams
seasons = [2023, 2024]
teams = ['BAL', 'BOS', 'NYY', 'TBR', 'TOR', 'CHW', 'CLE', 'DET', 'KCR', 'MIN', 'HOU', 'LAA', 'OAK', 'SEA', 'TEX', 
         'ATL', 'MIA', 'NYM', 'PHI', 'WSN', 'CHC', 'CIN', 'MIL', 'PIT', 'STL', 'ARI', 'COL', 'LAD', 'SDP', 'SFG']

# Initialize an empty list for storing game data
data = []

# Loop through each season
for season in seasons:
    # Fetch team stats for the season
    batting_stats = team_batting(season)
    pitching_stats = team_pitching(season)
    fielding_stats = team_fielding(season)
    
    # Loop through each team to get their schedule
    for team in teams:
        schedule = schedule_and_record(season, team)
        
        # Filter only completed games
        schedule = schedule[schedule['W/L'].notna()]
        
        # Loop through each game in the schedule
        for index, row in schedule.iterrows():
            game_date = row['Date']
            
            if row['Home_Away'] == '@':
                home_team = row['Opp']
                away_team = row['Tm']
            else:
                home_team = row['Tm']
                away_team = row['Opp']
                
            home_runs = row['R']
            away_runs = row['RA']
            run_diff = home_runs - away_runs
            
            # Fetch team stats
            home_batting = batting_stats[batting_stats['Team'] == home_team]
            away_batting = batting_stats[batting_stats['Team'] == away_team]
            
            home_pitching = pitching_stats[pitching_stats['Team'] == home_team]
            away_pitching = pitching_stats[pitching_stats['Team'] == away_team]
            
            home_fielding = fielding_stats[fielding_stats['Team'] == home_team]
            away_fielding = fielding_stats[fielding_stats['Team'] == away_team]
            
            # Collect relevant stats
            data.append({
                'Date': game_date,
                'Home Team': home_team,
                'Away Team': away_team,
                'Home Runs': home_runs,
                'Away Runs': away_runs,
                'Run Differential': run_diff,
                'Home AVG': home_batting['AVG'].values[0],
                'Home ERA': home_pitching['ERA'].values[0],
                'Home FPCT': home_fielding['FPCT'].values[0],
                'Away AVG': away_batting['AVG'].values[0],
                'Away ERA': away_pitching['ERA'].values[0],
                'Away FPCT': away_fielding['FPCT'].values[0],
                # Add other relevant stats as needed
            })

# Convert the collected data to a DataFrame
df = pd.DataFrame(data)

# Save the data to a CSV file
df.to_csv('mlb_run_differential_training_data_2023_2024.csv', index=False)

print("Training data CSV for 2023 and 2024 created successfully!")
