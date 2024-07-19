import pandas as pd
import os
from datetime import datetime
from pandas import DataFrame

# Function to convert the date format to a pandas datetime object
def convert_date(date_str):
    if isinstance(date_str, str):
        date_str = date_str.split(' (')[0]
        current_year = datetime.now().year
        date_str_with_year = f"{date_str}, {current_year}"
        return pd.to_datetime(date_str_with_year, format='%A, %b %d, %Y')
    return date_str

# Directory containing the CSV files
directory = 'Team Schedules'

# List to store matchups for past games
past_matchups = []

# Read the team_stats CSV
team_stats_filepath = 'combined_team_stats.csv'
team_stats_df = pd.read_csv(team_stats_filepath)
team_stats_df.columns = team_stats_df.columns.str.strip()
team_stats_df['Team'] = team_stats_df['Team'].str.upper().str.strip()

# Iterate over each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        df['Date'] = df['Date'].apply(convert_date)
        past_games = df[df['Date'] <= datetime.now()]

        for index, game in past_games.iterrows():
            home_team = filename.split('_')[0].upper().strip() if game['Home_Away'] == 'Home' else game['Opp'].upper().strip()
            away_team = game['Opp'].upper().strip() if game['Home_Away'] == 'Home' else filename.split('_')[0].upper().strip()

            home_stats = team_stats_df[team_stats_df['Team'] == home_team]
            away_stats = team_stats_df[team_stats_df['Team'] == away_team]

            if not home_stats.empty and not away_stats.empty:
                home_stats = home_stats.add_prefix('Home_').iloc[0]
                away_stats = away_stats.add_prefix('Away_').iloc[0]

                result = game['W/L']
                home_win = 1 if result == 'W' and game['Home_Away'] == 'Home' else 0

                combined_stats = pd.concat([home_stats, away_stats])
                combined_stats['Home'] = home_team
                combined_stats['Away'] = away_team
                combined_stats['Home_Win'] = home_win

                past_matchups.append(combined_stats)

# Create DataFrame for past matchups
past_matchups_df = pd.DataFrame(past_matchups)

# Save the past matchups DataFrame to CSV
past_matchups_df.to_csv('2024_Training_Data.csv', index=False)

print("Past matchups with outcomes added to 2024_Training_Data.csv successfully!")
