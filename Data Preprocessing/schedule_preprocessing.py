import pandas as pd
import os
from datetime import datetime

# Function to convert the date format to a pandas datetime object
def convert_date(date_str):
    # Remove any doubleheader notation (e.g., '(1)' or '(2)')
    date_str = date_str.split(' (')[0]
    current_year = datetime.now().year
    date_str_with_year = f"{date_str}, {current_year}"
    return pd.to_datetime(date_str_with_year, format='%A, %b %d, %Y')

# Function to calculate win rate, average runs per game, and average opponent runs per game for the last 5 games
def calculate_stats(df):
    # Convert the 'Date' column to datetime
    df['Date'] = df['Date'].apply(convert_date)
    # Filter out future games
    df = df[df['Date'] <= datetime.now()]
    # Sort the dataframe by date
    df = df.sort_values(by='Date', ascending=False)
    # Get the last 5 games
    last_5_games = df.head(5)
    # Calculate the win rate
    wins = last_5_games['W/L'].str.count('W').sum()
    win_rate = wins / 5
    # Calculate average runs per game
    avg_runs = last_5_games['R'].mean()
    # Calculate average opponent runs per game
    avg_opp_runs = last_5_games['RA'].mean()
    return win_rate, avg_runs, avg_opp_runs

# Directory containing the CSV files
directory = 'Team Schedules'

# Dictionary to store stats
stats = {}

# Iterate over each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        # Read the CSV file
        df = pd.read_csv(filepath)
        # Calculate the stats
        team_name = filename.split('_')[0].upper()
        win_rate, avg_runs, avg_opp_runs = calculate_stats(df)
        stats[team_name] = {
            'WinRate': win_rate,
            'AvgRuns': avg_runs,
            'AvgOppRuns': avg_opp_runs
        }

# Read the team_stats CSV
team_stats_filepath = 'combined_team_stats.csv'
team_stats_df = pd.read_csv(team_stats_filepath)

# Strip whitespace from column names
team_stats_df.columns = team_stats_df.columns.str.strip()

# Debug: Print the columns of the DataFrame
print("Columns in team_stats_df:", team_stats_df.columns)

# Check if 'Team' column exists before applying updates
if 'Team' in team_stats_df.columns:
    # Add the stats to the team_stats DataFrame
    team_stats_df['L5%'] = team_stats_df['Team'].apply(lambda team: stats.get(team.upper(), {}).get('WinRate', 0))
    team_stats_df['AvgRuns'] = team_stats_df['Team'].apply(lambda team: stats.get(team.upper(), {}).get('AvgRuns', 0))
    team_stats_df['AvgOppRuns'] = team_stats_df['Team'].apply(lambda team: stats.get(team.upper(), {}).get('AvgOppRuns', 0))
    

    # Save the updated DataFrame to a new CSV file
    team_stats_df.to_csv('combined_team_stats.csv', index=False)

    print("Win rates, average runs per game, and average opponent runs per game added to combined_team_stats.csv successfully!")
else:
    print("Error: 'Team' column not found in the team_stats_df DataFrame.")
