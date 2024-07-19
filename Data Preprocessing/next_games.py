import pandas as pd
import os
from datetime import datetime
from pandas import DataFrame

# Function to convert the date format to a pandas datetime object
def convert_date(date_str):
    if isinstance(date_str, str):  # Only apply conversion if the date is a string
        # Remove any doubleheader notation (e.g., '(1)' or '(2)')
        date_str = date_str.split(' (')[0]
        current_year = datetime.now().year
        date_str_with_year = f"{date_str}, {current_year}"
        return pd.to_datetime(date_str_with_year, format='%A, %b %d, %Y')
    return date_str  # Return the date as is if it's not a string

# Function to find the next game for a team
def find_next_game(df: DataFrame):
    # Convert the 'Date' column to datetime
    df['Date'] = df['Date'].apply(convert_date)
    # Filter out past games
    future_games = df[df['Date'] > datetime.now()]
    # Sort the dataframe by date
    future_games = future_games.sort_values(by='Date')
    # Get the next game
    next_game = future_games.iloc[0] if not future_games.empty else None
    return next_game

# Function to get the most recent win streak from a team's schedule
def get_recent_streak(df: DataFrame):
    # Convert the 'Date' column to datetime
    df['Date'] = df['Date'].apply(convert_date)
    # Filter out future games
    past_games = df[df['Date'] <= datetime.now()]
    # Sort the dataframe by date
    past_games = past_games.sort_values(by='Date', ascending=False)
    # Get the most recent streak
    recent_streak = past_games.iloc[0]['Streak'] if not past_games.empty else None
    return recent_streak

# Directory containing the CSV files
directory = 'Team Schedules'

# Dictionary to store next games and win streaks
next_games = {}
win_streaks = {}

# Iterate over each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        # Read the CSV file
        df = pd.read_csv(filepath)
        # Find the next game
        team_name = filename.split('_')[0]
        next_game = find_next_game(df)
        if next_game is not None:
            next_games[team_name] = next_game
        # Get the most recent win streak
        recent_streak = get_recent_streak(df)
        if recent_streak is not None:
            win_streaks[team_name] = recent_streak

# Read the team_stats CSV
team_stats_filepath = 'combined_team_stats.csv'
team_stats_df = pd.read_csv(team_stats_filepath)

# Strip any whitespace from column names
team_stats_df.columns = team_stats_df.columns.str.strip()

# Debug: Print the column names to ensure 'Team' column exists
print("Columns in team_stats_df:", team_stats_df.columns)

team_stats_df['Team'] = team_stats_df['Team'].str.upper().str.strip()  # Ensure team names are uppercase and stripped of whitespace

# Debug: Print unique team names in team_stats_df
print("Unique team names in team_stats_df:", team_stats_df['Team'].unique())

# List to store rows for the new CSV
matchups = []

# Process each next game
for team, game in next_games.items():
    # Identify the home and away teams
    if game['Home_Away'] == 'Home':
        home_team = team.upper().strip()
        away_team = game['Opp'].upper().strip()
    else:
        home_team = game['Opp'].upper().strip()
        away_team = team.upper().strip()

    # Get stats for home and away teams
    home_stats = team_stats_df[team_stats_df['Team'] == home_team]
    away_stats = team_stats_df[team_stats_df['Team'] == away_team]

    if not home_stats.empty and not away_stats.empty:
        home_stats = home_stats.add_prefix('Home_').iloc[0]
        away_stats = away_stats.add_prefix('Away_').iloc[0]

        # Get the most recent win streaks
        home_streak = win_streaks.get(home_team, '0')
        away_streak = win_streaks.get(away_team, '0')

        # Combine stats into one row
        combined_stats = pd.concat([home_stats, away_stats])
        combined_stats['Home'] = home_team
        combined_stats['Away'] = away_team
        combined_stats['Home_Streak'] = home_streak
        combined_stats['Away_Streak'] = away_streak

        # Add the combined stats to the list of matchups
        matchups.append(combined_stats)

# Create a new DataFrame for the matchups
matchups_df = pd.DataFrame(matchups)

# Save the matchups DataFrame to a new CSV file
matchups_df.to_csv('testing.csv', index=False)

print("Next game stats with streaks added to next_game_stats_with_streaks.csv successfully!")
