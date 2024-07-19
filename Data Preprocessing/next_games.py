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

# Function to find the next game for a team
def find_next_game(df):
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
def get_recent_streak(df):
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

# Set to keep track of processed games to avoid duplicates
processed_games = set()

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
team_stats_df['Team'] = team_stats_df['Team'].str.upper()  # Ensure team names are uppercase

# List to store rows for the new CSV
matchups = []

# Process each next game
for team, game in next_games.items():
    # Identify the home and away teams
    if game['Home_Away'] == 'Home':
        home_team = team
        away_team = game['Opp'].upper()
    else:
        home_team = game['Opp'].upper()
        away_team = team

    # Create a unique identifier for the game to check for duplicates
    game_id = (home_team, away_team, game['Date'])

    # Check if the game has already been processed
    if game_id not in processed_games:
        # Mark the game as processed
        processed_games.add(game_id)

        # Get stats for home and away teams
        home_stats = team_stats_df[team_stats_df['Team'] == home_team].add_prefix('Home_').iloc[0]
        away_stats = team_stats_df[team_stats_df['Team'] == away_team].add_prefix('Away_').iloc[0]

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
matchups_df.to_csv('MLB_Game_Prediction_Input.csv', index=False)

print("Next game stats with streaks added to MLB_Game_Prediction_Input.csv successfully!")
