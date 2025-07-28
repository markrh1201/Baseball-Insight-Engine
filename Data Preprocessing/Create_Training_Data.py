import pandas as pd
import os
from datetime import datetime
from sklearn.impute import SimpleImputer
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

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

# Dictionary to store wins, games count, and streaks for each team
team_stats = {}

# Function to get the streak value and handle NaNs
def get_streak_value(streak):
    if pd.isna(streak):
        return 0
    return int(streak)

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
                away_win = 1 if result == 'W' and game['Home_Away'] == 'Away' else 0

                # Use the 'Streak' column directly and handle NaNs
                home_streak = get_streak_value(game['Streak']) if game['Home_Away'] == 'Home' else -get_streak_value(game['Streak'])
                away_streak = get_streak_value(game['Streak']) if game['Home_Away'] == 'Away' else -get_streak_value(game['Streak'])

                # Initialize team stats if not already done
                if home_team not in team_stats:
                    team_stats[home_team] = {'wins': 0, 'games': 0, 'streak': home_streak}
                if away_team not in team_stats:
                    team_stats[away_team] = {'wins': 0, 'games': 0, 'streak': away_streak}

                # Update the streaks
                team_stats[home_team]['streak'] = home_streak
                team_stats[away_team]['streak'] = away_streak

                # Update matchup stats
                matchup_key = (home_team, away_team)
                if matchup_key not in team_stats:
                    team_stats[matchup_key] = {'wins': 0, 'games': 0}

                team_stats[matchup_key]['games'] += 1
                if home_win:
                    team_stats[matchup_key]['wins'] += 1

                combined_stats = pd.concat([home_stats, away_stats])
                combined_stats['Home_Win'] = home_win
                combined_stats['Home_Team'] = home_team
                combined_stats['Away_Team'] = away_team
                combined_stats['Home_Streak'] = home_streak
                combined_stats['Away_Streak'] = away_streak

                past_matchups.append(combined_stats)

# Create DataFrame for past matchups
past_matchups_df = pd.DataFrame(past_matchups)

# Strip any leading/trailing whitespace from column names
past_matchups_df.columns = past_matchups_df.columns.str.strip()

# Remove exact duplicates based on home and away teams
past_matchups_df = past_matchups_df.drop_duplicates(subset=['Home_Team', 'Away_Team'])

# Separate features and target
X_train = past_matchups_df.drop(columns=['Home_Win', 'Home_Team', 'Away_Team'])
y_train = past_matchups_df['Home_Win']

# Handle missing values
imputer = SimpleImputer(strategy='constant')
X_train_imputed = imputer.fit_transform(X_train)

# Ensure the columns match after imputation
X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_train.columns)
print("Columns before resampling:", X_train_imputed_df.columns)
print("Shape before resampling:", X_train_imputed_df.shape)

# Define oversampling and undersampling strategy
oversample = SMOTE()
undersample = RandomUnderSampler()

# Create a pipeline for resampling
resampling_pipeline = ImbPipeline(steps=[('o', oversample), ('u', undersample)])
X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train_imputed_df, y_train) # type: ignore

print("Shape after resampling:", X_resampled.shape)

# Convert back to DataFrame
X_resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
y_resampled_df = pd.Series(y_resampled, name='Home_Win')

# Combine features and target into one DataFrame
resampled_past_matchups_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)

# Save the resampled past matchups DataFrame to CSV
resampled_past_matchups_df.to_csv('Training.csv', index=False)

print("Resampled past games saved to Training.csv successfully!")
