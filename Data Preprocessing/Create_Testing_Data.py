import pandas as pd
import os
from datetime import datetime

from sklearn.impute import SimpleImputer

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

# Dictionary to store the next matchup for each team
next_matchups = []

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
        next_game = df[df['Date'] > datetime.now()].sort_values(by='Date').head(1)

        for index, game in next_game.iterrows():
            home_team = filename.split('_')[0].upper().strip() if game['Home_Away'] == 'Home' else game['Opp'].upper().strip()
            away_team = game['Opp'].upper().strip() if game['Home_Away'] == 'Home' else filename.split('_')[0].upper().strip()

            home_stats = team_stats_df[team_stats_df['Team'] == home_team]
            away_stats = team_stats_df[team_stats_df['Team'] == away_team]

            if not home_stats.empty and not away_stats.empty:
                home_stats = home_stats.add_prefix('Home_').iloc[0]
                away_stats = away_stats.add_prefix('Away_').iloc[0]
                
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

                combined_stats = pd.concat([home_stats, away_stats])
                combined_stats['Home_Team'] = home_team
                combined_stats['Away_Team'] = away_team
                combined_stats['Home_Streak'] = home_streak
                combined_stats['Away_Streak'] = away_streak

                next_matchups.append(combined_stats)


# Create DataFrame for next matchups
next_matchups_df = pd.DataFrame(next_matchups)

# Remove exact duplicates based on home and away teams
next_matchups_df = next_matchups_df.drop_duplicates(subset=['Home_Team', 'Away_Team'])

# Drop the Home_Team and Away_Team columns
next_matchups_df = next_matchups_df.drop(columns=['Home_Team', 'Away_Team'])

# # Drop columns with all NaN values
# next_matchups_df = next_matchups_df.dropna(axis=1, how='all')

# # Impute missing values
# imputer = SimpleImputer(strategy='mean')
# next_matchups_df_imputed = imputer.fit_transform(next_matchups_df)

# past_matchups_df = pd.DataFrame(next_matchups_df_imputed, columns=next_matchups_df.columns)

# Save the next matchups DataFrame to CSV
next_matchups_df.to_csv('Testing.csv', index=False)

print("Next games saved to Testing.csv successfully!")
