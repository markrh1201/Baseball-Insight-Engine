import pandas as pd 
from pybaseball import batting_stats, pitching_stats
import os

# Create a directory for the raw data if it doesn't exist
os.makedirs('raw_data', exist_ok=True)

# --- Batting Data ---
print("Fetching batting stats...")
batting_df = batting_stats('2022', '2025')

# FIX: Filter out rows where the team is 'TOT' (for players traded mid-season)
batting_df_cleaned = batting_df[~batting_df['Team'].isin(['TOT', '- - -'])]

# Save the cleaned data
batting_df_cleaned.to_csv('raw_data/batting_data.csv', index=False)
print(f"Saved {len(batting_df_cleaned)} rows of batting data.")


# --- Pitching Data ---
print("\nFetching pitching stats...")
pitching_df = pitching_stats('2022', '2025')

# FIX: Filter out rows where the team is 'TOT'
pitching_df_cleaned = pitching_df[~pitching_df['Team'].isin(['TOT', '- - -'])]

# Save the cleaned data
pitching_df_cleaned.to_csv('raw_data/pitching_data.csv', index=False)
print(f"Saved {len(pitching_df_cleaned)} rows of pitching data.")