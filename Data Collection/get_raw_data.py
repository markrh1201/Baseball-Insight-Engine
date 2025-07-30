import pandas as pd
from pybaseball import batting_stats, pitching_stats
import statsapi  # Using statsapi for schedules
import os
import time

# --- Configuration ---
YEARS = [2022, 2023, 2024, 2025]
RAW_DIR = 'raw_data'

# Create a directory for the raw data if it doesn't exist
os.makedirs(RAW_DIR, exist_ok=True)

# --- Batting Data ---
print("Fetching batting stats...")
# Setting qual=1 ensures we get all players with at least 1 plate appearance
batting_df = batting_stats(2022, 2025, qual=1)

# Filter out rows where the team is 'TOT' (for players traded mid-season)
batting_df_cleaned = batting_df[~batting_df['Team'].isin(['TOT', '- - -'])].copy()

# **FIX: Update 'OAK' team abbreviation to 'ATH'**
batting_df_cleaned['Team'] = batting_df_cleaned['Team'].replace('OAK', 'ATH')

# Save the cleaned data
batting_output_path = os.path.join(RAW_DIR, 'batting_data.csv')
batting_df_cleaned.to_csv(batting_output_path, index=False)
print(f"✅ Saved {len(batting_df_cleaned)} rows of batting data to '{batting_output_path}'")


# --- Pitching Data ---
print("\nFetching pitching stats...")
# Set qual=1 to fetch all pitchers, regardless of innings pitched
pitching_df = pitching_stats(2022, 2025, qual=1)

# Filter out rows where the team is 'TOT'
pitching_df_cleaned = pitching_df[~pitching_df['Team'].isin(['TOT', '- - -'])].copy()

# **FIX: Update 'OAK' team abbreviation to 'ATH'**
pitching_df_cleaned['Team'] = pitching_df_cleaned['Team'].replace('OAK', 'ATH')

# Save the cleaned data
pitching_output_path = os.path.join(RAW_DIR, 'pitching_data.csv')
pitching_df_cleaned.to_csv(pitching_output_path, index=False)
print(f"✅ Saved {len(pitching_df_cleaned)} rows of pitching data to '{pitching_output_path}'")


# --- Schedule Data (Using MLB-StatsAPI for Reliability) ---
print("\nFetching full league schedules using MLB-StatsAPI...")

all_games_data = []
for year in YEARS:
    try:
        print(f"  Fetching full league schedule for {year}...")
        # statsapi.schedule is more direct and reliable for this task
        yearly_schedule = statsapi.schedule(season=year)
        all_games_data.extend(yearly_schedule) # Use extend since it returns a list of dicts
        time.sleep(1) # Be polite to the server
    except Exception as e:
        print(f"    Could not fetch schedule for {year}. Reason: {e}")

if all_games_data:
    # Convert the list of dictionaries to a DataFrame
    schedule_df = pd.DataFrame(all_games_data)
    
    schedule_output_path = os.path.join(RAW_DIR, 'schedule_data.csv')
    schedule_df.to_csv(schedule_output_path, index=False)
    print(f"\n✅ Saved {len(schedule_df)} total games to '{schedule_output_path}'")
else:
    print("\n⚠️ No schedule data was downloaded.")