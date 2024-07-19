import pandas as pd

batting_data = pd.read_csv("team_batting_2024.csv")
fielding_data = pd.read_csv("team_fielding_2024.csv")
pitching_data = pd.read_csv("team_pitching_2024.csv")

# Merge fielding and batting data
merged_df = batting_data.merge(fielding_data, on='teamIDfg', suffixes=('', '_df2'))

# Merge the result with pitching data
merged_df = merged_df.merge(pitching_data, on='teamIDfg', suffixes=('', '_df3'))

# Identify columns with duplicates based on suffixes
cols_to_remove = [col for col in merged_df.columns if col.endswith('_df2') or col.endswith('_df3')]

# Drop the duplicate columns
merged_df.drop(columns=cols_to_remove, inplace=True)

# Save the resulting DataFrame to a new CSV file
merged_df.to_csv('combined_team_stats.csv', index=False)


