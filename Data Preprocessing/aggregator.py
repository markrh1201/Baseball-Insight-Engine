import pandas as pd

batting_data = pd.read_csv("team_batting_2024.csv")
fielding_data = pd.read_csv("team_fielding_2024.csv")
pitching_data = pd.read_csv("team_pitching_2024.csv")

merged_data = pd.merge(batting_data, fielding_data, on="teamIDfg")
merged_data = pd.merge(merged_data, pitching_data, on="teamIDfg")

merged_data.to_csv("team_stats.csv")