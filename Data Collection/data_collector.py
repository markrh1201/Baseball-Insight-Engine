import pandas as pd
from pybaseball import team_batting, team_pitching, team_fielding, schedule_and_record, playerid_reverse_lookup

batting_data = team_batting(2024)
batting_data.to_csv('team_batting_2024.csv', index=False)

fielding_data =team_fielding(2024)
fielding_data.to_csv('team_fielding_2024.csv', index=False)

pitching_data = team_pitching(2024)
pitching_data.to_csv('team_pitching_2024.csv', index=False)

teams = ["LAD", "BAL", "MIN", "NYY", "PHI", "HOU", "MIL", "BOS", "NYM", "ARI", "SDP", "CLE", 
         "ATL", "SFG", "KCR", "TEX", "COL", "STL", "CIN", "TOR", "LAA", "CHC", "TBR", "WSN", 
         "OAK", "DET", "SEA", "PIT", "CHW", "MIA"]

for team in teams:    
    schedule = schedule_and_record(2024, team)
    schedule.to_csv(f"{team}_schedule.csv", index=False)