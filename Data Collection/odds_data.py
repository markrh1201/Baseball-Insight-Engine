import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, date
import time
import os
import concurrent.futures
from tqdm import tqdm
import numpy as np

def scrape_odds_for_date(target_date_str: str):
    """
    Scrapes MLB moneyline odds from Sportsbook Review for a specific date.
    
    Args:
        target_date_str: The date to scrape in 'YYYY-MM-DD' format.
        
    Returns:
        A pandas DataFrame with the odds data for that day, or None if no games are found.
    """
    url = f"https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date={target_date_str}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        script_tag = soup.find('script', id='__NEXT_DATA__')
        if not script_tag:
            return None

        json_data = json.loads(script_tag.string) # type: ignore
        game_rows = json_data['props']['pageProps']['oddsTables'][0]['oddsTableModel']['gameRows']
        
        if not game_rows:
            return None

        extracted_data = []
        for game in game_rows:
            game_view = game.get('gameView', {})
            
            start_date_utc = game_view.get('startDate', '')
            
            # --- FIX: Validate that the game's date matches the requested date ---
            # This prevents data from other days from leaking into the results.
            if start_date_utc and pd.to_datetime(start_date_utc).strftime('%Y-%m-%d') != target_date_str:
                continue # Skip this game as it's not for the target date

            home_team = game_view.get('homeTeam', {}).get('displayName', 'N/A')
            away_team = game_view.get('awayTeam', {}).get('displayName', 'N/A')
            
            # --- FIX for Athletics naming inconsistency ---
            if home_team == "Athletics Athletics":
                home_team = "Athletics"
            if away_team == "Athletics Athletics":
                away_team = "Athletics"
            
            if start_date_utc:
                game_time = datetime.fromisoformat(start_date_utc.replace('Z', '+00:00')).strftime('%Y-%m-%d %I:%M %p ET')
            else:
                game_time = f"{target_date_str} Unknown Time"

            consensus = game_view.get('consensus')
            if consensus and consensus.get('homeMoneyLinePickPercent') is not None:
                home_wager = f"{consensus.get('homeMoneyLinePickPercent'):.2f}%"
                away_wager = f"{consensus.get('awayMoneyLinePickPercent'):.2f}%"
            else:
                home_wager, away_wager = 'N/A', 'N/A'
            
            opening_line = None
            if game.get('openingLineViews') and game['openingLineViews'][0]:
                opening_line = game['openingLineViews'][0].get('openingLine')
            
            if opening_line:
                home_opener = opening_line.get('homeOdds', 'N/A')
                away_opener = opening_line.get('awayOdds', 'N/A')
            else:
                home_opener, away_opener = 'N/A', 'N/A'

            extracted_data.append([
                game_time, home_team, away_team,
                home_wager, away_wager, home_opener, away_opener
            ])
            
        return pd.DataFrame(extracted_data, columns=[
            'Game Time', 'Home Team', 'Away Team', 
            'Home Wager %', 'Away Wager %', 
            'Home Opener Odds', 'Away Opener Odds'
        ])

    except requests.exceptions.RequestException:
        return None
    except (KeyError, IndexError, TypeError):
        return None

if __name__ == "__main__":
    START_YEAR = 2022
    OUTPUT_DIR = "processed_data"
    OUTPUT_FILENAME = "mlb_odds_2022_present.csv"
    MAX_WORKERS = 6
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_date = date(START_YEAR, 1, 1)
    end_date = date.today()
    date_range = pd.date_range(start_date, end_date)
    
    all_odds_dfs = []
    
    dates_to_scrape = [
        d.strftime('%Y-%m-%d') for d in date_range if d.month in range(3, 12)
    ]

    print(f"Starting to scrape MLB odds for {len(dates_to_scrape)} days using up to {MAX_WORKERS} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(scrape_odds_for_date, dates_to_scrape), total=len(dates_to_scrape)))

    for daily_df in results:
        if daily_df is not None and not daily_df.empty:
            all_odds_dfs.append(daily_df)

    if all_odds_dfs:
        print("\nCombining all collected data...")
        master_odds_df = pd.concat(all_odds_dfs, ignore_index=True)
        
        print("Processing N/A values...")
        
        for col in ['Home Wager %', 'Away Wager %']:
            numeric_col = pd.to_numeric(master_odds_df[col].str.replace('%', ''), errors='coerce')
            mean_value = numeric_col.mean()
            master_odds_df[col] = numeric_col.fillna(mean_value).map('{:.2f}%'.format)
            print(f"  - Filled N/A values in '{col}' with mean: {mean_value:.2f}%")

        for col in ['Home Opener Odds', 'Away Opener Odds']:
            numeric_col = pd.to_numeric(master_odds_df[col], errors='coerce')
            master_odds_df[col] = numeric_col.fillna(0).astype(int)
            print(f"  - Filled N/A values in '{col}' with 0.")

        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        master_odds_df.to_csv(output_path, index=False)
        
        print("\n--- Scraping Complete! ---")
        print(f"✅ Full dataset with {len(master_odds_df)} games saved to '{output_path}'")
        print("\n--- Sample of the Final DataFrame ---")
        print(master_odds_df.head())
    else:
        print("\n--- Scraping Complete ---")
        print("No game data was found in the specified date range.")