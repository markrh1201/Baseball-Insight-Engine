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
        
        all_odds_tables = json_data['props']['pageProps']['oddsTables']
        if not all_odds_tables:
            return None

        extracted_data = []
        
        for odds_table in all_odds_tables:
            game_rows = odds_table.get('oddsTableModel', {}).get('gameRows', [])
            
            for game in game_rows:
                game_view = game.get('gameView', {})
                
                # <<< FIX: REMOVED the problematic date validation check here.
                # The URL parameter already ensures we are on the correct day.
                
                start_date_utc = game_view.get('startDate', '')

                home_team = game_view.get('homeTeam', {}).get('displayName', 'N/A')
                away_team = game_view.get('awayTeam', {}).get('displayName', 'N/A')
                
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
                
        if not extracted_data:
            return None

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
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # --- Determine which dates need to be scraped ---

    # 1. Generate all possible dates from the start year to today's date
    start_date = date(START_YEAR, 3, 1) # Start with MLB season
    end_date = date.today()
    date_range = pd.date_range(start_date, end_date)
    
    # Filter for the MLB season months (March-November)
    all_possible_dates = {d.strftime('%Y-%m-%d') for d in date_range if d.month in range(3, 12)}
    
    # 2. If the file exists, find out which dates are already saved
    existing_df = pd.DataFrame()
    if os.path.exists(output_path):
        print(f"Reading existing data from '{output_path}'...")
        try:
            existing_df = pd.read_csv(output_path)
            # Reliably parse dates from the 'Game Time' column
            scraped_dates = set(pd.to_datetime(existing_df['Game Time']).dt.strftime('%Y-%m-%d'))
            print(f"Found {len(scraped_dates)} previously scraped dates.")
        except (pd.errors.EmptyDataError, KeyError):
            print("⚠️ Existing file is empty or invalid. Starting a fresh scrape.")
            scraped_dates = set()
    else:
        print("No existing data file found. Starting a new scrape.")
        scraped_dates = set()

    # 3. Determine the final list of dates to scrape
    dates_to_scrape = sorted(list(all_possible_dates - scraped_dates))

    # --- Execute Scraping ---

    if not dates_to_scrape:
        print("\n--- Scraping Complete ---")
        print("✅ All dates up to today are already in the dataset. No new data to scrape.")
    else:
        print(f"\nFound {len(dates_to_scrape)} new day(s) to scrape...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(scrape_odds_for_date, dates_to_scrape), total=len(dates_to_scrape)))

        newly_scraped_dfs = [df for df in results if df is not None and not df.empty]

        if newly_scraped_dfs:
            print("\nProcessing and combining new data...")
            new_master_df = pd.concat(newly_scraped_dfs, ignore_index=True)
            
            # Process N/A values for the newly scraped data
            for col in ['Home Opener Odds', 'Away Opener Odds']:
                numeric_col = pd.to_numeric(new_master_df[col], errors='coerce')
                new_master_df[col] = numeric_col.fillna(0).astype(int)
            
            # Combine old and new dataframes
            final_df = pd.concat([existing_df, new_master_df], ignore_index=True)

            # Sort the final dataset by game time for consistency
            final_df['Game Time'] = pd.to_datetime(final_df['Game Time'])
            final_df = final_df.sort_values(by='Game Time').reset_index(drop=True)
            
            # Save the updated complete dataset
            final_df.to_csv(output_path, index=False)
            
            print("\n--- Scraping Complete! ---")
            print(f"✅ Appended {len(new_master_df)} new games. Full dataset with {len(final_df)} games saved to '{output_path}'")
            print("\n--- Sample of the Latest Data ---")
            print(final_df.tail()) # Use .tail() to show the most recent entries
        else:
            print("\n--- Scraping Complete ---")
            print("No new game data was found for the missing dates.")