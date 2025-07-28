import pandas as pd
import requests
import json
import re
import time
import os
from bs4 import BeautifulSoup, Tag

# --- Configuration ---
YEARS = [2022, 2023, 2024, 2025] # Expanded years for a more complete dataset
STAT_TYPES = ['pitcher', 'batter']
all_data = {}

# NEW: Define a directory to store the CSV files
OUTPUT_DIR = "savant_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_savant_data_direct(year, stat_type):
    """
    Downloads data by finding the JSON object embedded directly in the page's HTML.
    """
    base_url = "https://baseballsavant.mlb.com/leaderboard/custom"
    min_pa = 'q'
    url = f"{base_url}?year={year}&type={stat_type}&lg=all&ind=1&min={min_pa}&sort=xwoba&sortDir=desc"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Requesting {stat_type} data for {year}...")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')
        scripts = soup.find_all('script')
        
        json_data = None
        for script in scripts:
            if isinstance(script, Tag) and script.string and 'var data = ' in script.string:
                match = re.search(r'var data = (\[.*\]);', script.string)
                if match:
                    json_text = match.group(1)
                    json_data = json.loads(json_text)
                    break
        
        if not json_data:
            print(f"  ERROR: Could not find embedded JSON data for {year} {stat_type}.")
            return None

        df = pd.DataFrame(json_data)
        return df

    except Exception as e:
        print(f"  ERROR: An exception occurred for {year} {stat_type}. Reason: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    for year in YEARS:
        for stat_type in STAT_TYPES:
            key = f"{stat_type}_{year}"
            df = download_savant_data_direct(year, stat_type)
            
            if df is not None and not df.empty:
                all_data[key] = df
                print(f"  Successfully loaded data for '{key}'")

                # --- NEW: Save the DataFrame to a CSV file ---
                filename = f"savant_{key}.csv"
                filepath = os.path.join(OUTPUT_DIR, filename)
                df.to_csv(filepath, index=False)
                print(f"  Successfully saved data to '{filepath}'")
                # ---------------------------------------------

            else:
                print(f"  Failed to get data for {key}.")
            
            time.sleep(3)

    print("\n--- Download and save complete! ---")
    print(f"All files are located in the '{OUTPUT_DIR}' folder.")