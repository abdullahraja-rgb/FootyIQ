import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_KEY = os.environ.get('FOOTBALL_API_KEY')
BASE_URL = "http://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}
PREMIER_LEAGUE_ID = 'PL' # Competition code for Premier League
OUTPUT_CSV = 'raw_matches.csv'

# Seasons to fetch data for (e.g., 2018 to 2024)
SEASONS = range(2018, 2025) 

def parse_match_data(match):
    """
    Parses a single match dictionary from the API into a flat dictionary.
    """
    return {
        'match_id': match.get('id'),
        'date': match.get('utcDate'),
        'season': match.get('season', {}).get('startDate', '')[:4],
        'matchday': match.get('matchday'),
        'home_team_id': match.get('homeTeam', {}).get('id'),
        'home_team_name': match.get('homeTeam', {}).get('name'),
        'away_team_id': match.get('awayTeam', {}).get('id'),
        'away_team_name': match.get('awayTeam', {}).get('name'),
        'home_goals_full': match.get('score', {}).get('fullTime', {}).get('home'),
        'away_goals_full': match.get('score', {}).get('fullTime', {}).get('away'),
        'home_goals_half': match.get('score', {}).get('halfTime', {}).get('home'),
        'away_goals_half': match.get('score', {}).get('halfTime', {}).get('away'),
        'result': match.get('score', {}).get('winner'),
        'referee': match.get('referees', [{}])[0].get('name') if match.get('referees') else None
    }

def fetch_all_seasons_data():
    """
    Fetches detailed match data for multiple Premier League seasons and saves it to a CSV.
    """
    all_matches_list = []
    
    print("Starting detailed data collection...")
    for season in SEASONS:
        print(f"Fetching data for the {season}-{season+1} season...")
        try:
            url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/matches"
            params = {'season': season, 'status': 'FINISHED'}
            
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            
            data = response.json()
            season_matches = data.get('matches', [])
            
            if not season_matches:
                print(f"Warning: No matches found for the {season} season.")
                continue
            
            # Parse each match and add it to our list
            for match in season_matches:
                all_matches_list.append(parse_match_data(match))
                
            print(f"Successfully fetched and parsed {len(season_matches)} matches for the {season} season.")
            
            # Respect API rate limits (the free tier allows 10 calls per minute)
            time.sleep(7) # Sleep for 7 seconds between seasons

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for season {season}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for season {season}: {e}")
            
    if all_matches_list:
        # Convert the list of dictionaries to a pandas DataFrame
        matches_df = pd.DataFrame(all_matches_list)
        
        # Save the combined DataFrame to a CSV file
        matches_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nData collection complete. All matches saved to '{OUTPUT_CSV}'.")
        print(f"Total matches collected: {len(matches_df)}")
    else:
        print("\nData collection failed. No data was saved.")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: FOOTBALL_API_KEY not found. Please set it in your .env file.")
    else:
        fetch_all_seasons_data()