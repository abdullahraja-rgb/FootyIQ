import pandas as pd
import os
import joblib

# --- File Paths ---
PREM_HISTORICAL_FILE = "premier_league_1993_2024.csv"
CHAMPIONSHIP_FILE = "championship_2004_2024.csv"
OUTPUT_FILE = "historical_training_data_with_ids.csv"
TEAM_MAP_FILE = "team_id_map.joblib"

# --- Column Mapping ---
COLUMN_MAPPING = {
    'Date': 'date',
    'Season': 'season',
    'HomeTeam': 'home_team_name',
    'AwayTeam': 'away_team_name',
    'FTH Goals': 'home_goals_full',
    'FTA Goals': 'away_goals_full',
    'FT Result': 'result',
    'HT Result': 'half_time_result',
    'Referee': 'referee',
    'H Shots': 'home_shots',
    'A Shots': 'away_shots',
    'H SOT': 'home_shots_on_target',
    'A SOT': 'away_shots_on_target',
    'H Fouls': 'home_fouls',
    'A Fouls': 'away_fouls',
    'H Corners': 'home_corners',
    'A Corners': 'away_corners',
    'H Yellow': 'home_yellow_cards',
    'A Yellow': 'away_yellow_cards',
    'H Red': 'home_red_cards',
    'A Red': 'away_red_cards',
    'League': 'league'
}

def process_historical_data(file_path):
    """
    Reads a historical CSV file, renames columns, and standardizes data.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    
    print(f"Processing data from: {file_path}")
    df = pd.read_csv(file_path)

    # Rename columns based on the mapping
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    # Standardize the date format
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    
    # Standardize the result
    result_map = {'H': 'HOME_TEAM', 'D': 'DRAW', 'A': 'AWAY_TEAM'}
    df['result'] = df['result'].replace(result_map)

    # Clean up column names for half time goals
    df = df.rename(columns={'HTH Goals': 'home_goals_half', 'HTA Goals': 'away_goals_half'})
        
    return df

def main():
    """
    Combines historical CSVs, creates team IDs, and saves the final dataset and map.
    """
    print("Starting data combination and ID creation process...")

    # Load and process the two historical datasets
    df_prem = process_historical_data(PREM_HISTORICAL_FILE)
    df_champ = process_historical_data(CHAMPIONSHIP_FILE)

    # Combine the dataframes
    combined_df = pd.concat([df_prem, df_champ], ignore_index=True, sort=False)

    # Drop potential duplicates
    combined_df.drop_duplicates(subset=['date', 'home_team_name', 'away_team_name'], inplace=True)
    
    # Sort the data by date
    combined_df = combined_df.sort_values(by='date', ascending=False).reset_index(drop=True)

    # --- NEW LOGIC FOR ID CREATION ---
    print("Creating unique team IDs...")
    
    # Get a list of all unique team names from both home and away columns
    all_teams = pd.Series(
        combined_df['home_team_name'].tolist() + 
        combined_df['away_team_name'].tolist()
    ).unique()
    
    # Create a mapping from team name to a unique integer ID
    team_map = {name: i for i, name in enumerate(all_teams)}
    
    # Use the map to create new ID columns in the DataFrame
    combined_df['home_team_id'] = combined_df['home_team_name'].map(team_map)
    combined_df['away_team_id'] = combined_df['away_team_name'].map(team_map)
    
    # Save the final combined data to a new CSV
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data combination complete. Saved {len(combined_df)} unique rows to '{OUTPUT_FILE}'.")
    
    # Save the team mapping to a file for use in the prediction script
    # FIX: Use a newer protocol for joblib dump to prevent corruption
    joblib.dump(team_map, TEAM_MAP_FILE, protocol=4)
    print(f"Team ID mapping saved to '{TEAM_MAP_FILE}'.")

    return combined_df

if __name__ == "__main__":
    main()