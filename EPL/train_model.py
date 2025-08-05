import pandas as pd
import numpy as np
import os
import joblib
# FIX: Import XGBClassifier for Scikit-Learn compatibility
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import geopy.distance
import xgboost as xgb

INPUT_CSV = 'historical_training_data_with_ids.csv'
MODEL_FILE = 'match_predictor_model.joblib'
TEAM_MAP_FILE = 'team_id_map.joblib'
REFEREE_MAP_FILE = 'referee_map.joblib'

stadium_name_to_coords = {
    'Arsenal': (51.555, -0.1086),
    'Aston Villa': (52.5106, -1.8942),
    'Barnsley': (53.5505, -1.4750),
    'Birmingham': (52.4722, -1.8681),
    'Blackburn': (53.705, -2.4891),
    'Blackpool': (53.8022, -3.0519),
    'Bolton': (53.5891, -2.5342),
    'Bournemouth': (50.7352, -1.8383),
    'Bradford': (53.8044, -1.7658),
    'Brentford': (51.4908, -0.2886),
    'Brighton': (50.862, -0.083),
    'Brighton & Hove Albion': (50.862, -0.083),
    'Bristol City': (51.4394, -2.6308),
    'Burnley': (53.7889, -2.2303),
    'Burton': (52.8133, -1.6153),
    'Cardiff': (51.4731, -3.2001),
    'Charlton': (51.4864, 0.0372),
    'Chelsea': (51.4816, -0.191),
    'Colchester': (51.9075, 0.8731),
    'Coventry': (52.4394, -1.4955),
    'Crewe': (53.0869, -2.4411),
    'Crystal Palace': (51.3983, -0.0836),
    'Derby': (52.9161, -1.4472),
    'Doncaster': (53.5153, -1.1094),
    'Everton': (53.4388, -2.9664),
    'Fulham': (51.475, -0.2217),
    'Gillingham': (51.3914, 0.5556),
    'Huddersfield': (53.6453, -1.7708),
    'Hull': (53.7403, -0.3719),
    'Ipswich': (52.055, 1.1447),
    'Ipswich Town': (52.055, 1.1447),
    'Leeds': (53.7788, -1.5721),
    'Leicester': (52.6197, -1.1422),
    'Liverpool': (53.4308, -2.9608),
    'Luton': (51.8864, -0.4403),
    'Man City': (53.4831, -2.2004),
    'Man United': (53.4631, -2.2914),
    'Middlesbrough': (54.5779, -1.2144),
    'Millwall': (51.4869, -0.0519),
    'Milton Keynes Dons': (52.0058, -0.7406),
    'Newcastle': (54.9757, -1.6212),
    'Norwich': (52.6225, 1.3094),
    'Nott\'m Forest': (52.9400, -1.1328),
    'Oldham': (53.5525, -2.1058),
    'Oxford': (51.7131, -1.2061),
    'Peterboro': (52.5642, -0.2428),
    'Plymouth': (50.3914, -4.1458),
    'Portsmouth': (50.7964, -1.0500),
    'Preston': (53.7675, -2.6944),
    'QPR': (51.5089, -0.2314),
    'Reading': (51.4214, -0.9983),
    'Rotherham': (53.4316, -1.3583),
    'Scunthorpe': (53.5911, -0.7131),
    'Sheffield United': (53.3697, -1.4681),
    'Sheffield Weds': (53.4114, -1.5058),
    'Southend': (51.5392, 0.7044),
    'Southampton': (50.9147, -1.4131),
    'Stoke': (52.9886, -2.1889),
    'Sunderland': (54.9144, -1.3753),
    'Swansea': (51.642, -3.9405),
    'Swindon': (51.5631, -1.7733),
    'Tottenham': (51.6044, -0.0664),
    'Watford': (51.6492, -0.4019),
    'West Brom': (52.5083, -1.9681),
    'West Ham': (51.5386, -0.0164),
    'Wigan': (53.5469, -2.6536),
    'Wimbledon': (51.4352, -0.1888),
    'Wolves': (52.5903, -2.1303),
    'Wycombe': (51.6408, -0.785),
    'Yeovil': (50.9444, -2.6347)
}


def compute_distance_km(home_id, away_id, id_to_coords):
    if home_id in id_to_coords and away_id in id_to_coords:
        coords_home = id_to_coords[home_id]
        coords_away = id_to_coords[away_id]
        return geopy.distance.distance(coords_home, coords_away).km
    else:
        return 0

def weighted_avg(stats, weights=[0.6, 0.3, 0.1, 0, 0]):
    # Use only the first 3 weights/stats for weighted average
    weights = np.array(weights[:len(stats)])
    stats = np.array(stats[:len(weights)])
    if np.sum(weights) == 0:
        return 0
    return np.average(stats, weights=weights)

def engineer_features(df):
    print("Starting feature engineering...")
    df.dropna(subset=['home_goals_full', 'away_goals_full'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    try:
        team_id_map = joblib.load(TEAM_MAP_FILE)
        id_to_name_map = {v: k for k, v in team_id_map.items()}
    except FileNotFoundError:
        print(f"Error: '{TEAM_MAP_FILE}' not found.")
        return None, None, None

    id_to_coords = {
        team_id: stadium_name_to_coords.get(team_name, (0, 0))
        for team_name, team_id in team_id_map.items()
    }

    all_referees = df['referee'].dropna().unique()
    referee_map = {ref: i for i, ref in enumerate(all_referees)}
    df['referee_encoded'] = df['referee'].map(referee_map).fillna(-1).astype(int)

    df['result'] = df.apply(
        lambda row: 2 if row['home_goals_full'] > row['away_goals_full'] else (0 if row['home_goals_full'] < row['away_goals_full'] else 1),
        axis=1
    )

    # Add is_home = 1 for every row (since each row is from home team perspective)
    df['is_home'] = 1

    team_stats = {}
    last_played = {}
    head_to_head = {}
    processed_rows = []

    for index, row in df.iterrows():
        home_id, away_id, date = row['home_team_id'], row['away_team_id'], row['date']

        def get_form_stats(team_id):
            stats = team_stats.get(team_id, {'goals_scored': [0]*5, 'goals_conceded': [0]*5, 'points': [0]*5})
            # Use weighted average over last 3 matches
            gs = weighted_avg(stats['goals_scored'])
            gc = weighted_avg(stats['goals_conceded'])
            pts = weighted_avg(stats['points'])
            strength = np.sum(stats['points']) / 5.0  # keep strength as sum over 5
            return gs, gc, pts, strength
        
        row['home_avg_gs'], row['home_avg_gc'], row['home_avg_pts'], row['home_strength'] = get_form_stats(home_id)
        row['away_avg_gs'], row['away_avg_gc'], row['away_avg_pts'], row['away_strength'] = get_form_stats(away_id)

        row['team_strength_gap'] = row['home_strength'] - row['away_strength']
        row['travel_distance'] = compute_distance_km(home_id, away_id, id_to_coords)

        row['rest_days_home'] = (date - last_played.get(home_id, date)).days
        row['rest_days_away'] = (date - last_played.get(away_id, date)).days
        last_played[home_id] = date
        last_played[away_id] = date

        fixture_key = tuple(sorted((home_id, away_id)))
        past_h2h = head_to_head.get(fixture_key, [])[-5:]
        row['h2h_avg_goal_diff'] = np.mean(past_h2h) if past_h2h else 0

        processed_rows.append(row)

        home_points = 3 if row['result'] == 2 else (1 if row['result'] == 1 else 0)
        away_points = 3 if row['result'] == 0 else (1 if row['result'] == 1 else 0)

        def update_stats(team_id, gf, ga, pts):
            stats = team_stats.get(team_id, {'goals_scored': [0]*5, 'goals_conceded': [0]*5, 'points': [0]*5})
            stats['goals_scored'] = stats['goals_scored'][1:] + [gf]
            stats['goals_conceded'] = stats['goals_conceded'][1:] + [ga]
            stats['points'] = stats['points'][1:] + [pts]
            team_stats[team_id] = stats

        update_stats(home_id, row['home_goals_full'], row['away_goals_full'], home_points)
        update_stats(away_id, row['away_goals_full'], row['home_goals_full'], away_points)

        goal_diff = row['home_goals_full'] - row['away_goals_full'] if home_id < away_id else row['away_goals_full'] - row['home_goals_full']
        head_to_head.setdefault(fixture_key, []).append(goal_diff)

    print("Feature engineering complete.")
    return pd.DataFrame(processed_rows), team_id_map, referee_map



def train_and_evaluate_model():
    try:
        raw_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: '{INPUT_CSV}' not found.")
        return

    processed_df, team_map, referee_map = engineer_features(raw_df)
    if processed_df is None:
        return

    features = [
        'home_team_id', 'away_team_id', 'referee_encoded',
        'home_avg_gs', 'home_avg_gc', 'home_avg_pts',
        'away_avg_gs', 'away_avg_gc', 'away_avg_pts',
        'team_strength_gap', 'travel_distance',
        'rest_days_home', 'rest_days_away',
        'h2h_avg_goal_diff',
        'is_home'
    ]
    target = 'result'
    
    # Drop rows where features have NaN (including travel_distance 0 replaced with NaN if you want)
    processed_df.dropna(subset=features, inplace=True)
    X = processed_df[features]
    y = processed_df[target]

    # Use 380 last matches as test set (like before)
    num_test_matches = 380
    X_train_full, X_test = X[:-num_test_matches], X[-num_test_matches:]
    y_train_full, y_test = y[:-num_test_matches], y[-num_test_matches:]

    # Split training set into train + validation (e.g. 90% train, 10% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    print(f"\nTraining on {len(X_train)} matches, validating on {len(X_val)} matches, testing on {len(X_test)} matches.")

    # FIX: Use XGBClassifier for scikit-learn API compatibility
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softprob',
        early_stopping_rounds=20, 
    )
    
    # FIX: Use the scikit-learn .fit() method
    model.fit(X_train, y_train, 
 
              eval_set=[(X_val, y_val)], 
              verbose=True)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on test set: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win'], zero_division=0))

    # FIX: Use joblib.dump to save the model and protocol=4
    joblib.dump(model, MODEL_FILE, protocol=4)
    joblib.dump(referee_map, REFEREE_MAP_FILE, protocol=4)
    print(f"\nModel saved to '{MODEL_FILE}'")
    print(f"Referee map saved to '{REFEREE_MAP_FILE}'")

if __name__ == "__main__":
    train_and_evaluate_model()
