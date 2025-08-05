import os
import sqlite3
import requests
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
from flask_session import Session
from datetime import datetime
import joblib
# NEW: Import for distance calculation and numerical operations
import geopy.distance
import numpy as np

# --- NEW: Stadium Coordinates (Required for travel_distance feature) ---
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

# Configure application
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# API Configuration
api_key = os.environ.get('FOOTBALL_API_KEY', "46133a431ef84980bf20ed5eef34949e")
BASE_URL = "http://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": api_key}
PREMIER_LEAGUE_ID = 2021

# --- Load the Prediction Model and Mappings ---
try:
    model = joblib.load('match_predictor_model.joblib')
    team_map = joblib.load('team_id_map.joblib')
    referee_map = joblib.load('referee_map.joblib')
    print("✅ Prediction model and mappings loaded successfully.")
except FileNotFoundError:
    model = None
    team_map = None
    referee_map = None
    print("⚠️ Warning: Model or mapping files not found. Prediction endpoint will be disabled.")

# Prevent caching
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# DB helper
def get_db():
    connection = sqlite3.connect("predictor.db")
    connection.row_factory = sqlite3.Row
    return connection

# NEW: get_team_api_id helper function to get team ID for API calls
def get_team_api_id(team_name):
    standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
    try:
        standings_response = requests.get(standings_url, headers=HEADERS)
        standings_response.raise_for_status()
        standings_data = standings_response.json()
        for standing in standings_data['standings'][0]['table']:
            if standing['team']['name'].lower() == team_name.lower():
                return standing['team']['id']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team API ID: {e}")
    return None

# NEW: compute_distance_km helper function for travel distance
def compute_distance_km(home_name, away_name):
    coords_home = stadium_name_to_coords.get(home_name)
    coords_away = stadium_name_to_coords.get(away_name)
    if coords_home and coords_away:
        return geopy.distance.distance(coords_home, coords_away).km
    else:
        return 0

# NEW: get_h2h_goal_diff helper function for head-to-head feature
def get_h2h_goal_diff(home_name, away_name):
    home_api_id = get_team_api_id(home_name)
    away_api_id = get_team_api_id(away_name)
    if not home_api_id or not away_api_id:
        return 0

    url = f"{BASE_URL}/teams/{home_api_id}/matches?status=FINISHED&limit=10&competitions={PREMIER_LEAGUE_ID}"
    try:
        response = requests.get(url, headers=HEADERS, params={'limit': 20})
        response.raise_for_status()
        matches = response.json().get('matches', [])
        
        goal_diffs = []
        for match in matches:
            if match['homeTeam']['id'] == away_api_id or match['awayTeam']['id'] == away_api_id:
                if match['homeTeam']['id'] == home_api_id:
                    goal_diffs.append(match['score']['fullTime']['home'] - match['score']['fullTime']['away'])
                else:
                    goal_diffs.append(match['score']['fullTime']['away'] - match['score']['fullTime']['home'])
        
        return np.mean(goal_diffs) if goal_diffs else 0
    except requests.exceptions.RequestException:
        return 0

# --- CORRECTED get_form_and_h2h HELPER FUNCTION ---
# It now handles teams not in the current season gracefully
def get_form_and_h2h(team1_name, team2_name):
    form_data = {}
    h2h_results = []
    
    # Get team IDs from the API to make the API call
    team1_api_id = get_team_api_id(team1_name)
    team2_api_id = get_team_api_id(team2_name)
    
    if not team1_api_id or not team2_api_id:
        # Gracefully return empty data instead of raising an error
        return {'form_data': {}, 'h2h_results': []}

    try:
        # Fetch form for Team 1 and H2H data
        url1 = f"{BASE_URL}/teams/{team1_api_id}/matches?status=FINISHED&limit=20&competitions={PREMIER_LEAGUE_ID}"
        team1_response = requests.get(url1, headers=HEADERS)
        team1_response.raise_for_status()
        team1_matches = team1_response.json().get('matches', [])
        
        form1 = []
        for match in team1_matches[:5]:
            if (match['score']['winner'] == 'HOME_TEAM' and match['homeTeam']['id'] == team1_api_id) or \
               (match['score']['winner'] == 'AWAY_TEAM' and match['awayTeam']['id'] == team1_api_id):
                form1.append('W')
            elif match['score']['winner'] == 'DRAW':
                form1.append('D')
            else:
                form1.append('L')
        form_data[team1_name] = form1

        # Filter the fetched matches for head-to-head games
        for match in team1_matches:
            if (match['homeTeam']['id'] == team2_api_id or match['awayTeam']['id'] == team2_api_id) and len(h2h_results) < 5:
                h2h_results.append({
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'],
                    'away_score': match['score']['fullTime']['away']
                })
        
        # Fetch form for Team 2
        url2 = f"{BASE_URL}/teams/{team2_api_id}/matches?status=FINISHED&limit=5&competitions={PREMIER_LEAGUE_ID}"
        team2_response = requests.get(url2, headers=HEADERS)
        team2_response.raise_for_status()
        team2_matches = team2_response.json().get('matches', [])

        form2 = []
        for match in team2_matches:
            if (match['score']['winner'] == 'HOME_TEAM' and match['homeTeam']['id'] == team2_api_id) or \
               (match['score']['winner'] == 'AWAY_TEAM' and match['awayTeam']['id'] == team2_api_id):
                form2.append('W')
            elif match['score']['winner'] == 'DRAW':
                form2.append('D')
            else:
                form2.append('L')
        form_data[team2_name] = form2

    except requests.exceptions.RequestException as e:
        print(f"Could not fetch form/h2h data: {e}")
        return {'form_data': {}, 'h2h_results': []}

    return {'form_data': form_data, 'h2h_results': h2h_results}

# --- UPDATED get_team_form HELPER FUNCTION ---
# It now handles teams not in the current season gracefully
# NEW: get_team_form helper function to fetch stats and last match date
def get_team_form(team_name):
    """Fetches the last 5 finished matches for a team and calculates form stats."""
    team_api_id = get_team_api_id(team_name)
    if not team_api_id:
        return {'avg_gs': 0, 'avg_gc': 0, 'avg_pts': 0, 'last_match_date': datetime.now()}

    url = f"{BASE_URL}/teams/{team_api_id}/matches"
    params = {'status': 'FINISHED', 'limit': 10, 'competitions': PREMIER_LEAGUE_ID}
    
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        matches = response.json().get('matches', [])
        
        goals_scored = []
        goals_conceded = []
        points = []
        last_match_date = datetime.now()

        if matches:
            last_match_date = datetime.strptime(matches[0]['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
            for match in matches[:5]:
                is_home = match['homeTeam']['id'] == team_api_id
                score = match['score']['fullTime']
                if score.get('home') is not None and score.get('away') is not None:
                    if is_home:
                        goals_scored.append(score['home'])
                        goals_conceded.append(score['away'])
                        if score['home'] > score['away']: points.append(3)
                        elif score['home'] == score['away']: points.append(1)
                        else: points.append(0)
                    else:
                        goals_scored.append(score['away'])
                        goals_conceded.append(score['home'])
                        if score['away'] > score['home']: points.append(3)
                        elif score['away'] == score['home']: points.append(1)
                        else: points.append(0)

        return {
            'avg_gs': np.mean(goals_scored) if goals_scored else 0,
            'avg_gc': np.mean(goals_conceded) if goals_conceded else 0,
            'avg_pts': np.mean(points) if points else 0,
            'last_match_date': last_match_date
        }
    except requests.exceptions.RequestException:
        return {'avg_gs': 0, 'avg_gc': 0, 'avg_pts': 0, 'last_match_date': datetime.now()}
# Home Route
@app.route("/")
def index():
    return render_template("layout.html")

# Team stats helper
def get_team_stats(team_name):
    # This function is left as-is, but it would also need the get_team_api_id helper if it was to be more robust
    standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
    response = requests.get(standings_url, headers=HEADERS)
    standings_data = response.json()

    team_id = None
    for standing in standings_data['standings'][0]['table']:
        if standing['team']['name'].lower() == team_name.lower():
            team_id = standing['team']['id']
            break
    if not team_id:
        raise ValueError(f"Team {team_name} not found")

    matches_url = f"{BASE_URL}/teams/{team_id}/matches"
    params = {'competitions': PREMIER_LEAGUE_ID, 'status': 'FINISHED', 'limit': 10}
    response = requests.get(matches_url, headers=HEADERS, params=params)
    matches_data = response.json()

    total_goals_scored = 0
    total_goals_conceded = 0
    wins = 0
    matches_played = 0

    for match in matches_data['matches']:
        if match['competition']['id'] == PREMIER_LEAGUE_ID:
            is_home = match['homeTeam']['id'] == team_id
            team_score = match['score']['fullTime']['home'] if is_home else match['score']['fullTime']['away']
            opponent_score = match['score']['fullTime']['away'] if is_home else match['score']['fullTime']['home']
            if team_score is not None and opponent_score is not None:
                total_goals_scored += team_score
                total_goals_conceded += opponent_score
                if team_score > opponent_score:
                    wins += 1
                matches_played += 1

    return {
        'goals_scored_avg': total_goals_scored / matches_played if matches_played else 0,
        'goals_conceded_avg': total_goals_conceded / matches_played if matches_played else 0,
        'win_rate': wins / matches_played if matches_played else 0,
        'matches_played': matches_played
    }
@app.route("/upcoming")
def fixtures():
    try:
        url = f"{BASE_URL}/competitions/PL/matches"
        params = {'status': 'SCHEDULED', 'limit': 10}
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        fixtures_list = []
        for match in data.get('matches', []):
            fixtures_list.append({
                'match_id': match['id'],
                'home_team_name': match['homeTeam']['name'],
                'away_team_name': match['awayTeam']['name'],
                'date': datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%A, %b %d')
            })
            
        return render_template("fixtures.html", fixtures=fixtures_list)
        
    except requests.exceptions.RequestException as e:
        flash(f"Could not fetch upcoming fixtures: {e}", "danger")
        return render_template("fixtures.html", fixtures=[])

@app.route("/rankings")
def ranking():
    standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
    response = requests.get(standings_url, headers=HEADERS)
    standings_data = response.json()
    standings = standings_data['standings'][0]['table']

    rankings = [{
        'position': pos + 1,
        'team_name': team['team']['name'],
        'played': team['playedGames'],
        'won': team['won'],
        'drawn': team['draw'],
        'lost': team['lost'],
        'points': team['points'],
        'goals_for': team['goalsFor'],
        'goals_against': team['goalsAgainst'],
        'goal_difference': team['goalDifference']
    } for pos, team in enumerate(standings)]

    return render_template("rankings.html", rankings=rankings)

@app.route("/results")
def results():
    try:
        results_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/matches"
        params = {"status": "FINISHED", "limit": 10}
        response = requests.get(results_url, headers=HEADERS, params=params)
        matches_data = response.json()

        recent_results = []
        for match in matches_data['matches']:
            date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
            recent_results.append({
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'home_score': match['score']['fullTime']['home'],
                'away_score': match['score']['fullTime']['away'],
                'match_date': date.strftime('%B %d, %Y')
            })

        return render_template("results.html", results=recent_results)
    except Exception as e:
        flash(f"Error fetching match results: {e}")
        return redirect("/")

@app.route("/live")
def live_matches():
    try:
        live_url = f"{BASE_URL}/matches"
        params = {"status": "LIVE"}
        response = requests.get(live_url, headers=HEADERS, params=params)
        live_data = response.json()

        live_matches = [{
            'home_team': match['homeTeam']['name'],
            'away_team': match['awayTeam']['name'],
            'home_score': match['score']['fullTime']['home'],
            'away_score': match['score']['fullTime']['away']
        } for match in live_data.get('matches', [])]

        return render_template("live.html", matches=live_matches)
    except Exception as e:
        flash(f"Error fetching live matches: {e}")
        return redirect("/")

@app.route("/polls", methods=["GET", "POST"])
def polls():
    connection = get_db()
    if request.method == "POST":
        question = request.form.get("question")
        options = request.form.getlist("options")
        if not question or not options:
            flash("Poll question and options are required")
            return redirect("/polls")
        connection.execute("INSERT INTO polls (question) VALUES (?)", (question,))
        poll_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
        for option in options:
            connection.execute("INSERT INTO poll_options (poll_id, option_text) VALUES (?, ?)", (poll_id, option))
        connection.commit()
        flash("Poll created successfully!")
        return redirect("/polls")
    else:
        polls = connection.execute("SELECT * FROM polls").fetchall()
        return render_template("polls.html", polls=polls)

@app.route("/players")
def player_stats():
    try:
        scorers_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/scorers"
        scorers_response = requests.get(scorers_url, headers=HEADERS)
        scorers_data = scorers_response.json()

        standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
        standings_response = requests.get(standings_url, headers=HEADERS)
        standings_data = standings_response.json()

        teams_dict = {
            team['team']['id']: {
                'matches_played': team['playedGames'],
                'goals_for': team['goalsFor'],
                'goals_against': team['goalsAgainst']
            }
            for team in standings_data['standings'][0]['table']
        }

        top_players = []
        for scorer in scorers_data['scorers']:
            player = scorer['player']
            team = scorer['team']
            stats = teams_dict.get(team['id'], {})
            matches_played = stats.get('matches_played', 0)
            goals = scorer['goals']
            goals_per_game = round(goals / matches_played, 2) if matches_played else 0

            player_stats = {
                'name': player['name'],
                'position': player.get('position', 'N/A'),
                'nationality': player.get('nationality', 'N/A'),
                'team': team['name'],
                'goals': goals,
                'goals_per_game': goals_per_game
            }
            top_players.append(player_stats)

        top_players.sort(key=lambda x: x['goals'], reverse=True)
        return render_template("players.html", players=top_players)
    except Exception as e:
        flash(f"Error fetching player stats: {e}")
        return redirect("/")

# ... (all other code, including imports, helper functions, and other routes, remains the same) ...

@app.route("/predict", methods=["GET", "POST"])
def predict_match():
    if model is None or team_map is None:
        flash("Prediction functionality is currently unavailable.", "danger")
        return render_template("prediction.html", teams=[])

    sorted_teams = sorted(team_map.keys())

    if request.method == "POST":
        try:
            home_team_name = request.form.get('home_team_name')
            away_team_name = request.form.get('away_team_name')
            
            if not home_team_name or not away_team_name:
                flash("Please select both a home and away team.", "warning")
                return render_template("prediction.html", teams=sorted_teams, prediction=None)

            if home_team_name == away_team_name:
                flash("A team cannot play against itself.", "warning")
                return render_template("prediction.html", teams=sorted_teams, prediction=None)

        except (TypeError, ValueError):
            flash("Invalid team selection.", "warning")
            return render_template("prediction.html", teams=sorted_teams, prediction=None)

        home_team_id_for_model = team_map.get(home_team_name, 999)
        away_team_id_for_model = team_map.get(away_team_name, 999)
        
        if home_team_id_for_model == 999 or away_team_id_for_model == 999:
            flash("One or both teams not in historical data. Prediction will be based on recent form only.", "warning")

        referee_encoded = -1

        # Get live form stats and last match date
        home_form = get_team_form(home_team_name)
        away_form = get_team_form(away_team_name)
        
        # --- NEW: Calculate the additional 5 features ---
        travel_distance = compute_distance_km(home_team_name, away_team_name)
        
        today = datetime.now()
        rest_days_home = (today - home_form['last_match_date']).days
        rest_days_away = (today - away_form['last_match_date']).days
        
        h2h_avg_goal_diff = get_h2h_goal_diff(home_team_name, away_team_name)
        team_strength_gap = home_form['avg_pts'] - away_form['avg_pts']

        # Create the feature vector with all 14 features
        features = [
            home_team_id_for_model, away_team_id_for_model, referee_encoded,
            home_form['avg_gs'], home_form['avg_gc'], home_form['avg_pts'],
            away_form['avg_gs'], away_form['avg_gc'], away_form['avg_pts'],
            team_strength_gap, travel_distance,
            rest_days_home, rest_days_away,
            h2h_avg_goal_diff,
            1
        ]
        
        prediction_input = pd.DataFrame([features], columns=[
            'home_team_id', 'away_team_id', 'referee_encoded',
            'home_avg_gs', 'home_avg_gc', 'home_avg_pts',
            'away_avg_gs', 'away_avg_gc', 'away_avg_pts',
            'team_strength_gap', 'travel_distance',
            'rest_days_home', 'rest_days_away',
            'h2h_avg_goal_diff'
            ,    'is_home' 
        ])
        
        probabilities = model.predict_proba(prediction_input)[0]
        prediction_outcome = model.predict(prediction_input)[0]
        
        result_map = {2: "Home Win", 1: "Draw", 0: "Away Win"}
        
        # Get insights for the HTML page
        insights = get_form_and_h2h(home_team_name, away_team_name)
        
        prediction_data = {
            'home_team': home_team_name,
            'away_team': away_team_name,
            'prediction': result_map[prediction_outcome],
            'prob_home_win': round(probabilities[2] * 100),
            'prob_draw': round(probabilities[1] * 100),
            'prob_away_win': round(probabilities[0] * 100),
            'home_form_list': insights.get('form_data', {}).get(home_team_name, []),
            'away_form_list': insights.get('form_data', {}).get(away_team_name, []),
            'h2h': insights.get('h2h_results', [])
        }
        
        return render_template("prediction.html", teams=sorted_teams, prediction=prediction_data)
        
    else:
        return render_template("prediction.html", teams=sorted_teams, prediction=None)

# ... (all other code remains the same) ...

#if __name__ == "__main__":
    #app.run(debug=True)