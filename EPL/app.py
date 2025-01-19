import os
from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import sqlite3
import requests
import pandas as pd
from datetime import datetime
import numpy as np
  


# Configure application
app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# API Configuration
api_key = "46133a431ef84980bf20ed5eef34949e"
BASE_URL = "http://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": api_key}
PREMIER_LEAGUE_ID = 2021

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Establish a connection with the database
def get_db():
    connection = sqlite3.connect("predictor.db")
    connection.row_factory = sqlite3.Row
    return connection


# Get Team Stats
def get_team_stats(team_name):
    """Fetch team statistics from the API"""
    # Get current season's standings
    standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
    response = requests.get(standings_url, headers=HEADERS)
    standings_data = response.json()

    # Find the team ID from standings
    team_id = None
    for standing in standings_data['standings'][0]['table']:
        if standing['team']['name'].lower() == team_name.lower():
            team_id = standing['team']['id']
            break

    if not team_id:
        raise ValueError(f"Team {team_name} not found in current standings")

    # Get team's matches
    matches_url = f"{BASE_URL}/teams/{team_id}/matches"
    params = {
        'competitions': PREMIER_LEAGUE_ID,
        'status': 'FINISHED',
        'limit': 10  # Last 10 matches
    }
    response = requests.get(matches_url, headers=HEADERS, params=params)
    matches_data = response.json()

    # Calculate statistics
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
                if (is_home and team_score > opponent_score) or (not is_home and team_score > opponent_score):
                    wins += 1
                matches_played += 1

    return {
        'goals_scored_avg': total_goals_scored / matches_played if matches_played > 0 else 0,
        'goals_conceded_avg': total_goals_conceded / matches_played if matches_played > 0 else 0,
        'win_rate': wins / matches_played if matches_played > 0 else 0,
        'matches_played': matches_played
    }

# Home Route
@app.route("/")
def index():
    return render_template("layout.html")

# Logout Route
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# Register Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirmation = request.form.get('confirmation')

        if not username or not email or not password or not confirmation:
            return "All fields are required!"
        if password != confirmation:
            return "Passwords do not match!"

        password_hash = generate_password_hash(password)
        connection = get_db()
        try:
            connection.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", 
                            (username, password_hash, email))
            connection.commit()
        except sqlite3.IntegrityError:
            return "Username or email already exists!"
        finally:
            connection.close()
        
        return redirect('/')
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    session.clear()

    if request.method == "POST":
        if not request.form.get("username"):
            return ("must provide username", 403)
        elif not request.form.get("password"):
            return ("must provide password", 403)

        try:
            connection = get_db()
            rows = connection.execute("SELECT * FROM users WHERE username = ?", 
                                   (request.form.get("username"),)).fetchall()
            connection.close()

            if len(rows) != 1 or not check_password_hash(rows[0]["password"], 
                                                       request.form.get("password")):
                return ("invalid username and/or password", 403)

            session["user_id"] = rows[0]["id"]
            return redirect("/")

        except Exception as e:
            return (f"An error occurred: {str(e)}", 500)

    else:
        return render_template("login.html")
    
# Fetch upcoming Premier League fixtures
@app.route("/upcoming", methods=["GET", "POST"])
def fixtures():
    try:
        # URL to fetch upcoming matches
        fixtures_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/matches"
        # Get the next 10 upcoming matches
        params = {
            'status': 'SCHEDULED',  # We want upcoming, scheduled matches
            'limit': 10  # Limit the number of matches to fetch
        }
        response = requests.get(fixtures_url, headers=HEADERS, params=params)
        fixtures_data = response.json()

        # Format the fixtures into a list
        upcoming_fixtures = []
        for match in fixtures_data['matches']:
            match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            match_time = match_date.strftime('%A, %B %d, %Y at %I:%M %p')
            
            upcoming_fixtures.append({
                'home_team': home_team,
                'away_team': away_team,
                'match_time': match_time
            })
        
        # Pass the fixtures to the template
        return render_template("fixtures.html", fixtures=upcoming_fixtures)

    except Exception as e:
        flash(f"Error fetching upcoming fixtures: {str(e)}")
        return redirect("/")

@app.route("/rankings", methods=["GET", "POST"])
def ranking():
    # Fetch current Premier League standings
    standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
    response = requests.get(standings_url, headers=HEADERS)
    standings_data = response.json()
    
    # Extract relevant data for the leaderboard
    standings = standings_data['standings'][0]['table']

    # Prepare a list to send to the template
    rankings = []
    for position, team in enumerate(standings, start=1):
        rankings.append({
            'position': position,
            'team_name': team['team']['name'],
            'played': team['playedGames'],
            'won': team['won'],
            'drawn': team['draw'],
            'lost': team['lost'],
            'points': team['points'],
            'goals_for': team['goalsFor'],
            'goals_against': team['goalsAgainst'],
            'goal_difference': team['goalDifference']
        })

    return render_template("rankings.html", rankings=rankings)


import os
from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import sqlite3
import requests
import pandas as pd
from datetime import datetime
import numpy as np


# New: Fetch recent match results
@app.route("/results")
def results():
    try:
        results_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/matches"
        params = {"status": "FINISHED", "limit": 10}  # Last 10 matches
        response = requests.get(results_url, headers=HEADERS, params=params)
        matches_data = response.json()

        recent_results = []
        for match in matches_data['matches']:
            match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            home_score = match['score']['fullTime']['home']
            away_score = match['score']['fullTime']['away']
            recent_results.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'match_date': match_date.strftime('%B %d, %Y')
            })

        return render_template("results.html", results=recent_results)
    except Exception as e:
        flash(f"Error fetching match results: {str(e)}")
        return redirect("/")


# New: Fetch live matches
@app.route("/live")
def live_matches():
    try:
        live_url = f"{BASE_URL}/matches"
        params = {"status": "LIVE"}
        response = requests.get(live_url, headers=HEADERS, params=params)
        live_data = response.json()

        live_matches = []
        for match in live_data.get('matches', []):
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            home_score = match['score']['fullTime']['home']
            away_score = match['score']['fullTime']['away']
            live_matches.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score
            })

        return render_template("live.html", matches=live_matches)
    except Exception as e:
        flash(f"Error fetching live matches: {str(e)}")
        return redirect("/")



# New: Polling feature
@app.route("/polls", methods=["GET", "POST"])
def polls():
    connection = get_db()
    if request.method == "POST":
        question = request.form.get("question")
        options = request.form.getlist("options")
        if not question or not options:
            flash("Poll question and options are required")
            return redirect("/polls")
        # Save poll
        connection.execute("INSERT INTO polls (question) VALUES (?)", (question,))
        poll_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
        for option in options:
            connection.execute("INSERT INTO poll_options (poll_id, option_text) VALUES (?, ?)", (poll_id, option))
        connection.commit()
        flash("Poll created successfully!")
        return redirect("/polls")
    else:
        # Display polls
        polls = connection.execute("SELECT * FROM polls").fetchall()
        return render_template("polls.html", polls=polls)

@app.route("/players")
def player_stats():
    try:
        # Get top scorers
        scorers_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/scorers"
        scorers_response = requests.get(scorers_url, headers=HEADERS)
        scorers_data = scorers_response.json()

        # Get team standings to get team stats
        standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
        standings_response = requests.get(standings_url, headers=HEADERS)
        standings_data = standings_response.json()

        # Create team dictionary for quick lookup
        teams_dict = {}
        for standing in standings_data['standings'][0]['table']:
            team_id = standing['team']['id']
            teams_dict[team_id] = {
                'matches_played': standing['playedGames'],
                'goals_for': standing['goalsFor'],
                'goals_against': standing['goalsAgainst']
            }

        top_players = []
        for scorer in scorers_data['scorers']:
            player = scorer['player']
            team = scorer['team']
            team_stats = teams_dict.get(team['id'], {})
            matches_played = team_stats.get('matches_played', 0)

            # Calculate goals per game
            goals = scorer['goals']
            goals_per_game = round(goals / matches_played, 2) if matches_played > 0 else 0

            # Get player matches for detailed stats
            player_matches_url = f"{BASE_URL}/players/{player['id']}/matches"
            matches_response = requests.get(player_matches_url, headers=HEADERS)
            
            if matches_response.status_code == 200:
                matches_data = matches_response.json()
                premier_league_matches = [
                    match for match in matches_data['matches'] 
                    if match['competition']['id'] == PREMIER_LEAGUE_ID
                ]
                
                minutes_played = sum(match.get('minutesPlayed', 0) for match in premier_league_matches)
                goals_per_90 = round((goals * 90) / minutes_played, 2) if minutes_played > 0 else 0
                appearances = len(premier_league_matches)
            else:
                minutes_played = 0
                goals_per_90 = 0
                appearances = 0

            player_stats = {
                'id': player['id'],
                'name': player['name'],
                'position': player.get('position', 'N/A'),
                'nationality': player.get('nationality', 'N/A'),
                'team': team['name'],
                'team_id': team['id'],
                
                # Available scoring stats
                'goals': goals,
                'goals_per_game': goals_per_game,
                'goals_per_90': goals_per_90,
                
                # Match participation stats
                'appearances': appearances,
                'minutes_played': minutes_played,
                
                # Team context stats
                'team_games': matches_played,
                'team_goals_scored': team_stats.get('goals_for', 0),
                'team_goals_conceded': team_stats.get('goals_against', 0),
                
                # Calculate percentage of team goals scored by player
                'percentage_team_goals': round(
                    (goals / team_stats.get('goals_for', 1)) * 100, 1
                ) if team_stats.get('goals_for', 0) > 0 else 0
            }
            top_players.append(player_stats)

        # Sort players by goals
        top_players.sort(key=lambda x: x['goals'], reverse=True)
        
        return render_template("players.html", players=top_players)
    except Exception as e:
        flash(f"Error fetching player stats: {str(e)}")
        return redirect("/")


@app.route("/charts")
def charts():
    try:
        # Fetch standings data
        standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
        standings_response = requests.get(standings_url, headers=HEADERS)
        standings_data = standings_response.json()

        # Fetch top scorers
        scorers_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/scorers"
        scorers_response = requests.get(scorers_url, headers=HEADERS)
        scorers_data = scorers_response.json()

        # Fetch matches
        matches_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/matches"
        matches_response = requests.get(matches_url, headers=HEADERS)
        matches_data = matches_response.json()

        # Process standings data
        standings = standings_data['standings'][0]['table']
        team_stats = {
            'teams': [team['team']['name'] for team in standings],
            'points': [team['points'] for team in standings],
            'goals_for': [team['goalsFor'] for team in standings],
            'goals_against': [team['goalsAgainst'] for team in standings],
            'goal_diff': [team['goalDifference'] for team in standings],
            'wins': [team['won'] for team in standings],
            'draws': [team['draw'] for team in standings],
            'losses': [team['lost'] for team in standings]
        }

        # Process top scorers data
        top_scorers = {
            'names': [scorer['player']['name'] for scorer in scorers_data['scorers'][:10]],
            'goals': [scorer['goals'] for scorer in scorers_data['scorers'][:10]],
            'teams': [scorer['team']['name'] for scorer in scorers_data['scorers'][:10]]
        }

        # Process matches data for form analysis
        matches = matches_data['matches']
        recent_matches = [match for match in matches if match['status'] == 'FINISHED'][-50:]  # Last 50 matches
        
        # Calculate match statistics
        total_goals = sum(match['score']['fullTime']['home'] + match['score']['fullTime']['away'] 
                         for match in recent_matches)
        avg_goals = round(total_goals / len(recent_matches), 2)
        
        matches_with_both_scored = sum(1 for match in recent_matches 
                                     if match['score']['fullTime']['home'] > 0 
                                     and match['score']['fullTime']['away'] > 0)
        btts_percentage = round((matches_with_both_scored / len(recent_matches)) * 100, 1)

        match_stats = {
            'total_matches': len(recent_matches),
            'avg_goals': avg_goals,
            'btts_percentage': btts_percentage
        }

        # Calculate home vs away statistics
        home_wins = sum(1 for match in recent_matches 
                       if match['score']['fullTime']['home'] > match['score']['fullTime']['away'])
        away_wins = sum(1 for match in recent_matches 
                       if match['score']['fullTime']['home'] < match['score']['fullTime']['away'])
        draws = sum(1 for match in recent_matches 
                   if match['score']['fullTime']['home'] == match['score']['fullTime']['away'])

        venue_stats = {
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws
        }

        return render_template("charts.html",
                             team_stats=team_stats,
                             top_scorers=top_scorers,
                             match_stats=match_stats,
                             venue_stats=venue_stats)

    except Exception as e:
        flash(f"Error fetching chart data: {str(e)}")
        return redirect("/")
    



if __name__ == "__main__":
    app.run(debug=True)