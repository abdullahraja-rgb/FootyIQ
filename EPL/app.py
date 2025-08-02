import os
import sqlite3
import requests
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, jsonify
from flask_session import Session
from datetime import datetime

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

# Home Route
@app.route("/")
def index():
    return render_template("layout.html")

# Logout (clears session variables only)
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# Team stats helper
def get_team_stats(team_name):
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

@app.route("/upcoming", methods=["GET"])
def fixtures():
    try:
        fixtures_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/matches"
        params = {'status': 'SCHEDULED', 'limit': 10}
        response = requests.get(fixtures_url, headers=HEADERS, params=params)
        fixtures_data = response.json()

        upcoming_fixtures = []
        for match in fixtures_data['matches']:
            match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
            upcoming_fixtures.append({
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'match_time': match_date.strftime('%A, %B %d, %Y at %I:%M %p')
            })

        return render_template("fixtures.html", fixtures=upcoming_fixtures)

    except Exception as e:
        flash(f"Error fetching fixtures: {e}")
        return redirect("/")

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

@app.route("/charts")
def charts():
    try:
        standings_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/standings"
        standings_response = requests.get(standings_url, headers=HEADERS)
        standings_data = standings_response.json()

        scorers_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/scorers"
        scorers_response = requests.get(scorers_url, headers=HEADERS)
        scorers_data = scorers_response.json()

        matches_url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE_ID}/matches"
        matches_response = requests.get(matches_url, headers=HEADERS)
        matches_data = matches_response.json()

        standings = standings_data['standings'][0]['table']
        team_stats = {
            'teams': [t['team']['name'] for t in standings],
            'points': [t['points'] for t in standings],
        }

        top_scorers = {
            'names': [s['player']['name'] for s in scorers_data['scorers'][:10]],
            'goals': [s['goals'] for s in scorers_data['scorers'][:10]]
        }

        recent_matches = [m for m in matches_data['matches'] if m['status'] == 'FINISHED'][-50:]
        total_goals = sum(m['score']['fullTime']['home'] + m['score']['fullTime']['away'] for m in recent_matches)
        avg_goals = round(total_goals / len(recent_matches), 2)

        return render_template("charts.html", team_stats=team_stats, top_scorers=top_scorers, avg_goals=avg_goals)
    except Exception as e:
        flash(f"Error fetching chart data: {e}")
        return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
