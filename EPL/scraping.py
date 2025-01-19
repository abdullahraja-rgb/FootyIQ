import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class PremierLeagueDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_key}
        self.current_teams = None  # Cache for current Premier League teams

    def get_current_teams(self):
        """Fetch and cache current Premier League teams."""
        if self.current_teams:
            return self.current_teams  # Return cached teams if available

        url = f"{self.base_url}/competitions/PL/teams"  # Updated competition code
        retries_left = 5

        while retries_left > 0:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                teams = response.json()['teams']
                # Normalize team names for comparison
                self.current_teams = {team['name'].strip().lower(): team['id'] for team in teams}
                return self.current_teams
            elif response.status_code == 429:  # Rate limit exceeded
                print("Rate limit reached. Waiting before retrying...")
                time.sleep(60)  # Wait for a minute before retrying
                retries_left -= 1
            else:
                raise Exception(f"Failed to fetch teams: {response.content}")

        raise Exception("Exceeded maximum retries while fetching current teams.")

    def get_team_id(self, team_name):
        """Fetch the team ID for the given team name."""
        teams = self.get_current_teams()
        normalized_team_name = team_name.strip().lower()
        if normalized_team_name in teams:
            return teams[normalized_team_name]
        raise ValueError(f"Team '{team_name}' not found in current Premier League standings")

    def get_team_stats(self, team_name):
        """Fetch statistics for a specific team."""
        try:
            team_id = self.get_team_id(team_name)
        except ValueError as e:
            print(e)
            return None  # Return None if the team is not found

        matches_url = f"{self.base_url}/teams/{team_id}/matches"
        params = {
            'competitions': 'PL',
            'status': 'FINISHED',
            'limit': 10  # Fetch last 10 matches
        }

        retries_left = 5
        while retries_left > 0:
            response = requests.get(matches_url, headers=self.headers, params=params)
            if response.status_code == 200:
                break
            elif response.status_code == 429:  # Rate limit exceeded
                print("Rate limit reached. Waiting before retrying...")
                time.sleep(60)  # Wait for a minute before retrying
                retries_left -= 1
            else:
                raise Exception(f"Failed to fetch matches for {team_name}: {response.content}")

        matches_data = response.json()

        # Calculate statistics based on match data
        total_goals_scored = 0
        total_goals_conceded = 0
        wins = 0
        matches_played = len(matches_data['matches'])

        for match in matches_data['matches']:
            if match['competition']['code'] == 'PL':
                is_home = match['homeTeam']['id'] == team_id
                team_score = match['score']['fullTime']['home'] if is_home else match['score']['fullTime']['away']
                opponent_score = match['score']['fullTime']['away'] if is_home else match['score']['fullTime']['home']

                if team_score is not None and opponent_score is not None:
                    total_goals_scored += team_score
                    total_goals_conceded += opponent_score
                    if (is_home and team_score > opponent_score) or (not is_home and team_score > opponent_score):
                        wins += 1

        return {
            'goals_scored_avg': total_goals_scored / matches_played if matches_played > 0 else 0,
            'goals_conceded_avg': total_goals_conceded / matches_played if matches_played > 0 else 0,
            'win_rate': wins / matches_played if matches_played > 0 else 0,
            'matches_played': matches_played
        }

    def get_season_matches(self, seasons):
        """Fetch the matches of a given season."""
        # Placeholder for actual implementation
        # This should return a DataFrame with match information such as home_team, away_team, and result
        # For simplicity, we assume this is coming from another API endpoint or prepared data
        data = {
            'home_team': ['Liverpool FC', 'Manchester United FC'],
            'away_team': ['Chelsea FC', 'Arsenal FC'],
            'result': ['H', 'A']
        }
        return pd.DataFrame(data)


class EnhancedMatchPredictor:
    def __init__(self, data_collector):
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = ['H', 'D', 'A']
        self.data_collector = data_collector
        self.team_stats = {}

    def train_model(self, X_train, y_train):
        pipeline = Pipeline([ 
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }

        self.model = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        self.model.fit(X_train, y_train)

        print("Best parameters found: ", self.model.best_params_)

    def predict(self, home_team, away_team, season=2023):
        return self.predict_match(home_team, away_team, season)

    def predict_match(self, home_team, away_team, season=2023):
        if self.model is None:
            raise ValueError("Model is not trained. Please train the model first using `train_model_with_data`.")
        
        try:
            home_team = self.data_collector.get_team_stats(home_team)
            away_team = self.data_collector.get_team_stats(away_team)
            
            home_stats = home_team
            away_stats = away_team

            match_data = np.array([[home_stats['matches_played'],
                                    away_stats['matches_played'],
                                    home_stats['win_rate'],
                                    away_stats['win_rate'],
                                    home_stats['goals_scored_avg'],
                                    away_stats['goals_scored_avg']]])

            print("Match data for prediction:", match_data)

            match_data_scaled = self.model.best_estimator_.named_steps['scaler'].transform(match_data)
            print("Scaled match data:", match_data_scaled)

            prediction = self.model.predict(match_data_scaled)
            print("Predicted match result:", prediction)

            return prediction[0]
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return str(e)

    def train_model_with_data(self, seasons=None):
        data = self.data_collector.get_season_matches(seasons)
        
        X = data[['home_team', 'away_team']].apply(self._prepare_features, axis=1)
        y = data['result']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.train_model(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    def _prepare_features(self, row):
        home_stats = self.data_collector.get_team_stats(row['home_team'])
        away_stats = self.data_collector.get_team_stats(row['away_team'])
        return pd.Series([ 
            home_stats['matches_played'],
            away_stats['matches_played'],
            home_stats['win_rate'],
            away_stats['win_rate'],
            home_stats['goals_scored_avg'],
            away_stats['goals_scored_avg']
        ])


if __name__ == "__main__":
    API_KEY = "46133a431ef84980bf20ed5eef34949e"  # Replace with your actual API key
    collector = PremierLeagueDataCollector(API_KEY)
    predictor = EnhancedMatchPredictor(collector)

    predictor.train_model_with_data()
    prediction = predictor.predict_match("Liverpool FC", "Manchester United FC")
    print(f"Prediction: {prediction}")

#------------------------------------------------------------------------------------------------------------------------------
'''The following code is responsible for getting predictions integrating sn ML Model in this which is a bit problematic
as the integration wont work properly for now'''
from datetime import datetime
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


collector = PremierLeagueDataCollector(api_key)
predictor = EnhancedMatchPredictor(collector)

# Train the model when the app starts.
try:
    predictor.train_model()
except Exception as e:
    logger.error(f"Error during model training: {str(e)}")

@app.route("/predictor", methods=["GET", "POST"])
def predicting():
    """Handle prediction requests."""
    if request.method == "POST":
        home_team = request.form.get("home_team", "").strip()
        away_team = request.form.get("away_team", "").strip()

        # Validation
        if not all([home_team, away_team]):
            flash("Please fill in both team names", "error")
            return redirect("/predictor")
        
        if home_team.lower() == away_team.lower():
            flash("Please select different teams", "error")
            return redirect("/predictor")
        
        try:
            # Get team statistics and make prediction
            prediction, confidence = predictor.predict_match(home_team, away_team)
            
            result = {
                'prediction': {
                    'H': 'Home Team Win',
                    'A': 'Away Team Win',
                    'D': 'Draw'
                }.get(prediction, 'Unknown'),
                'confidence': f"{confidence * 100:.1f}%"
            }
            
            # Get team statistics for display
            home_stats = collector.get_team_stats(home_team)
            away_stats = collector.get_team_stats(away_team)

            if home_stats is None or away_stats is None:
                raise ValueError(f"One of the teams '{home_team}' or '{away_team}' is not found in current standings.")

            # Log prediction
            logger.info(f"Prediction made for {home_team} vs {away_team}: {result['prediction']}")
            
            return render_template(
                "predictor.html",
                prediction=result,
                confidence=confidence,
                home_team=home_team,
                away_team=away_team,
                home_stats=home_stats,
                away_stats=away_stats,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        except ValueError as ve:
            logger.warning(f"Validation error: {str(ve)}")
            flash(str(ve), "error")
            return redirect("/predictor")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            flash(f"Error making prediction: {str(e)}", "error")
            return redirect("/predictor")

    # GET request - show form with available teams
    try:
        available_teams = sorted(collector.get_current_teams().keys())
        logger.info(f"Available teams: {available_teams}")
        return render_template("predictor.html", teams=available_teams)
    except Exception as e:
        logger.error(f"Error loading teams: {str(e)}")
        flash("Error loading available teams", "error")
        return render_template("predictor.html")



@app.route("/get_teams")
def get_teams():
    """Provide autocomplete suggestions for team names."""
    query = request.args.get('q', '').lower()
    try:
        teams = collector.get_current_teams()  # Get current teams from the collector
        matching_teams = [team for team in teams.keys() if query in team.lower()]
        return jsonify(matching_teams)
    except Exception as e:
        logger.error(f"Error fetching teams for autocomplete: {str(e)}")
        return jsonify([]), 500