
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier for machine learning
from sklearn.metrics import accuracy_score, precision_score  # Import functions to evaluate the model

# Read the CSV file containing match data
matches = pd.read_csv("/workspaces/Final-Project/EPL/Prediction/matches.csv", index_col=0)  # Load the data and use the first column as the index

# Data preprocessing (preparing the data for the model)
matches["date"] = pd.to_datetime(matches["date"])  # Convert the 'date' column to datetime format
matches["h/a"] = matches["venue"].map({"Home": 1, "Away": 0})  # Convert venue to 1 (Home) and 0 (Away)
matches["opp"] = matches["opponent"].astype("category").cat.codes  # Convert 'opponent' names to numbers (categorical encoding)
matches["hour"] = pd.to_datetime(matches["time"]).dt.hour  # Extract the hour from the 'time' column
matches["day"] = matches["date"].dt.dayofweek  # Extract the day of the week from the 'date' column (0 = Monday, 6 = Sunday)
matches["target"] = (matches["result"] == "W").astype("int")  # Create a target column: 1 for Win ('W'), 0 for other results

# Define the features (predictors) that we will use to make predictions
predictors = ["h/a", "opp", "hour", "day"]

# Function to calculate rolling averages (the average over the last 3 matches)
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")  # Sort the data by date to get the correct order of matches
    rolling_stats = group[cols].rolling(3, closed='left').mean()  # Calculate the rolling average over 3 matches (excluding the current match)
    group[new_cols] = rolling_stats  # Add the new columns for the rolling averages
    return group.dropna(subset=new_cols)  # Drop any rows with missing data in the new rolling average columns

# Apply rolling averages to certain columns for each team
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]  # List of columns to calculate rolling averages for
new_cols = [f"{c}_rolling" for c in cols]  # Create new column names like "gf_rolling", "ga_rolling", etc.
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))  # Apply the rolling averages function for each team
matches_rolling = matches_rolling.reset_index(drop=True)  # Reset the index after grouping

# Prepare the data for training and testing
train = matches_rolling[matches_rolling["date"] < '2022-01-01']  # Use matches before 2022 for training
test = matches_rolling[matches_rolling["date"] >= '2022-01-01']  # Use matches from 2022 onwards for testing

# Train the machine learning model (Random Forest)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)  # Create a Random Forest model
rf.fit(train[predictors + new_cols], train["target"])  # Train the model using the predictors and target column

# Make predictions using the trained model
preds = rf.predict(test[predictors + new_cols])  # Predict the outcomes of the test set matches

# Combine the predictions with the actual results
combined = pd.DataFrame({
    "date": test["date"],  # The date of each match
    "team": test["team"],  # The team that played
    "opponent": test["opponent"],  # The opponent team
    "actual": test["target"],  # The actual result (1 for Win, 0 for other outcomes)
    "prediction": preds,  # The predicted result (1 for Win, 0 for other outcomes)
    "result": test["result"]  # The actual result as text (Win, Loss, or Draw)
})

# Function to display the predictions for each match
def displaypredictions(data):
    for _, row in data.iterrows():  # Loop through each row of the predictions data
        print(f"Match on {row['date'].strftime('%Y-%m-%d')}: {row['team']} vs {row['opponent']}")  # Print the match date and teams
        print(f"Prediction: {row['team']} {'win' if row['prediction'] == 1 else 'not win'}")  # Print predicted outcome
        print(f"Actual result: {row['result']}")  # Print actual result
        print("------------------------")  # Separate each prediction with a line

# Display the predictions for the test set
print("Match Predictions:")
displaypredictions(combined)

# Calculate and display the accuracy and precision of the model
accuracy = accuracy_score(combined["actual"], combined["prediction"])  # Calculate accuracy (correct predictions)
precision = precision_score(combined["actual"], combined["prediction"])  # Calculate precision (how many predicted wins were actual wins)
print(f"\nOverall Prediction Accuracy: {accuracy:.2%}")  # Print accuracy as a percentage
print(f"Precision Score: {precision:.2%}")  # Print precision score as a percentage
