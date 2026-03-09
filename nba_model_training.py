import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def add_rolling_features(nba, window=10):
    """
    this calculates 10-game rolling averages for team stats.
    Uses shift(1) to prevent data leakage.
    """
    #sortting chronologically
    nba = nba.sort_values('GAME_DATE').copy()
    
    features_to_roll = ['OFF_RTG', 'EFG']
    
    #rolling the stats for the home team and away team separately
    for col in features_to_roll:
        #grouping by Team ID, shift by 1 game, then average the last 'window' games
        nba[f'ROLL_HOME_{col}'] = nba.groupby('HOME_TEAM_ID')[f'HOME_{col}'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=5).mean()
        )
        nba[f'ROLL_AWAY_{col}'] = nba.groupby('AWAY_TEAM_ID')[f'AWAY_{col}'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=5).mean()
        )
        
    #rolling the PACE metric (using the Home team as the anchor for simplicity here)
    nba['ROLL_GAME_PACE'] = nba.groupby('HOME_TEAM_ID')['GAME_PACE'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=5).mean()
    )

    #droping rows that don't have enough history (the first few games of the dataset)
    nba = nba.dropna()
    return nba

def train_model(nba):
    #isolating the rolling features to use as predictors
    features = [c for c in nba.columns if 'ROLL_' in c]
    target = 'POINT_DIFF'
    
    print(f"Training on features: {features}")
    
    #time-Based split (80% Train, 20% Test)
    split_index = int(len(nba) * 0.8)
    
    X_train = nba[features].iloc[:split_index]
    y_train = nba[target].iloc[:split_index]
    
    X_test = nba[features].iloc[split_index:]
    y_test = nba[target].iloc[split_index:]
    
    #initialising and training the XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    #evaluating the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\n--- Model Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    
    return model

if __name__ == "__main__":
    print("Loading data and calculating rolling averages...")
    nba = pd.read_csv('nba_training_set.csv')
    nba['GAME_DATE'] = pd.to_datetime(nba['GAME_DATE'])
    
    nba_rolled = add_rolling_features(nba)
    
    print("Training XGBoost Model...")
    model = train_model(nba_rolled)
    
    #save the model for deployment
    model.save_model("nba_spread_predictor.json")
    print("\nModel successfully saved to nba_spread_predictor.json")