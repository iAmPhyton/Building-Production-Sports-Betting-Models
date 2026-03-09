from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

#initialising API app
app = FastAPI(title="NBA Spread Predictor API", version="1.0")

#loading trained XGBoost model
print("Loading model...")
model = xgb.XGBRegressor()
model.load_model("nba_spread_predictor.json")

#defining the exact data structure the API expects to receive
class GameStats(BaseModel):
    ROLL_HOME_OFF_RTG: float
    ROLL_AWAY_OFF_RTG: float
    ROLL_HOME_EFG: float
    ROLL_AWAY_EFG: float
    ROLL_GAME_PACE: float

#creating the prediction endpoint
@app.post("/predict")
def predict_spread(stats: GameStats):
    """
    Receives rolling stats for a matchup and returns the predicted point differential.
    """
    #converting the incoming JSON payload into a Pandas DataFrame
    input_data = pd.DataFrame([stats.model_dump()])
    
    #running the model inference
    prediction = model.predict(input_data)[0]
    
    #interpreting the result (Positive = Home Win, Negative = Away Win)
    winner = "Home Team" if prediction > 0 else "Away Team"
    margin = abs(float(prediction))
    
    #returning a clean JSON response
    return {
        "status": "success",
        "predicted_winner": winner,
        "predicted_margin": round(margin, 2),
        "raw_point_diff": round(float(prediction), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 