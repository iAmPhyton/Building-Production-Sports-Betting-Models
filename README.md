NBA Spread Predictor: End-to-End Betting Model API

Overview:
This project is an end-to-end machine learning pipeline that predicts NBA point spreads. It demonstrates the ability to extract raw sports data, engineer advanced predictive metrics, train an XGBoost model while strictly preventing data leakage, and deploy the resulting model as a live API using FastAPI.

The core objective is to identify +EV (Expected Value) betting opportunities by comparing the model's predicted point differential against bookmaker lines.

Architecture & Pipeline:
The project is broken down into four scalable phases:

1. Data Ingestion (`nba.py`): Automates the extraction of raw game logs (over 7,000 games across 5 seasons) using the official `nba_api`.
2. Feature Engineering (`nba_feature_engineering.py`): Transforms raw box scores into advanced metrics (Offensive Rating, Effective Field Goal %, Pace). Normalizes data to create direct head-to-head matchup rows.
3. Model Training (`nba_model_training.py`): 
    * Data Leakage Prevention: Implements chronological 10-game rolling averages. The model is strictly trained on historical data available *prior* to tip-off.
    * Algorithm: Trains an `XGBRegressor` on the rolling features to predict the final point differential.
    * Evaluation: Evaluated using Mean Absolute Error (MAE) with a strict time-based train/test split (no random shuffling).
4. Deployment (`nba_app.py`): Wraps the trained model in a FastAPI application, providing a highly performant, queryable REST endpoint for real-time inference.

Tech Used:
* Data Processing: Python, Pandas, NumPy
* Machine Learning: XGBoost, Scikit-learn
* Data Extraction: `nba_api`
* Deployment: FastAPI, Uvicorn, Pydantic

How to Run the API Locally:
- Start the FastAPI server: `python nba_app.py`
- Navigate to `http://localhost:8000/docs` to access the interactive Swagger UI and test the `/predict` endpoint.

Future Enhancements:
* Integrate real-time odds fetching via external sportsbook APIs.
* Containerize the application using Docker for cloud deployment (AWS/GCP).
* Implement automated retraining pipelines via GitHub Actions.
