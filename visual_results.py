import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

#setting visual style
sns.set_theme(style="whitegrid")

def generate_visualizations():
    print("Loading data and model...")
    #loading engineered data
    nba = pd.read_csv('nba_training_set.csv')
    nba['GAME_DATE'] = pd.to_datetime(nba['GAME_DATE'])
    nba = nba.sort_values('GAME_DATE').copy()
    
    #re-calculate the rolling features so they are consistent with the training data
    features_to_roll = ['OFF_RTG', 'EFG']
    for col in features_to_roll:
        nba[f'ROLL_HOME_{col}'] = nba.groupby('HOME_TEAM_ID')[f'HOME_{col}'].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
        nba[f'ROLL_AWAY_{col}'] = nba.groupby('AWAY_TEAM_ID')[f'AWAY_{col}'].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    nba['ROLL_GAME_PACE'] = nba.groupby('HOME_TEAM_ID')['GAME_PACE'].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    nba = nba.dropna()

    # 2. Isolate features and target, recreate the 80/20 split
    features = [c for c in nba.columns if 'ROLL_' in c]
    split_index = int(len(nba) * 0.8)
    
    X_test = nba[features].iloc[split_index:]
    y_test = nba['POINT_DIFF'].iloc[split_index:]

    #loading trained model
    model = xgb.XGBRegressor()
    model.load_model("nba_spread_predictor.json")
    
    #generating predictions for scatter plot
    predictions = model.predict(X_test)

    #PLOT 1: Feature Importance
    print("Generating Feature Importance Plot...")
    plt.figure(figsize=(10, 6))
    #XGBoost's built-in plot_importance
    xgb.plot_importance(model, max_num_features=5, importance_type='weight', 
                        title='XGBoost Feature Importance', xlabel='F-Score (Weight)', 
                        ylabel='Features', grid=False)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()

    #PLOT 2: actual vs predicted
    print("Generating Actual vs Predicted Plot...")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.5, color='blue')
    
    #adding prediction diagonal line
    plt.plot([-40, 40], [-40, 40], color='red', linestyle='--')
    
    plt.title('Actual vs. Predicted Point Differential', fontsize=14)
    plt.xlabel('Actual Point Differential', fontsize=12)
    plt.ylabel('Predicted Point Differential', fontsize=12)
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.close()

    print("Success! Saved 'feature_importance.png' and 'actual_vs_predicted.png' to local folder.")

if __name__ == "__main__":
    generate_visualizations()