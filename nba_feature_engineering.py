import pandas as pd

def process_nba_data(input_file='nba_games_last_5_years.csv'):
    #loading raw data
    nba = pd.read_csv(input_file)
    
    #identifying home and away teams
    #'vs.' indicates a Home game, '@' indicates an Away game
    nba['IS_HOME'] = nba['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    #splitting into home and away dataframes
    home_teams = nba[nba['IS_HOME'] == 1].copy()
    away_teams = nba[nba['IS_HOME'] == 0].copy()
    
    #prefix columns to keep them organized after merging
    home_teams.columns = ['HOME_' + str(col) for col in home_teams.columns]
    away_teams.columns = ['AWAY_' + str(col) for col in away_teams.columns]
    
    #merging on GAME_ID to get one row per matchup
    game_nba = pd.merge(
        home_teams, 
        away_teams, 
        left_on='HOME_GAME_ID', 
        right_on='AWAY_GAME_ID'
    )
    
    return game_nba

def calculate_advanced_metrics(nba):
    """
    this calculates PACE, Offensive Rating, and Effective Field Goal %.
    """
    #calculating estimated possessions
    nba['HOME_POSS'] = nba['HOME_FGA'] + 0.44 * nba['HOME_FTA'] + nba['HOME_TOV'] - nba['HOME_OREB']
    nba['AWAY_POSS'] = nba['AWAY_FGA'] + 0.44 * nba['AWAY_FTA'] + nba['AWAY_TOV'] - nba['AWAY_OREB']
    
    #average possessions per game (PACE)
    nba['GAME_PACE'] = 48 * ((nba['HOME_POSS'] + nba['AWAY_POSS']) / (2 * (nba['HOME_MIN'] / 5)))
    
    #offensive rating (points per 100 possessions)
    nba['HOME_OFF_RTG'] = 100 * (nba['HOME_PTS'] / nba['HOME_POSS'])
    nba['AWAY_OFF_RTG'] = 100 * (nba['AWAY_PTS'] / nba['AWAY_POSS'])
    
    #effective field goal percentage (eFG%)
    nba['HOME_EFG'] = (nba['HOME_FGM'] + 0.5 * nba['HOME_FG3M']) / nba['HOME_FGA']
    nba['AWAY_EFG'] = (nba['AWAY_FGM'] + 0.5 * nba['AWAY_FG3M']) / nba['AWAY_FGA']
    
    #target variables
    nba['TOTAL_PTS'] = nba['HOME_PTS'] + nba['AWAY_PTS']
    nba['POINT_DIFF'] = nba['HOME_PTS'] - nba['AWAY_PTS'] #positive = home win margin
    nba['HOME_WIN'] = (nba['POINT_DIFF'] > 0).astype(int)
    
    #cleaning up the date column name for next stage
    nba = nba.rename(columns={'HOME_GAME_DATE': 'GAME_DATE'})
    
    return nba

if __name__ == "__main__":
    print("Engineering features...")
    raw_matchups = process_nba_data()
    feature_df = calculate_advanced_metrics(raw_matchups)
    
    #keeping ONLY what is needed, including the crucial TEAM_IDs
    cols_to_keep = [
        'HOME_GAME_ID', 'GAME_DATE', 
        'HOME_TEAM_ID', 'AWAY_TEAM_ID', 
        'HOME_TEAM_NAME', 'AWAY_TEAM_NAME',
        'HOME_OFF_RTG', 'AWAY_OFF_RTG', 'HOME_EFG', 'AWAY_EFG', 'GAME_PACE',
        'TOTAL_PTS', 'POINT_DIFF', 'HOME_WIN'
    ]
    
    final_dataset = feature_df[cols_to_keep]
    final_dataset.to_csv('nba_training_set.csv', index=False)
    print("Feature Engineering Complete. Saved to nba_training_set.csv")