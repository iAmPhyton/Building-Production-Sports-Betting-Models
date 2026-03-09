import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamelog

def fetch_nba_data(seasons=['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']):
    """
    thi fetches regular season game logs for the specified seasons.
    Returns a single concatenated DataFrame.
    """
    all_games = []
    
    print(f"Fetching data for {len(seasons)} seasons...")
    
    for season in seasons:
        print(f"Processing season: {season}")
        try:
            #'P' indicates Player stats, 'T' indicates Team stats. We want Team ('T').
            #league_id='00' is the NBA.
            log = leaguegamelog.LeagueGameLog(
                season=season, 
                league_id='00', 
                player_or_team_abbreviation='T'
            )
            
            #API returns a list of dataframes; index 0 is the primary data
            nba = log.get_data_frames()[0]
            nba['SEASON_ID'] = season  #added a column to know which season this came from
            all_games.append(nba)
            
            #respecting API rate limits 
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error fetching {season}: {e}")

    if not all_games:
        return pd.DataFrame()

    #combine all seasons into one big table
    final_nba = pd.concat(all_games, ignore_index=True)
    return final_nba

if __name__ == "__main__":
    #getting data
    print("Starting download...")
    nba = fetch_nba_data()
    
    #quick checks
    print(f"\nSuccess! Downloaded {len(nba)} rows.")
    print("Columns acquired:", list(nba.columns))
    
    #saving to local
    filename = 'nba_games_last_5_years.csv'
    nba.to_csv(filename, index=False)
    print(f"Data saved to {filename}")