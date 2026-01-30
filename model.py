import numpy as np
from scipy.stats import poisson

class MonteCarloEngine:
    def __init__(self, home_xg, away_xg):
        self.home_xg = home_xg
        self.away_xg = away_xg

    def run_simulation(self, num_simulations=10000):
        # SIMULATION LOOP
        # We simulate 10,000 matches instantly using Poisson distribution
        home_goals = np.random.poisson(self.home_xg, num_simulations)
        away_goals = np.random.poisson(self.away_xg, num_simulations)

        # CALCULATE RESULTS
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        # Goal Markets
        total_goals = home_goals + away_goals
        over_2_5 = np.sum(total_goals > 2.5)
        btts = np.sum((home_goals > 0) & (away_goals > 0)) # Both Teams To Score

        return {
            "Home Win": home_wins / num_simulations,
            "Draw": draws / num_simulations,
            "Away Win": away_wins / num_simulations,
            "Over 2.5": over_2_5 / num_simulations,
            "BTTS": btts / num_simulations
        }

def calculate_xg(home_scored, home_conceded, home_games, 
                 away_scored, away_conceded, away_games, 
                 league_avg_home, league_avg_away):
    
    # 1. Calculate Attack and Defense Strengths
    home_att_rating = (home_scored / home_games) / league_avg_home
    home_def_rating = (home_conceded / home_games) / league_avg_away
    
    away_att_rating = (away_scored / away_games) / league_avg_away
    away_def_rating = (away_conceded / away_games) / league_avg_home

    # 2. Calculate Expected Goals (xG) for this specific match
    home_expected_goals = home_att_rating * away_def_rating * league_avg_home
    away_expected_goals = away_att_rating * home_def_rating * league_avg_away

    return home_expected_goals, away_expected_goals