# simulator.py — Premier League 2026-27 season simulator
import numpy as np
import pandas as pd

teams = {
    "Man City": 2.1,
    "Arsenal": 2.1,
    "Liverpool": 2.0,
    "Chelsea": 1.9,
    "Newcastle": 1.8,
    "Man United": 1.7,
    "Tottenham": 1.7,
    "Aston Villa": 1.7,
    "Brighton": 1.6,
    "Nottingham Forest": 1.5,
    "Bournemouth": 1.5,
    "Crystal Palace": 1.5,
    "Brentford": 1.5,
    "Fulham": 1.4,
    "Everton": 1.3,
    "Leeds": 1.2,
    "Sunderland": 1.2,
    "Coventry": 1.1,
    "Ipswich": 1.1,
    "Hull": 1.0,
}

LEAGUE_AVG = 1.35
HOME_ADV = 1.25

def simulate_match(home, away):
    home_strength = teams[home]
    away_strength = teams[away]

    # Expected goals = your attack, scaled by how leaky the opponent is,
    # plus a home bump for the home side.
    home_exp = home_strength * (LEAGUE_AVG / away_strength) * HOME_ADV
    away_exp = away_strength * (LEAGUE_AVG / home_strength)

    # Let Poisson turn each expected value into an actual scoreline.
    home_goals = np.random.poisson(home_exp)
    away_goals = np.random.poisson(away_exp)
    return home_goals, away_goals

def generate_fixtures():
    return [(home, away) for home in teams for away in teams if home != away]

def simulate_season():
    # Every team starts on zeros
    table = {team: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
             for team in teams}

    for home, away in generate_fixtures():
        hg, ag = simulate_match(home, away)

        # Games played and goals
        table[home]["P"] += 1
        table[away]["P"] += 1
        table[home]["GF"] += hg
        table[home]["GA"] += ag
        table[away]["GF"] += ag
        table[away]["GA"] += hg

        # Points and W/D/L
        if hg > ag:
            table[home]["Pts"] += 3
            table[home]["W"] += 1
            table[away]["L"] += 1
        elif ag > hg:
            table[away]["Pts"] += 3
            table[away]["W"] += 1
            table[home]["L"] += 1
        else:
            table[home]["Pts"] += 1
            table[away]["Pts"] += 1
            table[home]["D"] += 1
            table[away]["D"] += 1

    return table

def show_table(table):
    df = pd.DataFrame(table).T                       # teams as rows
    df["GD"] = df["GF"] - df["GA"]                   # goal difference
    df = df.sort_values(["Pts", "GD", "GF"], ascending=False)
    df = df[["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]
    df.insert(0, "Pos", range(1, len(df) + 1))
    df.index.name = "Team"
    print(df.to_string())


season = simulate_season()
show_table(season)