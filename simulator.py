# simulator.py — Premier League 2026-27 season simulator
import numpy as np
import pandas as pd
from collections import defaultdict

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

LEAGUE_AVG = 1.20
HOME_ADV = 1.21

def simulate_match(home, away):
    home_strength = teams[home]
    away_strength = teams[away]

    # Expected goals = your attack, scaled by how leaky the opponent is, plus a home bump for the home side.
    home_exp = home_strength * (LEAGUE_AVG / away_strength) * HOME_ADV
    away_exp = away_strength * (LEAGUE_AVG / home_strength)

    #Poisson turn each expected value into an actual scoreline.
    home_goals = np.random.poisson(home_exp)
    away_goals = np.random.poisson(away_exp)
    return home_goals, away_goals

def generate_fixtures():
    return [(home, away) for home in teams for away in teams if home != away]

def simulate_season():
    # Every team starts on zeros
    table = {team: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
             for team in teams}

    #record every scoreline for diagnostics purposes
    result = [] 

    for home, away in generate_fixtures():
        hg, ag = simulate_match(home, away)
        result.append((home, away, hg, ag))

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

    return table, result

def show_table(table):
    df = pd.DataFrame(table).T
    #goal difference = GF - GA
    df["GD"] = df["GF"] - df["GA"]
    df = df.sort_values(["Pts", "GD", "GF"], ascending=False)
    df = df[["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]
    df.insert(0, "Pos", range(1, len(df) + 1))
    df.index.name = "Team"
    print(df.to_string())


def finishing_order(table):
    return sorted(
        teams,
        key = lambda t: (table[t]["Pts"], table[t]["GF"] - table[t]["GA"], table[t]["GF"]),
        reverse = True,
    )
def run_monte_carlo(n_seasons=10000):
    titles = defaultdict(int)
    top4 = defaultdict(int)
    relegated = defaultdict(int)
    total_points = defaultdict(int)

    for i in range(n_seasons):
        table, _ = simulate_season()
        order = finishing_order(table)

        #1st place
        titles[order[0]] += 1

        #top 4
        for t in order[:4]:
            top4[t] += 1

        #bottom three
        for t in order[-3:]:
            relegated[t] += 1

        #total points for average points
        for t in teams:
            total_points[t] += table[t]["Pts"]

        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1} seasons simulated")

    rows = [{
        "Team": t,
        "Title %": 100 * titles[t] / n_seasons,
        "Top 4 %": 100 * top4[t] / n_seasons,
        "Relegation %": 100 * relegated[t] / n_seasons,
        "Avg Pts": total_points[t] / n_seasons,
    } for t in teams]

    df = pd.DataFrame(rows).sort_values("Title %", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    print("\n" + df.round(1).to_string())

def diagnostics(n_seasons=10000):
    home_wins = draws = away_wins = total_goals = total_matches = 0
    champ_points = last_points = 0

    for _ in range(n_seasons):
        table, results = simulate_season()
        order = finishing_order(table)

        champ_points += table[order[0]]["Pts"]
        last_points += table[order[-1]]["Pts"]

        for home, away, hg, ag in results:
            total_matches += 1
            total_goals += hg + ag
            if hg > ag:
                home_wins += 1
            elif ag > hg:
                away_wins += 1
            else:
                draws += 1

    print(f"Seasons simulated: {n_seasons}\n")
    print(f"Goals per game:    {total_goals / total_matches:.2f}    (real PL ≈ 2.8)")
    print(f"Home wins:         {100 * home_wins / total_matches:.1f}%    (real ≈ 44%)")
    print(f"Draws:             {100 * draws / total_matches:.1f}%    (real ≈ 25%)")
    print(f"Away wins:         {100 * away_wins / total_matches:.1f}%    (real ≈ 31%)")
    print(f"Champion points:   {champ_points / n_seasons:.1f}    (real ≈ 88)")
    print(f"Last-place points: {last_points / n_seasons:.1f}    (real ≈ 22)")

diagnostics()