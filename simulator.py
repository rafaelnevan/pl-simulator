# simulator.py — Premier League 2026-27 season simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

matches = pd.read_csv("E0.csv")

#fix spelling
matches["HomeTeam"] = matches["HomeTeam"].replace({"Nott'm Forest": "Nottingham Forest"})
matches["AwayTeam"] = matches["AwayTeam"].replace({"Nott'm Forest": "Nottingham Forest"})

league_avg = (matches["FTHG"].sum() + matches["FTAG"].sum()) / (2 * len(matches))
HOME_ADV = 1.21

scored, conceded, played = defaultdict(int), defaultdict(int), defaultdict(int)
for _, m in matches.iterrows():
    home, away = m["HomeTeam"], m["AwayTeam"]
    scored[home]   += m["FTHG"]
    conceded[home] += m["FTAG"]
    played[home]   += 1
    scored[away]   += m["FTAG"]
    conceded[away] += m["FTHG"]
    played[away]   += 1

ratings = {t: ((scored[t] / played[t]) / league_avg,
            (conceded[t] / played[t]) / league_avg) for t in scored}

promoted_by_strength = ["Coventry", "Ipswich", "Hull"]
relegated_by_strength = sorted(["West Ham", "Burnley", "Wolves"],
                               key=lambda t: (ratings[t][0] / ratings[t][1]), 
                               reverse=True)

for promoted, relegated in zip(promoted_by_strength, relegated_by_strength):
    ratings[promoted] = ratings[relegated]
    print(f"{promoted:<10} inherits {relegated:<10} -> "
          f"attack {ratings[relegated][0]:.2f}, defense {ratings[relegated][1]:.2f}")

print(f"\nLeague avg goals/team/game: {league_avg:.2f}")
print("Top attacks (real data):")
for t in sorted(ratings, key=lambda x: ratings[x][0], reverse=True)[:5]:
    print(f"  {t:<18} attack {ratings[t][0]:.2f}  defense {ratings[t][1]:.2f}")

def simulate_match(home, away):
    home_att, home_def = ratings[home]
    away_att, away_def = ratings[away]
    # expected goals = your attack × opponent's defense × league scoring level
    home_exp = home_att * away_def * league_avg * (HOME_ADV ** 0.5)
    away_exp = away_att * home_def * league_avg / (HOME_ADV ** 0.5)
    return np.random.poisson(home_exp), np.random.poisson(away_exp)

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
    return df

def plot_results(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    title = df.sort_values("Title %", ascending=False).head(8).iloc[::-1]
    ax1.barh(title["Team"], title["Title %"], color="blue")
    ax1.set_title("Title-winning probability")
    ax1.set_xlabel("Probability (%)")
    for i, v in enumerate(title["Title %"]):
        ax1.text(v + 0.3, i, f"{v:.1f}%", va="center")

    relegation = df.sort_values("Relegation %", ascending=False).head(8).iloc[::-1]
    ax2.barh(relegation["Team"], relegation["Relegation %"], color="red")
    ax2.set_title("Relegation probability")
    ax2.set_xlabel("Probability (%)")
    for i, v in enumerate(relegation["Relegation %"]):
        ax2.text(v + 0.3, i, f"{v:.1f}%", va="center")

    fig.suptitle("Premier League 2026-27: 10,000 Simulation Results", fontsize=16)
    fig.tight_layout()
    plt.savefig("results.png", dpi = 120)
    plt.show()

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

results = run_monte_carlo(10000)
plot_results(results)