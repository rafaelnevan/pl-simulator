# simulator.py — Premier League 2026-27 season simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, factorial, log
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

NAME_FIXES = {"Nott'm Forest": "Nottingham Forest"}

def load_matches(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["FTR"])               # skip any rows without a result yet
    for col in ("HomeTeam", "AwayTeam"):
        df[col] = df[col].replace(NAME_FIXES)
    return df

matches      = load_matches("E0.csv")            # 2025/26: training data
matches_2627 = load_matches("E0_2627.csv")       # 2026/27 so far: test data

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

unknown = (set(matches_2627["HomeTeam"]) | set(matches_2627["AwayTeam"])) - set(ratings)
if unknown:
    print(f"WARNING: 26/27 names not recognised: {unknown}")

print(f"\nLeague avg goals/team/game: {league_avg:.2f}")
print("Top attacks (real data):")
for t in sorted(ratings, key=lambda x: ratings[x][0], reverse=True)[:5]:
    print(f"  {t:<18} attack {ratings[t][0]:.2f}  defense {ratings[t][1]:.2f}")

def expected_goals(home, away):
    home_att, home_def = ratings[home]
    away_att, away_def = ratings[away]
    hfa = HOME_ADV ** 0.5                        # split the home effect symmetrically
    home_exp = home_att * away_def * league_avg * hfa
    away_exp = away_att * home_def * league_avg / hfa
    return home_exp, away_exp

def simulate_match(home, away):
    home_exp, away_exp = expected_goals(home, away)
    return np.random.poisson(home_exp), np.random.poisson(away_exp)

MAX_GOALS = 10

def poisson_pmf(k, lam):
    return exp(-lam) * lam ** k / factorial(k)

def match_probabilities(home_exp, away_exp):
    """Expected goals -> P(home win), P(draw), P(away win)."""
    home_pmf = np.array([poisson_pmf(i, home_exp) for i in range(MAX_GOALS + 1)])
    away_pmf = np.array([poisson_pmf(j, away_exp) for j in range(MAX_GOALS + 1)])
    score = np.outer(home_pmf, away_pmf)         # score[i, j] = P(home i, away j)
    p_home = np.tril(score, -1).sum()            # below diagonal: home scored more
    p_draw = np.trace(score)                     # diagonal: level
    p_away = np.triu(score, 1).sum()             # above diagonal: away scored more
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total

def market_probabilities(row, cols):
    """Bookmaker odds -> implied probabilities, with the overround removed."""
    inv = np.array([1 / row[c] for c in cols])
    return tuple(inv / inv.sum())

def brier(pred, actual_idx):
    actual = [0, 0, 0]
    actual[actual_idx] = 1
    return sum((p - a) ** 2 for p, a in zip(pred, actual))

def logloss(pred, actual_idx):
    return -log(max(pred[actual_idx], 1e-15))    # clip so we never take log(0)

def backtest(test_matches, label):
    # Use the sharpest odds available in this file
    odds_sets = [("AvgH", "AvgD", "AvgA"),       # market average (used for both seasons)
                 ("B365H", "B365D", "B365A")]    # Bet365 as a fallback
    cols = next((c for c in odds_sets if set(c).issubset(test_matches.columns)), None)
    if cols is None:
        raise ValueError(f"No known odds columns in {label}")

    # Baseline: last season's H/D/A split, so it never sees the test results
    freq = matches["FTR"].value_counts(normalize=True)
    baseline_p = (freq["H"], freq["D"], freq["A"])

    idx_of = {"H": 0, "D": 1, "A": 2}
    names = ("Model", "Market", "Baseline")
    brier_total = {n: 0.0 for n in names}
    ll = {n: [] for n in names}                  # per-match log-loss, kept for the error bar

    for _, m in test_matches.iterrows():
        home, away = m["HomeTeam"], m["AwayTeam"]
        if home not in ratings or away not in ratings:
            continue
        if any(pd.isna(m[c]) for c in cols):     # skip matches with missing odds
            continue

        preds = {
            "Model":    match_probabilities(*expected_goals(home, away)),
            "Market":   market_probabilities(m, cols),
            "Baseline": baseline_p,
        }
        idx = idx_of[m["FTR"]]
        for n, p in preds.items():
            brier_total[n] += brier(p, idx)
            ll[n].append(logloss(p, idx))

    n_matches = len(ll["Model"])
    print(f"\n=== {label}: {n_matches} matches, odds from {cols[0][:-1]} ===")
    print(f"{'':<10}{'Brier':>9}{'Log-loss':>11}")
    for n in names:
        print(f"{n:<10}{brier_total[n] / n_matches:>9.4f}{np.mean(ll[n]):>11.4f}")

    # Is the model-vs-market gap real, or just noise?
    diff = np.array(ll["Model"]) - np.array(ll["Market"])
    margin = 1.96 * diff.std(ddof=1) / np.sqrt(n_matches)
    print(f"Model minus Market log-loss: {diff.mean():+.4f}  (±{margin:.4f}, 95%)")

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

#results = run_monte_carlo(10000)
#plot_results(results)

backtest(matches, "2025/26 (in-sample)")
backtest(matches_2627, "2026/27 so far (out-of-sample)")