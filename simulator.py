# simulator.py — Premier League 2026-27 season simulator
# Day 1: define the 20 teams and a rough strength rating for each.
# (Strengths are eyeballed placeholders — we'll fit them to real
#  results in Week 2.)

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

print(f"{len(teams)} teams loaded\n")

# Print them strongest first
for team, strength in sorted(teams.items(), key=lambda item: item[1], reverse=True):
    print(f"{team:<18} {strength}")