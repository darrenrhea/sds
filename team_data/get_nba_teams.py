nba_teams = {
    "atl": {
        "name": "Atlanta Hawks",
        "abbreviation": "atl",
    },
    "bkn": {
        "name": "Brooklyn Nets",
        "abbreviation": "bkn",
    },
    "bos": {
        "name": "Boston Celtics",
        "abbreviation": "bos",
    },
    "cha": {
        "name": "Charlotte Hornets",
        "abbreviation": "cha",
    },
    "chi": {
        "name": "Chicago Bulls",
        "abbreviation": "chi",
    },
    "cle": {
        "name": "Cleveland Cavaliers",
        "abbreviation": "cle",
    },
    "dal": {
        "name": "Dallas Mavericks",
        "abbreviation": "dal",
    },
    "den": {
        "name": "Denver Nuggets",
        "abbreviation": "den",
    },
    "det": {
        "name": "Detroit Pistons",
        "abbreviation": "det",
    },
    "gsw": {
        "name": "Golden State Warriors",
        "abbreviation": "gsw",
    },
    "hou": {
        "name": "Houston Rockets",
        "abbreviation": "hou",
    },
    "ind": {
        "name": "Indiana Pacers",
        "abbreviation": "ind",
    },
    "lac": {
        "name": "Los Angeles Clippers",
        "abbreviation": "lac",
    },
    "lal": {
        "name": "Los Angeles Lakers",
        "abbreviation": "lal",
    },
    "mem": {
        "name": "Memphis Grizzlies",
        "abbreviation": "mem",
    },
    "mia": {
        "name": "Miami Heat",
        "abbreviation": "mia",
    },
    "mil": {
        "name": "Milwaukee Bucks",
        "abbreviation": "mil",
    },
    "min": {
        "name": "Minnesota Timberwolves",
        "abbreviation": "min",
    },
    "nop": {
        "name": "New Orleans Pelicans",
        "abbreviation": "nop",
    },
    "nyk": {
        "name": "New York Knicks",
        "abbreviation": "nyk",
    },
    "okc": {
        "name": "Oklahoma City Thunder",
        "abbreviation": "okc",
    },
    "orl": {
        "name": "Orlando Magic",
        "abbreviation": "orl",
    },
    "phi": {
        "name": "Philadelphia 76ers",
        "abbreviation": "phi",
    },
    "phx": {
        "name": "Phoenix Suns",
        "abbreviation": "phx",
    },
    "por": {
        "name": "Portland Trail Blazers",
        "abbreviation": "por",
    },
    "sac": {
        "name": "Sacramento Kings",
        "abbreviation": "sac",
    },
    "sas": {
        "name": "San Antonio Spurs",
        "abbreviation": "sas",
    },
    "tor": {
        "name": "Toronto Raptors",
        "abbreviation": "tor",
    },
    "uta": {
        "name": "Utah Jazz",
        "abbreviation": "uta",
    },
    "was": {
        "name": "Washington Wizards",
        "abbreviation": "was",
    },
}

assert len(nba_teams) == 30

team_ids = sorted([k for k in nba_teams.keys()])
for team_id in team_ids:
    print(team_id)

editions = ["association", "icon", "statement", "city"]
long_edition_to_short = {
        "association": "AE",
        "icon": "IE",
        "statement": "SE",
        "city": "CE",
}


for smaller_year_of_the_season in range(2022, 2024):

    bigger_year_of_the_season = smaller_year_of_the_season + 1
    smaller_year_str = str(smaller_year_of_the_season)
    bigger_year_str = str(bigger_year_of_the_season)
    for team_id in team_ids:
        team = nba_teams[team_id]
        team["league"] = "nba"
        upper_case_team_abbrev = team["abbreviation"].upper()

        escaped_lockervision_team_for_url = team["name"].replace(" ", "%20")
        for jersey_edition in editions:
            short_jersey_edition = long_edition_to_short[jersey_edition]
            url = "".join(
                [
                    "https://appimages.nba.com/p/tr:n-slnfre/",
                    smaller_year_str,
                    "/uniform/",
                    escaped_lockervision_team_for_url,
                    "/",
                    upper_case_team_abbrev,
                    "_",
                    short_jersey_edition,
                    ".jpg",
                ]
            )
            print(f"wget {url}")

