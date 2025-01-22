from get_valid_nba_teams import (
     get_valid_nba_teams
)
from pathlib import Path
from get_valid_euroleague_teams import (
     get_valid_euroleague_teams
)

import better_json as bj

from prompt_toolkit.shortcuts import radiolist_dialog

def nci_new_clip_id_cli_tool():
    """
    When you want to make a new clip_id, use this.

    https://www.euroleaguebasketball.net/euroleague/game-center/2023-24/fc-bayern-munich-zalgiris-kaunas/E2023/260/
    """

    league = radiolist_dialog(
        title="Favorite IDE",
        text="In what league did the game happen?",
        values=[
            ("euroleague", "euroleague"),
            ("nba", "nba"),
        ],
    ).run()

    print(f"league: {league}")

    if league == "euroleague":
        valid_euroleague_teams = get_valid_euroleague_teams()
    elif league == "nba":
        valid_euroleague_teams = get_valid_nba_teams()
    

    pairs_of_actual_value_display_name = [
        (
            team,
            team.replace("-", " ").title()
        ) for team in valid_euroleague_teams
    ]
    home_team = radiolist_dialog(
        title="Favorite IDE",
        text="What was the home team for this game?",
        values=pairs_of_actual_value_display_name,
    ).run()

    print(f"Home team: {home_team}")
    assert home_team in valid_euroleague_teams, f"Invalid team: {home_team}"


    away_team = radiolist_dialog(
        title="Favorite IDE",
        text="What was the away team for this game?",
        values=pairs_of_actual_value_display_name,
    ).run()

    print(f"Away team: {away_team}")
    assert away_team in valid_euroleague_teams, f"Invalid team: {away_team}"


    year_str = radiolist_dialog(
        title="Favorite IDE",
        text="In what year did this game happen?",
        values=[
            ("2024", "2024"),
            ("2023", "2023"),
            ("2022", "2022"),
        ],
    ).run()

    print(f"Year: {year_str}")
    assert year_str in ["2022", "2023", "2024"], f"Invalid year: {year_str}"

    month_str = radiolist_dialog(
        title="Favorite IDE",
        text="What month did this game happen?",
        values=[
            ("01", "January"),
            ("02", "February"),
            ("03", "March"),
            ("04", "April"),
            ("05", "May"),
            ("06", "June"),
            ("07", "July"),
            ("08", "August"),
            ("09", "September"),
            ("10", "October"),
            ("11", "November"),
            ("12", "December"),
        ],
    ).run()

    print(f"Month: {month_str}")
    assert month_str in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ], f"Invalid month: {month_str}"

    day_str = radiolist_dialog(
        title="Favorite IDE",
        text="What day did this game happen?",
        values=[
            ("01", "1"),
            ("02", "2"),
            ("03", "3"),
            ("04", "4"),
            ("05", "5"),
            ("06", "6"),
            ("07", "7"),
            ("08", "8"),
            ("09", "9"),
            ("10", "10"),
            ("11", "11"),
            ("12", "12"),
            ("13", "13"),
            ("14", "14"),
            ("15", "15"),
            ("16", "16"),
            ("17", "17"),
            ("18", "18"),
            ("19", "19"),
            ("20", "20"),
            ("21", "21"),
            ("22", "22"),
            ("23", "23"),
            ("24", "24"),
            ("25", "25"),
            ("26", "26"),
            ("27", "27"),
            ("28", "28"),
            ("29", "29"),
            ("30", "30"),
            ("31", "31"),
        ],
    ).run()

    print(f"Day: {day_str}")
    assert day_str in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
    ], f"Invalid day: {day_str}"


    quality_str = radiolist_dialog(
        title="Favorite IDE",
        text="what is the video source quality, including the deinterlacing method?",
        values=[
            ("sdi", "sdi"),
            ("mxf", "mxf"),
            ("mxf-yadif", "mxf-yadif"),
            ("mxf-ddv3", "mxf-ddv3"),
            ("srt", "srt"),
            ("yt", "yt"),
            ("hq", "hq"),

        ],
    ).run()

    print(f"quality: {quality_str}")

    clip_id = f"{home_team}-{away_team}-{year_str}-{month_str}-{day_str}-{quality_str}"

    print(f"clip_id: {clip_id}")

    new_dir = Path("~/r/clip_ids").expanduser() / "clips" / clip_id
    new_dir.mkdir(parents=True, exist_ok=True)
    file_path = new_dir / f"{clip_id}.json5"
    print(f"{file_path=}")
    iso8601 = f"{year_str}-{month_str}-{day_str}"
    
    
   

    jsonable = {
        "clip_id": clip_id,
         "use_cases": [
            "put a use case here",
            "put another use case here",
        ],
        "num_frames": None,
        "game_play": {
            "first_frame_index": None,
            "last_frame_index": None
        },
        "source_video": {
            "file_name": None,
            "file_extension": None,
            "s3": None,
            "sha256": None,
            "locations": [
            ],
        },
        "floor": {
            "city": None,
            "apron_rgb": None,
            "apron_color": None,
        },
        "round": None,
        "iso8601": iso8601,
        "home": home_team,
        "away": away_team,
        "league": "euroleague",
        "home_away_score": [None, None],
        "comments": [],
        "game": {
            "youtube_highlights": "https://www.youtube.com/watch?v=ByZs_2csUss",
            "official_page": "https://www.euroleaguebasketball.net/en/euroleague/game-center/2023-24/fenerbahce-beko-istanbul-fc-barcelona/E2023/269/",
            "league": "euroleague",
            "iso8601": iso8601,
            "round": None,
            "game_index": None,
            "teams": {
                "home": {
                    "abbreviation": "",
                    "name": home_team,
                    "score": None,
                    "jersey": {
                        "main": None,
                        "number": None,
                        "stripes": None,
                        "sleeves": None,
                        "short_stripes": "white",
                    },
                },
                "away": {
                    "abbreviation": "bar",
                    "team": "fc-barcelona",
                    "score": 74,
                    "jersey": {
                        "main": "yellow",
                        "number": "black",
                        "stripes": "red",
                        "sleeves": "black",
                        "neck": "black",
                    },
                },
            },
            "city": None,
            "arena": None,
        }
    }

    bj.dump(obj=jsonable, fp=file_path)
    print(f"bat {file_path}")



   