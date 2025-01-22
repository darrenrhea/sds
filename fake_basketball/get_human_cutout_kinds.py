def get_human_cutout_kinds(
    league: str
) -> list[str]:
    if league in ["euroleague", "nba"]:
        human_cutout_kinds = ["player", "referee", "coach", "randos"]
    elif league in ["mlb"]:
        human_cutout_kinds = ["pitcher",]
    else:
        raise Exception(f"ERROR: {league=} is not in ['euroleague', 'nba', 'mlb']")

    return human_cutout_kinds
