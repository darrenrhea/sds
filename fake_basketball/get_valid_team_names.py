def get_valid_team_names(
    sport: str,
    league: str
):
    assert sport == "basketball"
    assert league == "euroleague"
    valid_euroleague_team_names = [
        "bayern-munich",
        "asvel-villeurbanne",
        "allstar",
        "maccabi-tel-aviv",
        "fc-barcelona",
    ]
    return valid_euroleague_team_names
