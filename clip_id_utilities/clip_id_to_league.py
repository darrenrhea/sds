def clip_id_to_league(
    clip_id: str
):
    """
    TODO: This is ghetto. Use the clip_id entities to determine the league
    of a clip_id.  Too busy right now to do it properly.
    """
    if clip_id.startswith("bay"):
        return "euroleague"
    else:
        return "nba"
