def get_valid_cutout_kinds(
    sport: str="basketball",
):
    if sport == "basketball":
        valid_cutout_kinds = [
            "player",
            "referee",
            "coach",
            "ball",
            "led_screen_occluding_object",
        ]
    elif sport == "baseball":
        valid_cutout_kinds = [
            "pitcher",
            "batter",
            "baseball",
        ]
    else:
        raise Exception(f"ERROR: {sport=} is not in ['nba', 'euroleague', 'mlb']")
    return valid_cutout_kinds
