def get_valid_floor_ids():
    """
    The canonical source of truth for what floor_ids there are.
    TODO: pull from a database.
    """

    valid_floor_ids = [
        "22-23_ATL_CORE",
        "22-23_CHI_CORE",
        "22-23_WAS_CORE",
        "24-25_ALL_STAR",  # the all start court
        "24-25_HOU_CITY",
        "24-25_HOU_CORE",
        "24-25_HOU_STMT",
        "24-25_BAL_DAKAR",
        "24-25_IND_CORE",
    ]
    
    return valid_floor_ids