from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from get_valid_team_names import (
     get_valid_team_names
)
from get_human_cutout_kinds import (
     get_human_cutout_kinds
)

from typing import Dict, Any
import pprint

def assert_cutout_metadata_is_good(
    cutout_metadata: Dict[str, Any]
) -> dict:
    """
    This checks that the metadata json file for a cutout is good,
    or crashes with an informative error message.
    If it does not crash, it returns the good cutout_metadata.
    """
    league = "nba"
    sport = "basketball"
    
    # valid_teams = get_valid_team_names(
    #     sport=sport,
    #     league=league,
    # )

    # valid_uniform_colors = [
    #      "white",
    #      "maroon",
    #      "lightblue",
    #      "darkblue",
    #      "yellow",
    #      "black",
    #      "blue",
    #      "red",
    # ]

  
    assert isinstance(cutout_metadata, dict), f"{cutout_metadata=} must be a dict"
    
    assert (
        "kind" in cutout_metadata
    ), f"The key kind must be in {cutout_metadata=}"

    assert (
        "name_to_xy" in cutout_metadata
    ), f"The key name_to_xy must be in {cutout_metadata=}"

    
    kind = cutout_metadata["kind"]
    if "league" not in cutout_metadata:
        pprint.pprint(cutout_metadata)
        print("is missing league")
    valid_cutout_kinds = get_valid_cutout_kinds(sport=sport)
    assert (
        kind in valid_cutout_kinds
    ), f"{cutout_metadata=} has {kind=} which is not in {valid_cutout_kinds=} for {league=}"

    name_to_xy = cutout_metadata["name_to_xy"]
    assert (
        isinstance(name_to_xy, dict)
    ), f"{cutout_metadata=} has {name_to_xy=} which is not a dict"

    for point_name, point in name_to_xy.items():
        assert len(point) == 2, f"{cutout_metadata=} has {point=} which is not a length 2 list"
        for i in range(2):
            assert (
                isinstance(point[i], int)
                or
                isinstance(point[i], float)
            ), f"{cutout_metadata=} has {point=} which is not made of numbers."

    human_cutout_kinds = get_human_cutout_kinds(
        league=league
    )
    

    if league in ["euroleague", "nba"]:
        if kind in human_cutout_kinds:
            # check humans name_to_xy:
            for key in ["bottom_of_lowest_foot", "six_feet_above_that"]:
                assert key in name_to_xy, f"{cutout_metadata=} must have {key=}"

            bottom_of_lowest_foot = name_to_xy["bottom_of_lowest_foot"]
            assert len(bottom_of_lowest_foot) == 2, f"{cutout_metadata=} has {bottom_of_lowest_foot=} which is not a length 2 list"

            six_feet_above_that = name_to_xy["six_feet_above_that"]
            
            assert len(six_feet_above_that) == 2, f"{cutout_metadata=} has {six_feet_above_that=} which is not a length 2 list"
    elif league == "mlb":
         if kind in ["pitcher"]:
            # check humans name_to_xy:
            for key in ["interfering_point", "six_feet_below_that"]:
                assert key in name_to_xy, f"{cutout_metadata=} must have {key=}"

            interfering_point = name_to_xy["interfering_point"]
            assert len(interfering_point) == 2, f"{cutout_metadata=} has {interfering_point=} which is not a length 2 list"

            six_feet_below_that = name_to_xy["six_feet_below_that"]
            
            assert len(six_feet_below_that) == 2, f"{cutout_metadata=} has {six_feet_below_that=} which is not a length 2 list"

    elif kind == "ball":
        for key in ["name_to_xy", "ball_kind", "league"]:
            assert (
                key in cutout_metadata
            ), f"{cutout_metadata=} must have {key=}"
        name_to_xy = cutout_metadata["name_to_xy"]
        for key in ["ball_bottom", "ball_center", "ball_top"]:
            assert (
                key in name_to_xy
            ), f"{cutout_metadata=} must have {key=}"

        ball_bottom = name_to_xy["ball_bottom"]
        ball_center = name_to_xy["ball_center"]
        ball_top = name_to_xy["ball_top"]

        for thing in [ball_bottom, ball_center, ball_top]:
             assert len(thing) == 2, f"{cutout_metadata=} has {thing=} which is not a length 2 list"

        assert ball_center[1] < ball_bottom[1], f"{cutout_metadata=} has {ball_center=} which is not above {ball_bottom=}"
        assert ball_center[1] > ball_top[1], f"{cutout_metadata=} has {ball_center=} which is not below {ball_top=}"

  
        
    return cutout_metadata