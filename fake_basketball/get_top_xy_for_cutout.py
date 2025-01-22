from get_human_cutout_kinds import (
     get_human_cutout_kinds
)
from PasteableCutout import (
     PasteableCutout
)
from get_a_random_xy_where_mask_is_foreground import (
     get_a_random_xy_where_mask_is_foreground
)


def get_top_xy_for_cutout(
    cutout: PasteableCutout
):
    """
    Currently human cutouts the top_xy is the bottom of the lowest foot,
    whereas for balls it's a random point within the ball, although
    another choice would be the center of the ball.
    """     
    league = "nba"
    original_cutout_rgba_np_u8 = cutout.rgba_np_u8
    cutout_kind = cutout.kind
    # use the cutout metadata to get their toe point,
    # unless it's a ball, in which case we just pick a random point
    # in front of the screen:
    top_xy = None
    human_cutout_kinds = get_human_cutout_kinds(
        league=league
    )
    
    if cutout_kind in human_cutout_kinds and league in ["euroleague", "nba"]:
        assert hasattr(cutout, "foot_bottom")
        top_xy = (
            int(cutout.foot_bottom[0]),
            int(cutout.foot_bottom[1]),
        )
    elif cutout_kind in ["pitcher", "batter"] and league in ["mlb"]:
        assert hasattr(cutout, "interfering_point")
        top_xy = (
            int(cutout.interfering_point[0]),
            int(cutout.interfering_point[1]),
        )
    elif cutout_kind in ["ball", "baseball"]:
        # this is more likely to cause glancing blows to the LED board:
        top_xy = get_a_random_xy_where_mask_is_foreground(
            original_cutout_rgba_np_u8[:, :, 3]
        )
    elif cutout_kind in ["led_screen_occluding_object",]:
        # this is more likely to cause glancing blows to the LED board:
        top_xy = get_a_random_xy_where_mask_is_foreground(
            original_cutout_rgba_np_u8[:, :, 3]
        )
    else:
        print(f"What are we doing here? {cutout_kind=}, {league=}")
        assert False
    
    assert (
        top_xy is not None
    ), f"ERROR: {top_xy=} is None. Maybe {cutout_kind=} is not recognized?"
    return top_xy