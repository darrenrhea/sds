from colorama import Fore, Style
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
import numpy as np


def make_placement_descriptors_for_this_cutout_kind_homographic_rectangle_based_version(
    league: str,
    context_id: str,
    cutout_kind: str,
    tl_bl_br_tr: list,
    photograph_width_in_pixels: int,
    photograph_height_in_pixels: int,
    num_cutouts_to_paste: int
) -> list:
    """
    Returns a list of placement_descriptors which may be as long as num_cutouts_to_paste, but might be shorter due to no placements been visible.

    Coaches tend to lurk in certain places near the LED board,
    whereas players tend to be on the court,
    and referees tend to be on the court but in the periphery.
    Balls are all over, but we might want to place them for maximum learning.
    led_screen_occluding_objects are in front of the LED screen on the floor.
    This determines the attachment points for cutouts of a given kind, and
    will help order them by distance from the camera.
    """
    assert (
        league in ["mlb"]
    ), f"{league=} is not in ['euroleague', 'nba']"
    
    assert isinstance(num_cutouts_to_paste, int), f"{num_cutouts_to_paste=} must be an int"
    assert num_cutouts_to_paste >= 0, f"{num_cutouts_to_paste=} must be >= 0"

    valid_cutout_kinds = get_valid_cutout_kinds(league=league)
    assert cutout_kind in valid_cutout_kinds, f"{cutout_kind=} is not in {valid_cutout_kinds=}"

    placement_descriptors = []
    max_iterations = 10000
    # go until we have enough placement_descriptors or you hit max_iterations and barf.
    for cntr in range(max_iterations):
        if len(placement_descriptors) >= num_cutouts_to_paste:
            break
        # pick a random point in the homographic rectangle given by tl_bl_br_tr:
        # for now just the midpoint up 50
        tl, bl, br, tr = tl_bl_br_tr
        if np.random.rand() < 0.5:
            x0, y0 = bl
            x1, y1 = br
            t = np.random.rand()
            x_pixel = t * x0 + (1-t) * x1
            y_pixel = t * y0 + (1-t) * y1
            y_pixel -= np.random.randint(1, 250)
        else:
            x0, y0 = br
            x1, y1 = tr
            t = np.random.rand()
            x_pixel = t * x0 + (1-t) * x1
            y_pixel = t * y0 + (1-t) * y1
            x_pixel -= np.random.randint(1, 250)


        # TODO: unbullshit this:
        how_many_pixels_is_six_feet_at_that_point = 500
       
        is_visible = (
            x_pixel >= 0
            and
            x_pixel < photograph_width_in_pixels
            and
            y_pixel >= 0
            and
            y_pixel < photograph_height_in_pixels   
        )

        if is_visible:
            placement_descriptor = dict(
                bottom_xy=(
                    int(round(x_pixel)),
                    int(round(y_pixel)),
                ),
                how_many_pixels_is_six_feet_at_that_point=how_many_pixels_is_six_feet_at_that_point,
                cutout_kind=cutout_kind,
                index_within_kind=len(placement_descriptors),
                distance_from_camera=np.random.rand(),  # in order to paste closer to the camera objects later
            )
            placement_descriptors.append(placement_descriptor)

        

    if cntr == max_iterations - 1:
        print(f"{Fore.YELLOW}WARNING: hit {max_iterations=}. Maybe the rectangle is off-screen? is bad?{Style.RESET_ALL}")
    
    return placement_descriptors