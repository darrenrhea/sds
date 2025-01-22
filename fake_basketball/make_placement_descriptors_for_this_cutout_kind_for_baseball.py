from colorama import Fore, Style
from get_random_world_coordinates_for_a_cutout_of_this_kind import (
     get_random_world_coordinates_for_a_cutout_of_this_kind
)
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
import numpy as np

from nuke_lens_distortion import nuke_world_to_pixel_coordinates
from CameraParameters import CameraParameters


def make_placement_descriptors_for_this_cutout_kind_for_baseball(
    league: str,
    context_id: str,
    cutout_kind: str,
    camera_pose: CameraParameters,
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
        league in ["euroleague", "nba"]
    ), f"{league=} is not in ['euroleague', 'nba']"
    
    assert isinstance(num_cutouts_to_paste, int), f"{num_cutouts_to_paste=} must be an int"
    assert num_cutouts_to_paste >= 0, f"{num_cutouts_to_paste=} must be >= 0"

    valid_cutout_kinds = get_valid_cutout_kinds()
    assert cutout_kind in valid_cutout_kinds, f"{cutout_kind=} is not in {valid_cutout_kinds=}"

    placement_descriptors = []
    max_iterations = 10000
    # go until we have enough placement_descriptors or you hit max_iterations and barf.
    for cntr in range(max_iterations):
        if len(placement_descriptors) >= num_cutouts_to_paste:
            break
        # pick a random point where a referee would be:
        p_giwc = get_random_world_coordinates_for_a_cutout_of_this_kind(
            cutout_kind=cutout_kind,
            league=league,
            context_id=context_id,
        )

        
        distance_from_camera = np.linalg.norm(p_giwc - camera_pose.loc)
        # and a little higher, like 0.1 foot/meter above sea level:
        p_giwc_plus_one_foot = p_giwc + np.array([0, 0, 0.1])

        x_pixel, y_pixel, is_visible = nuke_world_to_pixel_coordinates(
            p_giwc=p_giwc,
            camera_parameters=camera_pose,
            photograph_width_in_pixels=photograph_width_in_pixels,
            photograph_height_in_pixels=photograph_height_in_pixels
        )
        
        head_x_pixel, head_y_pixel, slightly_higher_is_visible = nuke_world_to_pixel_coordinates(
            p_giwc=p_giwc_plus_one_foot,
            camera_parameters=camera_pose,
            photograph_width_in_pixels=photograph_width_in_pixels,
            photograph_height_in_pixels=photograph_height_in_pixels
        )
        how_many_pixels_is_one_tenth = y_pixel - head_y_pixel
        # one tenth of a meter is 0.328084 feet
        if league == "euroleague":
            how_many_pixels_is_six_feet_at_that_point = 6 / .328084 * how_many_pixels_is_one_tenth
        # one tenth of a foot is 0.1 feet
        elif league == "nba":
            how_many_pixels_is_six_feet_at_that_point = 6 / 0.1 * how_many_pixels_is_one_tenth # whatever unit
        else:
            raise Exception(f"ERROR: {league=} is not in ['euroleague', 'nba']")
        if is_visible and slightly_higher_is_visible:
            placement_descriptor = dict(
                bottom_xy=(
                    int(round(x_pixel)),
                    int(round(y_pixel)),
                ),
                how_many_pixels_is_six_feet_at_that_point=how_many_pixels_is_six_feet_at_that_point,
                distance_from_camera=distance_from_camera,
                cutout_kind=cutout_kind,
                index_within_kind=len(placement_descriptors)
            )
            placement_descriptors.append(placement_descriptor)

        

    if cntr == max_iterations - 1:
        print(f"{Fore.YELLOW}WARNING: hit {max_iterations=}. Maybe the camera pose is bad?{Style.RESET_ALL}")
    
    return placement_descriptors