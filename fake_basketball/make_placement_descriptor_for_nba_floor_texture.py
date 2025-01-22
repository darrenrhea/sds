from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from typing import Any, Dict


def make_placement_descriptor_for_nba_floor_texture(
    points: Dict[str, Any],
    texture_width_in_pixels: int,
    texture_height_in_pixels: int,
) -> AdPlacementDescriptor:
    """
    This came up when stuffing the floor texture under segmentation annotations.
    """
    assert (
        isinstance(points, dict)
    ), f"ERROR: points should be a dict, but it is {points=}"
    assert (
        isinstance(texture_width_in_pixels, int)
    ), f"ERROR: texture_width_in_pixels should be an int, but it is {texture_width_in_pixels=}"
    assert (
        isinstance(texture_height_in_pixels, int)
    ), f"ERROR: texture_height_in_pixels should be an int, but it is {texture_height_in_pixels=}"

    # truths about the nba rules / court:
    court_width = 47.0 * 2.0
    court_height = 25.0 * 2.0
    top_left_xy = points["top_left_interior_court_corner_xy"]
    bottom_right_xy = points["bottom_right_interior_court_corner_xy"]
    left_margin_in_pixels = top_left_xy[0]
    top_margin_in_pixels = top_left_xy[1]
    right_margin_in_pixels = texture_width_in_pixels - bottom_right_xy[0] - 1
    bottom_margin_in_pixels = texture_height_in_pixels - bottom_right_xy[1] - 1

    # truths about the texture:
    how_tall_is_the_interior_of_the_legal_area_of_the_court_in_pixels = bottom_right_xy[1] - top_left_xy[1]
    how_wide_is_the_interior_of_the_legal_area_of_the_court_in_pixels = bottom_right_xy[0] - top_left_xy[0]
   
    left_x_margin = left_margin_in_pixels / how_wide_is_the_interior_of_the_legal_area_of_the_court_in_pixels * court_width
    right_x_margin = right_margin_in_pixels / how_wide_is_the_interior_of_the_legal_area_of_the_court_in_pixels * court_width
    top_y_margin = top_margin_in_pixels / how_tall_is_the_interior_of_the_legal_area_of_the_court_in_pixels * court_height
    bottom_y_margin = bottom_margin_in_pixels / how_tall_is_the_interior_of_the_legal_area_of_the_court_in_pixels * court_height

    # print(f"{left_x_margin=}")
    # print(f"{right_x_margin=}")
    # print(f"{top_y_margin=}")
    # print(f"{bottom_y_margin=}")

    x_min = - court_width / 2 - left_x_margin
    x_max =   court_width / 2 + right_x_margin

    y_min = - court_height / 2 - bottom_y_margin
    y_max =   court_height / 2 + top_y_margin
    
    descriptor = AdPlacementDescriptor(
        name="floor_texture",
        origin=[x_min, y_min, 0.0],
        u=[1.0, 0.0, 0.0],
        v=[0.0, 1.0, 0.0],
        height=y_max - y_min,
        width=x_max - x_min,
    )

    return descriptor
            
