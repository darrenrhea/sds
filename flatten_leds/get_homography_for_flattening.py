import time
from typing import Optional


from find_2d_convex_hull import (
     find_2d_convex_hull
)
from draw_points_on_rasterized_image import (
     draw_points_on_rasterized_image
)
from project_3d_points_to_2d_pixel_coordinates import (
     project_3d_points_to_2d_pixel_coordinates
)
from prii import (
     prii
)
from draw_3d_points import (
     draw_3d_points
)

import numpy as np


import cv2

from get_indicator_of_largest_area_quadrilateral_makeable_from_these_points import (
     get_indicator_of_largest_area_quadrilateral_makeable_from_these_points
)



def get_homography_for_flattening(
    photograph_height_in_pixels: int,
    photograph_width_in_pixels: int,
    camera_pose: np.ndarray,
    ad_origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ad_height: float,
    ad_width: float,
    rip_height: int,
    rip_width: int,
    original_rgb_hwc_np_u8: Optional[np.ndarray] = None,  # If you want to debug this, it can be good to have the original image.
) -> np.ndarray:
    """
    Given the physical world coordinates of the LED screen,
    the camera pose,
    and the dimensions of the photograph and the rip,
    return the homography that will flatten the LED region to the rip.

    Pseudo-code:

    Get a camera pose for the image.

    Take a grid of points across the world coordinates of the LED screen, a bit finer towards the ends.

    For each grid point, see if it is visible, and if so, where is the observation, say in pixel units.

    Pick 4 visible observations, trying to get a maximal area quad.

    Find a homography between the pixel units of the 4 observations and the pixel units of the standard size rip image.

    Use the homography to flatten the image.
    """
    assert original_rgb_hwc_np_u8 is None or isinstance(original_rgb_hwc_np_u8, np.ndarray)
    debug = original_rgb_hwc_np_u8 is not None

    assert isinstance(ad_origin, np.ndarray)
    assert ad_origin.shape == (3,)
    assert ad_origin.dtype == np.float64 or ad_origin.dtype == np.float32

    assert isinstance(u, np.ndarray)
    assert u.shape == (3,)
    assert u.dtype == np.float64 or u.dtype == np.float32
    assert np.isclose(np.linalg.norm(u), 1)

    assert isinstance(v, np.ndarray)
    assert v.shape == (3,)
    assert v.dtype == np.float64 or v.dtype == np.float32
    assert np.isclose(np.linalg.norm(v), 1)

    assert isinstance(ad_height, float)
    assert ad_width > 0
    assert isinstance(ad_width, float)
    assert ad_height > 0


    num_grid_points_high = 10

    width_coefficients = np.array(
        list(np.linspace(0, 0.1, 10))
        +
        list(np.linspace(0.0, 1, 10))
        +
        list(np.linspace(0.9, 1, 10))
    )

    num_grid_points_wide = width_coefficients.shape[0]
    
    height_coefficients = np.linspace(0, 1, num_grid_points_high)

    grid_points = np.zeros(
        shape=(
            num_grid_points_wide * num_grid_points_high,
            3
        ),
        dtype=np.float64
    )

    grid_points_in_2d_rip_space = np.zeros(
        shape=(
            num_grid_points_wide * num_grid_points_high,
            2,
        ),
        dtype=np.float64
    )

    for j, height_coefficient in enumerate(height_coefficients):
        for i, width_coefficient in enumerate(width_coefficients):
            grid_points[j * num_grid_points_wide + i] = ad_origin + width_coefficient * u * ad_width + height_coefficient * v * ad_height
            grid_points_in_2d_rip_space[j * num_grid_points_wide + i, :] = np.array([width_coefficient * (rip_width - 1), (1.0 - height_coefficient) * (rip_height - 1)])

    
    # print(f"grid_points: {grid_points}")

    if debug:
        drawing_of_the_grid_points = draw_3d_points(
            original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            camera_pose=camera_pose,
            xyzs=grid_points,
        )

        prii(drawing_of_the_grid_points)

    xyz_xy_is_visible_rows = project_3d_points_to_2d_pixel_coordinates(
        photograph_width_in_pixels=photograph_width_in_pixels,
        photograph_height_in_pixels=photograph_height_in_pixels,
        camera_pose=camera_pose,
        xyzs=grid_points,
    )

    indicator = xyz_xy_is_visible_rows[:, 5] > 0.5 

    visible_pixel_locations_in_photo = xyz_xy_is_visible_rows[indicator, 3:5]
    visible_three_dee_locations = grid_points[indicator, :]
    visible_points_in_2d_rip_space = grid_points_in_2d_rip_space[indicator, :]

    # print(f"visible_pixel_locations_in_photo: {visible_pixel_locations_in_photo}")

    if debug:
        drawing2 = draw_points_on_rasterized_image(
            original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            xys=visible_pixel_locations_in_photo,
        )

        prii(drawing2)

    _, indicator = find_2d_convex_hull(visible_pixel_locations_in_photo)
    convex_hull_points = visible_pixel_locations_in_photo[indicator]
    corresponding_xyzs = visible_three_dee_locations[indicator]
    corresponding_in_2d_rip_space = visible_points_in_2d_rip_space[indicator]
        # print(convex_hull_points)

    if debug:
        drawing_of_convex_hull = draw_points_on_rasterized_image(
            original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            xys=convex_hull_points,
        )

        prii(drawing_of_convex_hull)

    large_area_quad_indicator = get_indicator_of_largest_area_quadrilateral_makeable_from_these_points(convex_hull_points)


    large_area_quad_points = convex_hull_points[large_area_quad_indicator]
    corresponding_xyzs_for_the_quadrilateral = corresponding_xyzs[large_area_quad_indicator]
    corresponding_in_2d_rip_space_for_the_quadrilateral  = corresponding_in_2d_rip_space[large_area_quad_indicator]
    
    if debug:
        drawing_of_quadrilateral = draw_points_on_rasterized_image(
            original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            xys=large_area_quad_points
        )
        prii(drawing_of_quadrilateral)

    if debug:
        drawing96 = draw_3d_points(
            original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            camera_pose=camera_pose,
            xyzs=corresponding_xyzs_for_the_quadrilateral
        )

        prii(drawing96)

    if debug:
        rip_rgb_np_u8 = np.zeros(
            shape=(rip_height, rip_width, 3),
            dtype=np.uint8
        )
    
        drawing_in_rip_space = draw_points_on_rasterized_image(
            original_rgb_hwc_np_u8=rip_rgb_np_u8,
            xys=corresponding_in_2d_rip_space_for_the_quadrilateral
        )

        prii(drawing_in_rip_space)

    large_area_quad_points_f32 = large_area_quad_points.astype(np.float32)
    corresponding_in_2d_rip_space_for_the_quadrilateral_f32 = corresponding_in_2d_rip_space_for_the_quadrilateral.astype(np.float32)
    H = cv2.getPerspectiveTransform(
        src=large_area_quad_points_f32,
        dst=corresponding_in_2d_rip_space_for_the_quadrilateral_f32
    )
    return H

