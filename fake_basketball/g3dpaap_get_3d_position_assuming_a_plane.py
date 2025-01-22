from CameraParameters import (
     CameraParameters
)
from nuke_get_world_coordinates_of_pixel_coordinates_assuming_plane import (
     nuke_get_world_coordinates_of_pixel_coordinates_assuming_plane
)
import numpy as np


def g3dpaap_get_3d_position_assuming_a_plane(
    xy_pixel_point: np.ndarray,
    original_rgb_hwc_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    implicit_plane: np.ndarray,
):
    assert implicit_plane.shape == (4,), f"{implicit_plane.shape=} but should be the coefficients of a plane"

    x_pixel, y_pixel = xy_pixel_point
    
    photograph_height_in_pixels = original_rgb_hwc_np_u8.shape[0]
    photograph_width_in_pixels = original_rgb_hwc_np_u8.shape[1]

    p_giwc = nuke_get_world_coordinates_of_pixel_coordinates_assuming_plane(
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        camera_pose=camera_pose,
        photograph_width_in_pixels=photograph_width_in_pixels,
        photograph_height_in_pixels=photograph_height_in_pixels,
        plane_coeffs=implicit_plane,
    )

    return p_giwc