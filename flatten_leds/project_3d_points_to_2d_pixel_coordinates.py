from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import numpy as np
from CameraParameters import CameraParameters


def project_3d_points_to_2d_pixel_coordinates(
    photograph_width_in_pixels,
    photograph_height_in_pixels,
    camera_pose: CameraParameters,
    xyzs: np.ndarray,
) -> np.ndarray:
    """
    You have a bunch of 3D points in the world.
    write out a matrix that has their 2D pixel coordinates and whether they are visible.
    """
    assert isinstance(xyzs, np.ndarray)
    assert xyzs.ndim == 2
    assert xyzs.shape[1] == 3

    assert isinstance(camera_pose, CameraParameters)
    

    xyz_xy_is_visible = np.zeros(
        shape=(xyzs.shape[0], 6),
        dtype=np.double
    )

    for index, p_giwc in enumerate(xyzs):
       
        x_pixel, y_pixel, is_visible = nuke_world_to_pixel_coordinates(
            p_giwc=np.array(p_giwc),
            camera_parameters=camera_pose,
            photograph_width_in_pixels=photograph_width_in_pixels,
            photograph_height_in_pixels=photograph_height_in_pixels
        )
        
        really_is_visible = (
            is_visible
            and
            x_pixel >= 0.0 and x_pixel <= photograph_width_in_pixels
            and
            y_pixel >= 0.0 and y_pixel <= photograph_height_in_pixels
        )

        xyz_xy_is_visible[index] = np.array(
            [p_giwc[0], p_giwc[1], p_giwc[2], x_pixel, y_pixel, really_is_visible],
            dtype=np.double
        )
        
        
    return xyz_xy_is_visible
                    
