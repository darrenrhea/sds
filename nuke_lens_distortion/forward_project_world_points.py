from get_distorted_radius import (
     get_distorted_radius
)
import numpy as np
import rodrigues_utils
from CameraParameters import CameraParameters


def forward_project_world_points(
    photograph_width_in_pixels: int,
    photograph_height_in_pixels: int,
    camera_pose: CameraParameters,
    xyzs: np.ndarray, # a list of 3D world locations you want to convert to pixel locations
) -> np.ndarray:
    """
    Given a camera pose and a bunch of points in world coordinates,
    project them into pixel coordinates.
    Because of the Nuke direction, this is a bit hard.
    Any world point that is insanely, like 20% or more, offscreen
    is considered to be invisible, so we return -1, -1 for those.
    """
   
    assert isinstance(photograph_width_in_pixels, int)
    assert isinstance(photograph_height_in_pixels, int)
    assert isinstance(camera_pose, CameraParameters)
    
    assert isinstance(xyzs, np.ndarray)
    assert xyzs.ndim == 2
    assert xyzs.shape[1] == 3, "xyzs should have three columns"
    assert (
        xyzs.dtype == np.float32
        or
        xyzs.dtype == np.float64
    ), "xyzs should be float32 or float64"

        
    asp = photograph_height_in_pixels / photograph_width_in_pixels

    cameras_location_in_world_coordinates = camera_pose.loc
    assert cameras_location_in_world_coordinates.shape == (3,)
    xyzs_relative_to_camera = xyzs - cameras_location_in_world_coordinates
    rodrigues_vector = camera_pose.rod
    focal_length = camera_pose.f
    k1 = camera_pose.k1
    k2 = camera_pose.k2
   
    world_to_camera = rodrigues_utils.SO3_from_rodrigues(
        rodrigues_vector
    )
    
    cc_xyzs = np.dot(
        xyzs_relative_to_camera,
        world_to_camera.T
    ) 
    assert cc_xyzs.shape == xyzs.shape

    undistorted_us = cc_xyzs[:, 0] / cc_xyzs[:, 2]
    undistorted_vs = cc_xyzs[:, 1] / cc_xyzs[:, 2]

    xd_approx = undistorted_us * focal_length
    yd_approx = undistorted_vs * focal_length

    no_way_it_is_visible = ~ (
        (xd_approx > -1.2)
        &
        (xd_approx < 1.2)
        &
        (yd_approx > -1.2)
        &
        (yd_approx < 1.2)
    )

    # These yews and vees need to be distorted:
    r_undistorted2 = undistorted_us**2 + undistorted_vs**2
    r_undistorted = np.sqrt(r_undistorted2)

    r_distorted = get_distorted_radius(
        undistorted_radius_squared=r_undistorted2,
        k1=k1,
        k2=k2,
    )

    distorted_us = r_distorted * undistorted_us / r_undistorted
    distorted_vs = r_distorted * undistorted_vs / r_undistorted

    # now we have the normalized coordinates:
    xd = distorted_us * focal_length
    yd = distorted_vs * focal_length
    # Now in pixel units:
    ijs = np.zeros(
        shape=(xyzs.shape[0], 2),
        dtype=np.float32
    )

    ijs[:, 1] = (xd + 1) / 2.0 * (photograph_width_in_pixels - 1.0)
    ijs[:, 0] = (yd + asp) / 2.0 * (photograph_width_in_pixels - 1.0)
    ijs[no_way_it_is_visible, 0] = -1
    ijs[no_way_it_is_visible, 1] = -1

    i_s = ijs[:, 0]
    j_s = ijs[:, 1]
    
    i_s = np.maximum(i_s, -1)
    j_s = np.maximum(j_s, -1)
    
    i_s = np.minimum(i_s, photograph_height_in_pixels + 1)
    j_s = np.minimum(j_s, photograph_width_in_pixels + 1)
    
    ijs[:, 0] = i_s
    ijs[:, 1] = j_s
    
    return ijs
   
