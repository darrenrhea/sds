import numpy as np
from rip_world_points import (
     rip_world_points
)


def ffsi_flatten_for_segmentation_inference_implementation2(
    original_rgb_hwc_np_u8: np.ndarray,  # this gets flattened
    camera_pose: np.ndarray,
    ad_origin,
    u,
    v,
    ad_height,
    ad_width,
    rip_height: int,
    rip_width: int,
):
    """
    Given the ad position and the camera_pose,
    rip the led board into a rip_height x rip_width pixels image,
    as well as returning the visibility_mask.
    
    TODO: this seems like a duplication of the function in,
    ffst_flatten_for_segmentation_training_implementation2.py,
    which also flattens a single given mask.
    Unify them.  Maybe we should be able to flatten j masks and k images at once.
    """

    # get the 3D xyz positions of the 4 corners of the LED board in world coordinates (in feet since it is NBA which is American)

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

    u_sample_coeffs = np.linspace(0.0, 1.0, rip_width)
    v_sample_coeffs = np.linspace(1.0, 0.0, rip_height)
    # make a tensor such that T[i, j, :] is the 3D world point
    # from which we rip the color of the pixel at (i, j)
    multiples_of_u = ad_width * np.outer(u_sample_coeffs, u)
    multiples_of_v = ad_height * np.outer(v_sample_coeffs, v)
    three_dee_points = ad_origin[np.newaxis, np.newaxis, :] + multiples_of_u[np.newaxis, :, :] + multiples_of_v[:, np.newaxis, :]
    xyzs = three_dee_points.reshape(-1, 3)

    rgba_values_f32 = rip_world_points(
        rgb_hwc_np_u8=original_rgb_hwc_np_u8,
        camera_pose=camera_pose,
        xyzs=xyzs,
    )
    flattened_rgb_with_visibility_mask = np.round(rgba_values_f32).clip(0, 255).astype(np.uint8).reshape(rip_height, rip_width, 4)
    
    
    visibility_mask = flattened_rgb_with_visibility_mask[:, :, 3]
    flattened_rgb = flattened_rgb_with_visibility_mask[:, :, :3]

    return flattened_rgb, visibility_mask


