from get_homography_for_flattening import (
     get_homography_for_flattening
)

import numpy as np


import cv2


def ffst_flatten_for_segmentation_training(
    original_rgb_hwc_np_u8: np.ndarray,  # this gets flattened
    mask_hw_np_u8: np.ndarray,  # this also gets flattened
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
    This factors through a homography, so it is not as accurate as the other implementation:
    ffst_flatten_for_segmentation_training_implementation2
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

    photograph_height_in_pixels = original_rgb_hwc_np_u8.shape[0]
    photograph_width_in_pixels = original_rgb_hwc_np_u8.shape[1]


    H = get_homography_for_flattening(
        photograph_height_in_pixels=photograph_height_in_pixels,
        photograph_width_in_pixels=photograph_width_in_pixels,
        camera_pose=camera_pose,
        ad_origin=ad_origin,
        u=u,
        v=v,
        ad_height=ad_height,
        ad_width=ad_width,
        rip_height=rip_height,
        rip_width=rip_width,
    )
    
    flattened_rgb_with_visibility_mask = np.zeros(
        shape=(
            rip_height,
            rip_width,
            4
        ),
        dtype=np.uint8
    )
    original_rgba_np_u8 = np.zeros(
        (original_rgb_hwc_np_u8.shape[0], original_rgb_hwc_np_u8.shape[1], 4),
        dtype=np.uint8
    )
    original_rgba_np_u8[:, :, :3] = original_rgb_hwc_np_u8
    original_rgba_np_u8[:, :, 3] = 255

    cv2.warpPerspective(
        src=original_rgba_np_u8,
        dst=flattened_rgb_with_visibility_mask,
        M=H,
        dsize=(rip_width, rip_height),
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0)
    )

    flattened_mask = np.zeros(
        shape=(
            rip_height,
            rip_width,
        ),
        dtype=np.uint8
    )

    cv2.warpPerspective(
        src=mask_hw_np_u8,
        dst=flattened_mask,
        M=H,
        dsize=(rip_width, rip_height),
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0,)
    )

    visibility_mask = flattened_rgb_with_visibility_mask[:, :, 3]
    flattened_rgb = flattened_rgb_with_visibility_mask[:, :, :3]

    return flattened_rgb, visibility_mask, flattened_mask
    


