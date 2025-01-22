import numpy as np


import cv2


def ffsi_flatten_for_segmentation_inference(
    original_rgb_hwc_np_u8: np.ndarray,  # this gets flattened
    rip_height: int,
    rip_width: int,
    H: np.ndarray,
):
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
    original_rgba_np_u8[:, :, 3] = 255  # fully opaque unless it is pulling info from off-stage

    cv2.warpPerspective(
        src=original_rgba_np_u8,
        dst=flattened_rgb_with_visibility_mask,
        M=H,
        dsize=(rip_width, rip_height),
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0)
    )

    visibility_mask = flattened_rgb_with_visibility_mask[:, :, 3]
    flattened_rgb = flattened_rgb_with_visibility_mask[:, :, :3]

    return flattened_rgb, visibility_mask


