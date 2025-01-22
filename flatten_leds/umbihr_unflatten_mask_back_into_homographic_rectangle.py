import cv2
import numpy as np


def umbihr_unflatten_mask_back_into_homographic_rectangle(
    flattened_mask: np.ndarray,  # Usually 1024 x 256
    board_as_4_corners_within_full_image,
    full_height,  # Usually 1920
    full_width, # Usually 1080
) -> np.ndarray:
    assert flattened_mask.shape[0] == 256
    assert flattened_mask.shape[1] == 1024
    assert flattened_mask.ndim == 2
    assert flattened_mask.dtype == np.uint8

    assert board_as_4_corners_within_full_image.shape == (4, 2)

    h, w = flattened_mask.shape
    src_pad = np.array(
        [
            [0, 0],
            [0, h],
            [w, h],
            [w, 0],
        ],
        dtype=np.float32
    )

    H = cv2.getPerspectiveTransform(src_pad, board_as_4_corners_within_full_image)
    
    new_mask = cv2.warpPerspective(
        src=flattened_mask,
        dst=None,
        M=H,
        dsize=(full_width, full_height),
        borderValue=255,
    )

    assert (
        new_mask.shape == (full_height, full_width)
    )
    
    assert (
        new_mask.dtype == np.uint8
    )

    return new_mask

