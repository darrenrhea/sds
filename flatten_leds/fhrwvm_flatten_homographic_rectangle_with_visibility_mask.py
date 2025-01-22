import cv2
import numpy as np


def fhrwvm_flatten_homographic_rectangle_with_visibility_mask(
    image: np.ndarray,
    tl_bl_br_tr_np,
    out_height,  # Usually 256
    out_width, # Usually 1024
):
    """
    The caller is responsible for adding any margins to the homographic rectangle
    to get tl_bl_br_tr_np.
    """
    out_nchan = image.shape[2]
    assert out_nchan in [3, 4], f"ERROR: image.shape[2] must be 3 or 4, but it is {out_nchan}"

    dst_pad = np.array(
        [
            [0, 0],
            [0, out_height],
            [out_width, out_height],
            [out_width, 0],
        ],
        dtype=np.float32
    )

    H = cv2.getPerspectiveTransform(tl_bl_br_tr_np, dst_pad)
    
    new_image = np.zeros(shape=(out_height, out_width, out_nchan), dtype=np.uint8)
    cv2.warpPerspective(src=image, dst=new_image, M=H, dsize=(out_width, out_height))
    # compute visibility mask
    white_image = np.ones_like(image) * 255
    visibility_mask = cv2.warpPerspective(src=white_image, dst=None, M=H, dsize=(out_width, out_height))
    visibility_mask = cv2.cvtColor(visibility_mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, visibility_mask = cv2.threshold(visibility_mask, 1, 255, cv2.THRESH_BINARY)

    return new_image, visibility_mask

