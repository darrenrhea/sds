import cv2
import numpy as np
from pathlib import Path

def homographically_warp_advertisement(
    image_np: np.ndarray, # an rgb hwc np u8 image of the ad
    tl_bl_br_tr: np.ndarray,  # a 4 x 2 matrix of the bottom left, bottom right, top right, top left corners of the ad in the image
    out_width: int,  # the size of canvas you want to draw on is out_width x out_height
    out_height: int
):
    """
    See the test.
    usually image_np is a flat ad image.
    tl_bl_br_tr is a list of 4 the 4 corners of homographic "rectangle"
    in the out_width x out_height image that we want to warp the image into
    """
    ad_height = image_np.shape[0]
    ad_width = image_np.shape[1]

    src = np.array(
        [
            [0, 0],
            [0, ad_height],
            [ad_width, ad_height],
            [ad_width, 0],
        ],
        dtype=np.float32
    )

    dst = np.array(
        tl_bl_br_tr,
        dtype=np.float32
    )
    assert dst.shape == (4, 2)

    H = cv2.getPerspectiveTransform(src, dst)
    new_image = np.zeros(shape=(out_height, out_width, 3), dtype=np.uint8)
    cv2.warpPerspective(src=image_np, dst=new_image, M=H, dsize=(out_width, out_height))
    return new_image


