from prii import (
     prii
)
import cv2
import numpy as np


def rip_quad_out_of_image(
    src_image,
    name_to_xy,
    dst_height,
    dst_width,
) -> np.ndarray:
    """
    Uses where the 4 corners of the ad rectangle are in 2D to
    go to homographically rip the ad to two dimensions.
    """
    assert isinstance(src_image, np.ndarray)
   

    dst = np.array(
        [
            [0, 0],
            [0, dst_height],
            [dst_width, dst_height],
            [dst_width, 0],
        ],
        dtype=np.float32
    )

    src = np.array(
        [
            name_to_xy["tl"],
            name_to_xy["bl"],
            name_to_xy["br"],
            name_to_xy["tr"],
        ],
        dtype=np.float32
    )

    H = cv2.getPerspectiveTransform(src, dst)
    new_image = np.zeros(
        shape=(
            dst_height,
            dst_width,
            3
        ),
        dtype=np.uint8
    )
    cv2.warpPerspective(
        src=src_image,
        dst=new_image,
        M=H,
        dsize=(dst_width, dst_height),
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )
    
    prii(new_image)

    return new_image

