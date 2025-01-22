from fhrwvm_flatten_homographic_rectangle_with_visibility_mask import (
     fhrwvm_flatten_homographic_rectangle_with_visibility_mask
)
from amthr_add_margins_to_homographic_rectangle import (
     amthr_add_margins_to_homographic_rectangle
)
import cv2
import numpy as np


def amtfhrwvm_add_margin_then_flatten_homographic_rectangle_with_visibility_mask(
    image: np.ndarray,
    interior_homographic_rectangle,  # a np.float32 shape=(4, 2) of 4 (x, y) points in U order, y grows down, pixel units
    out_height,  # Usually 256
    out_width, # Usually 1024
    pad_frac_height=0,  # usually 0.15 for brewcub
    pad_frac_width=None, # usually None so as to pad the left and right margins the same as the top and bottom margins
):
    """
    usually the caller would build tl_bl_br_tr via:
    
    tl_bl_br_tr = [ann[screenName][corner] for corner in ["tl", "bl", "br", "tr"]]

    usually image is a flat ad image.
    tl_bl_br_tr is a list of 4 the 4 corners of homographic "rectangle"
    in the out_width x out_height image that we want to warp the image into.

    A common value for out_height is 256.

    screenLayer, visibilityMask = flatten_screen_w_margin( # rgba layer to be overlayed on top of frame
        image = image_hwc_np,
        tl_bl_br_tr = srcPoints, #np.vstack( [corners[cnr] for cnr in ["tl", "bl", "br", "tr"]] ),
        out_height = ripHeight,
        out_width = ripWidth,
        out_nchan = 4,
        pad_frac_width = -1, # use the same pad as pad height
        pad_frac_height = 0.15
    )
    """
    out_nchan = image.shape[2]
    assert out_nchan in [3, 4], f"ERROR: image.shape[2] must be 3 or 4, but it is {out_nchan}"

    src_pad = amthr_add_margins_to_homographic_rectangle(
        interior_homographic_rectangle=interior_homographic_rectangle,
        out_height=out_height,
        pad_frac_width=pad_frac_width,
        pad_frac_height=pad_frac_height,
        out_width=out_width,
    )

    new_image, visibility_mask = fhrwvm_flatten_homographic_rectangle_with_visibility_mask(
        image=image,
        src_pad=src_pad,
        out_height=out_height,
        out_width=out_width,
    )

    return new_image, visibility_mask, src_pad

