from typing import Optional
import cv2
import numpy as np


def amthr_add_margins_to_homographic_rectangle(
    interior_homographic_rectangle,  # a list of 4 (x, y) points in U order, y grows down, pixel units
    out_height,  # Usually 256
    out_width, # Usually 1024 or whatever the assumed real aspect ratio suggests it should be given the out_height
    pad_frac_height,  # usually 0.15 for brewcub
    pad_frac_width: Optional[float] = None,  # usually None
) -> np.ndarray:
    """
    If you see a homographic rectangle, whose true aspect ratio you know,
    you can add additional margin to it to make it a bit bigger.
    returns the 4 corners of the margin-added homographic rectangle as a shape (4, 2) np.ndarray.
    """
    assert (
        interior_homographic_rectangle.shape == (4, 2)
    ), f"ERROR: interior_homographic_rectangle.shape is {interior_homographic_rectangle.shape} but it should be (4, 2)"

    assert (
        pad_frac_width is None or pad_frac_width >= 0.0
    ), f"ERROR: pad_frac_width must be None or >= 0, but it is {pad_frac_width}"
    
    # embed the image in the output so the margins are pad_frac of the embedded image dims
    embedded_height = out_height / (1 + 2 * pad_frac_height) # embedded_height + 2 * pad_frac * embedded_height = out_height
    pad_height = embedded_height * pad_frac_height
    if pad_frac_width is None: # option to use same padding as height
        pad_frac_width = pad_height / (out_width - 2 * pad_height)
    embedded_width = out_width / (1 + 2 * pad_frac_width) 
    pad_width = embedded_width * pad_frac_width
    dst = np.array(
        [
            [pad_width, pad_height],
            [pad_width, out_height - pad_height],
            [out_width - pad_width, out_height - pad_height],
            [out_width - pad_width, pad_height],
        ],
        dtype=np.float32
    )
    # map the corners of the *embedded* image back to the interior_homographic_rectangle points to get the interior_homographic_rectangle of the margins
    iH = cv2.getPerspectiveTransform(dst, interior_homographic_rectangle)

    # now map the margins forward
    dst_pad = np.array(
        [
            [0, 0],
            [0, out_height],
            [out_width, out_height],
            [out_width, 0],
        ],
        dtype=np.float32
    )
    src_pad = cv2.perspectiveTransform(
        dst_pad.reshape(-1, 1, 2),  # Wow. has to be shape 4 x 1 x 2 for reasons.
        iH
    )
    src_pad = src_pad.reshape(-1, 2)  # the caller prefers the less confusing shape
    
    assert src_pad.shape == (4, 2), f"ERROR: src_pad.shape is {src_pad.shape} but it should be (4, 2)"
    
    return src_pad

