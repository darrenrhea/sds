import numpy as np


import cv2


def utsi_unflatten_the_segmentation_inferences(
    flat_mask_hw_np_u8: np.ndarray,  # this gets unflattened
    rip_height: int,
    rip_width: int,
    H: np.ndarray,  # we invert it for you.  Same as it was to flatten
    photograph_height_in_pixels: int,
    photograph_width_in_pixels: int,
):
    """
    Suppose for a given video frame,
    For each LED board, you have flattened the RGB then inferred in flattened space.
    This puts those inferences back into the original 3D perspective.
    
    Having inferred on flattened LED boards,
    you need to unflatten it back into a perspectivized segmentation.
    
    TODO: multiple LED boards unioned.
    """
    assert flat_mask_hw_np_u8.shape == (rip_height, rip_width)

    mask_hw_np_u8 = np.zeros(
        (
            photograph_height_in_pixels,
            photograph_width_in_pixels,
        ),
        dtype=np.uint8
    )
  
    cv2.warpPerspective(
        src=flat_mask_hw_np_u8,
        dst=mask_hw_np_u8,
        M=np.linalg.inv(H),
        dsize=(photograph_width_in_pixels, photograph_height_in_pixels),
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255,)  # That which is not mentioned is assumed foreground
    )

    

    return mask_hw_np_u8

