from cutout_a_patch_from_fullsize import (
     cutout_a_patch_from_fullsize
)
import numpy as np



def cutout_a_nonempty_patch_from_fullsize(
    patch_width: int,
    patch_height: int,
    fullsize_image_np_u8: np.ndarray,
    onscreen_channel_index: int
):
    """
    If fullsize_image_np_u8 is a numpy array of shape [H, W, C] where C is usually 3 or 4 or 5 (weight_mask is the 5th / 4-ith channel),
    Then this function cuts out a patch of size patch_height x patch_width from fullsize_image_np_u8
    If fullsize_image_np_u8 is the patch_size, then this function just returns fullsize_image_np_u8.
    """
    assert (
        np.sum(
            fullsize_image_np_u8[:, :, onscreen_channel_index].astype(np.int64)
        ) > 0
    ), "There is nothing on screen, so we cannot cut out a nonempty patch."
    while True:
        patch = cutout_a_patch_from_fullsize(
            patch_width=patch_width,
            patch_height=patch_height,
            fullsize_image_np_u8=fullsize_image_np_u8
        )
        if np.any(patch[:, :, onscreen_channel_index] > 0):
            return patch
        
  

