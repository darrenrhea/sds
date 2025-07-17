from cutout_a_patch_from_fullsize import (
     cutout_a_patch_from_fullsize
)
import numpy as np



def cutout_patches_from_fullsize(
    patch_width: int,
    patch_height: int,
    num_patches_to_generate: int,
    fullsize_image_np_u8: np.ndarray,
):
    """
    Randomly cuts out patches.
    """
    num_channels = fullsize_image_np_u8.shape[2]
    assert isinstance(patch_width, int), "ERROR: patch_width must be an integer"
    assert isinstance(patch_height, int), "ERROR: patch_height must be an integer"
    assert isinstance(num_patches_to_generate, int), "ERROR: num_patches_to_generate must be an integer"
    assert patch_width > 0, "ERROR: patch_width must be positive"
    assert patch_height > 0, "ERROR: patch_height must be positive"
    assert num_patches_to_generate >= 0, "ERROR: num_patches_to_generate must be non-negative?"

    assert isinstance(fullsize_image_np_u8, np.ndarray), "ERROR: fullsize_image_np_u8 must be a numpy array"
    assert fullsize_image_np_u8.ndim == 3, "ERROR: fullsize_image_np_u8 must be a numpy array of shape [H, W, C] where C is usually 3"
    assert fullsize_image_np_u8.shape[2] in  [3, 4], "ERROR: fullsize_image_np_u8 must be a numpy array of shape [H, W, C] where C is usually 3 or 4"

    patches = np.zeros((num_patches_to_generate, patch_height, patch_width, num_channels), dtype=np.uint8)
    num_generated = 0

    while True:
        if num_generated >= num_patches_to_generate:
            break
        
        patch_hwc_np_u8 = cutout_a_patch_from_fullsize(
            patch_width=patch_width,
            patch_height=patch_height,
            fullsize_image_np_u8=fullsize_image_np_u8,
        )
     

        patches[num_generated, ...] = patch_hwc_np_u8        
        num_generated += 1
    
    assert isinstance(patches, np.ndarray)
    assert patches.ndim == 4
    assert patches.shape[0] == num_patches_to_generate
    assert patches.shape[1] == patch_height
    assert patches.shape[2] == patch_width
    assert patches.shape[3] == fullsize_image_np_u8.shape[2]
    return patches


