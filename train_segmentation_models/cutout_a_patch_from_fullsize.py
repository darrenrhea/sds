import numpy as np



def cutout_a_patch_from_fullsize(
    patch_width: int,
    patch_height: int,
    fullsize_image_np_u8: np.ndarray,
):
    """
    If fullsize_image_np_u8 is a numpy array of shape [H, W, C] where C is usually 3 or 4 or 5 (weight_mask is the 5th / 4-ith channel),
    Then this function cuts out a patch of size patch_height x patch_width from fullsize_image_np_u8
    If fullsize_image_np_u8 is the patch_size, then this function just returns fullsize_image_np_u8.
    """
    if fullsize_image_np_u8.shape[0] == patch_height and fullsize_image_np_u8.shape[1] == patch_width:
        warped_np_u8 = fullsize_image_np_u8
        assert warped_np_u8.shape == (patch_height, patch_width, fullsize_image_np_u8.shape[2])
        assert warped_np_u8.dtype == np.uint8
        return warped_np_u8


    
    x0 = np.random.randint(low=0, high=fullsize_image_np_u8.shape[1] - patch_width + 1)
    y0 = np.random.randint(low=0, high=fullsize_image_np_u8.shape[0] - patch_height + 1)
    x1 = x0 + patch_width
    y1 = y0 + patch_height

    patch_hwc_np_u8 = fullsize_image_np_u8[y0:y1, x0:x1, :]
    assert patch_hwc_np_u8.shape == (patch_height, patch_width, fullsize_image_np_u8.shape[2])
    assert patch_hwc_np_u8.dtype == np.uint8
    return patch_hwc_np_u8


