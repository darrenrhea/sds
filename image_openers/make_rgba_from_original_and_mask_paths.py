from open_alpha_channel_image_as_a_single_channel_grayscale_image import open_alpha_channel_image_as_a_single_channel_grayscale_image
import cv2
import numpy as np
from pathlib import Path


def make_rgba_from_original_and_mask_paths(
    original_path: Path,
    mask_path: Path,
    flip_mask: bool,
    quantize: bool
) -> np.ndarray:
    """
    WARNING: will crush 16 bit pngs down to 8 bit pngs.
    TODO: change name to indicate that behavior.
    TODO: isolate the 16 bit png crushing to its own procedure.
    This is a helper function to make discrete black hole people style
    rgba images from original and mask paths.

    You probably want both flip_mask and quantize to be True
    considering that people are usually white against a black background,
    and scorebug or other forbidden things are usually white as well.
    
    https://stackoverflow.com/questions/11337499/how-to-convert-an-image-from-np-uint16-to-np-uint8
    """
    original_bgr = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
    original_rgb_unknown_bit_depth = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    assert (
        original_rgb_unknown_bit_depth.shape[2] == 3
    ), f"{original_path=} is not a 3 channel image"
    
    if original_rgb_unknown_bit_depth.dtype == np.uint16:
        original_rgb_hwc_np_u8 = cv2.convertScaleAbs(original_rgb_unknown_bit_depth, alpha=(255.0/65535.0))
    elif original_rgb_unknown_bit_depth.dtype == np.uint8:
        original_rgb_hwc_np_u8 = original_rgb_unknown_bit_depth
    else:
        raise ValueError(f"{original_path=} is not a np.uint8 or np.uint16 image? it is {original_rgb_unknown_bit_depth.dtype}")

    assert (
        original_rgb_hwc_np_u8.dtype == np.uint8
    ), f"{original_path=} is not a np.uint8 image, it is {original_rgb_hwc_np_u8.dtype}"

    mask = open_alpha_channel_image_as_a_single_channel_grayscale_image(mask_path)
    assert mask.ndim == 2
    assert mask.shape[0] == original_rgb_hwc_np_u8.shape[0]
    assert mask.shape[1] == original_rgb_hwc_np_u8.shape[1]

    if flip_mask:
        mask = 255 - mask

    if quantize:
        mask = (mask > 127).astype(np.uint8) * 255
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0:3] = original_rgb_hwc_np_u8
    rgba[:, :, 3] = mask

    return rgba