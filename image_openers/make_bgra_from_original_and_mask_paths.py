from open_alpha_channel_image_as_a_single_channel_grayscale_image import open_alpha_channel_image_as_a_single_channel_grayscale_image
import cv2
import numpy as np
from pathlib import Path


def make_bgra_from_original_and_mask_paths(
        original_path: Path,
        mask_path: Path,
        flip_mask: bool,
        quantize: bool
) -> np.ndarray:
    """
    This is a helper function to make discrete black hole people style
    bgra images from original and mask paths.

    You probably want both flip_mask and quantize to be True
    considering that people are usually white against a black background,
    and scorebug or other forbidden things are usually white as well.
    """
    original_bgr = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
    assert original_bgr.shape[2] == 3, f"{original_path=} is not a 3 channel image"

    mask = open_alpha_channel_image_as_a_single_channel_grayscale_image(mask_path)
    assert mask.ndim == 2
    assert mask.shape[0] == original_bgr.shape[0]
    assert mask.shape[1] == original_bgr.shape[1]

    if flip_mask:
        mask = 255 - mask

    if quantize:
        mask = (mask > 127).astype(np.uint8) * 255
    bgra = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    bgra[:, :, 0:3] = original_bgr
    bgra[:, :, 3] = mask

    return bgra