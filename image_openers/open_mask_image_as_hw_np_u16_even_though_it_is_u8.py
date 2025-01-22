from open_mask_image import (
     open_mask_image
)
import numpy as np
from pathlib import Path


def open_mask_image_as_hw_np_u16_even_though_it_is_u8(
    mask_path: Path
) -> np.ndarray:
    """
    We want to open a png which is either grayscale,
    or it is rgba (4 channels) and we only want the alpha channel.
    because some people want to represent masks as files that also contain rgb info,
    but others just use a grayscale image to represent the mask.
    """
    mask_hw_np_u8 = open_mask_image(mask_path=mask_path)
    assert mask_hw_np_u8.ndim == 2
    assert mask_hw_np_u8.dtype == np.uint8
    
    mask_hw_np_u16 = mask_hw_np_u8.astype(np.uint16) * 255
    assert mask_hw_np_u16.ndim == 2
    assert mask_hw_np_u16.dtype == np.uint16
    return mask_hw_np_u16
