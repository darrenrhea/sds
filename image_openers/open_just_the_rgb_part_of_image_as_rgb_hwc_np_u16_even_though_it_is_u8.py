from open_just_the_rgb_part_of_image import (
     open_just_the_rgb_part_of_image
)
import numpy as np
from pathlib import Path


def open_just_the_rgb_part_of_image_as_rgb_hwc_np_u16_even_though_it_is_u8(
    image_path: Path
) -> np.ndarray:
    """
    TODO:
    Make a version of this that opens the images RGB as 16 bits per channel regardless of whether it is 8 or 16 bits per channel.

    This should open any image file that has RGB color information
    in it at 8 bits per channel, but then converts it to range from 0 to 65535.

    TODO: this fails on
    anything that is actually more that 8 bits per channel,
    such as 16 bit PNGs, JPEG2000, etc.
    """
    rgb_hwc_np_u8 = open_just_the_rgb_part_of_image(
        image_path=image_path
    )
    assert rgb_hwc_np_u8.dtype == np.uint8
    assert rgb_hwc_np_u8.ndim == 3
    assert rgb_hwc_np_u8.shape[2] == 3

    rgb_hwc_np_u16 = rgb_hwc_np_u8.astype(np.uint16) * 256

    assert rgb_hwc_np_u16.dtype == np.uint16
    assert rgb_hwc_np_u16.ndim == 3
    assert rgb_hwc_np_u16.shape[2] == 3
    return rgb_hwc_np_u16
