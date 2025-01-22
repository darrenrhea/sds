"""
Functions for writing images on disk.
* write_rgb_hwc_np_u8_to_png
* write_rgb_and_alpha_to_png
* write_rgba_hwc_np_u8_to_png
"""
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from write_rgb_and_alpha_to_png import (
     write_rgb_and_alpha_to_png
)

# prepare for people to do: from image_writers import *:
__all__ = [
    "write_rgb_hwc_np_u8_to_png",
    "write_rgb_and_alpha_to_png",
    "write_rgba_hwc_np_u8_to_png",
    "write_grayscale_hw_np_u8_to_png"
]
