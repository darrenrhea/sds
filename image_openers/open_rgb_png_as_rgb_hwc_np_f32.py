from open_rgb_png_as_rgb_hwc_np_u16 import (
     open_rgb_png_as_rgb_hwc_np_u16
)
import cv2
import numpy as np
from pathlib import Path


def open_rgb_png_as_rgb_hwc_np_f32(
    original_path: Path,
) -> np.ndarray:
    """
    I.e. the minimum value is 0 and the maximum value is 65535.
    """


    rgb_hwc_np_u16 = open_rgb_png_as_rgb_hwc_np_u16(
        original_path=original_path,
    )
    assert rgb_hwc_np_u16.shape[2] == 3
    assert rgb_hwc_np_u16.dtype == np.uint16
    rgb_hwc_np_f32 = rgb_hwc_np_u16.astype(np.float32) / 65535.0
    return rgb_hwc_np_f32
