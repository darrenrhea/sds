from open_rgb_png_as_rgb_hwc_np_u16 import (
     open_rgb_png_as_rgb_hwc_np_u16
)
from print_green import (
     print_green
)
import cv2
import numpy as np
from pathlib import Path


def test_open_rgb_png_as_rgb_hwc_np_u16_1():
    
    original_path = Path(
        "/hd2/clips/bal_game2_bigzoom/frames/bal_game2_bigzoom_000000_original.png"
    )

    rgb_hwc_np_u16 = open_rgb_png_as_rgb_hwc_np_u16(
        original_path=original_path
    )
    assert rgb_hwc_np_u16.shape == (1080, 1920, 3)
    assert rgb_hwc_np_u16.dtype == np.uint16
    print_green(f"Maximum value = {rgb_hwc_np_u16.max()}")
    print_green(f"Minimum value = {rgb_hwc_np_u16.min()}")


if __name__ == "__main__":
    test_open_rgb_png_as_rgb_hwc_np_u16_1()