from prii_nonlinear_f32 import (
     prii_nonlinear_f32
)
from open_rgb_png_as_rgb_hwc_np_f32 import (
     open_rgb_png_as_rgb_hwc_np_f32
)
from print_green import (
     print_green
)
import cv2
import numpy as np
from pathlib import Path


def test_open_rgb_png_as_rgb_hwc_np_f32_1():
    
    original_path = Path(
        "/hd2/clips/bal_game2_bigzoom/frames/bal_game2_bigzoom_000000_original.png"
    )

    rgb_hwc_np_f32 = open_rgb_png_as_rgb_hwc_np_f32(
        original_path=original_path
    )
    assert rgb_hwc_np_f32.shape == (1080, 1920, 3)
    assert rgb_hwc_np_f32.dtype == np.float32
    print_green(f"Maximum value = {rgb_hwc_np_f32.max()}")
    print_green(f"Minimum value = {rgb_hwc_np_f32.min()}")
    prii_nonlinear_f32(
        x=rgb_hwc_np_f32,
        caption=None,
        out=None,
    )

def test_open_rgb_png_as_rgb_hwc_np_f32_2():
    
    png_path = Path(
        "/hd2/clips/bal_game2_bigzoom/frames/bal_game2_bigzoom_000000_original.png"
    )
    jpg_path = Path(
        "/hd2/clips/bal_game2_bigzoom/frames/bal_game2_bigzoom_000000_original.jpg"
    )

    png_rgb_hwc_np_f32 = open_rgb_png_as_rgb_hwc_np_f32(
        original_path=png_path
    )
    assert png_rgb_hwc_np_f32.shape == (1080, 1920, 3)
    assert png_rgb_hwc_np_f32.dtype == np.float32
    print_green(f"Maximum value = {png_rgb_hwc_np_f32.max()}")
    print_green(f"Minimum value = {png_rgb_hwc_np_f32.min()}")
    prii_nonlinear_f32(
        x=png_rgb_hwc_np_f32,
        caption=None,
        out=None,
    )
   
    jpg_bgr_u8 = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
    assert jpg_bgr_u8.shape == (1080, 1920, 3)
    assert jpg_bgr_u8.dtype == np.uint8
    jpg_rgb_u8 = cv2.cvtColor(jpg_bgr_u8, cv2.COLOR_BGR2RGB)
    jpg_rgb_hwc_np_f32 = jpg_rgb_u8.astype(np.float32) / 255.0


    prii_nonlinear_f32(
        x=jpg_rgb_hwc_np_f32,
        caption=None,
        out=None,
    )



if __name__ == "__main__":
    # test_open_rgb_png_as_rgb_hwc_np_f32_1()
    test_open_rgb_png_as_rgb_hwc_np_f32_2()