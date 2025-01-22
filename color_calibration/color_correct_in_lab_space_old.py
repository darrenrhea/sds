from color_print_json import (
     color_print_json
)

import better_json as bj

from prii import (
     prii
)
from pathlib import Path
import numpy as np

from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)

from skimage.color import rgb2lab, lab2rgb


def color_correct_in_lab_space(
    rgb_hwc_np_u8: np.ndarray,
):
    # hard-coded munich coefficients found by running
    # python show_color_map_for_color_correction_test.py
    LAB_coeffs_matrix = np.array(
        [
            [ 2.02756796e-03,  4.30176628e-01,  1.82335616e+01],
            [-1.40336593e-03,  7.93722835e-01,  8.53416496e+00],
            [-5.57306067e-03,  1.49884184e+00, -1.52387562e+01],
        ]
    )

    rgb_hwc_np_f64 = rgb_hwc_np_u8.astype(np.float64) / 255.0
    lab_hwc_np_f64 = rgb2lab(rgb_hwc_np_f64)

    assert lab_hwc_np_f64.dtype == np.float64
    corrected_lab_hwc_np_f64 = np.zeros(shape=rgb_hwc_np_u8.shape, dtype=np.float64)

    for c in range(3):
        model = np.poly1d(
                LAB_coeffs_matrix[c]
        )
        corrected_lab_hwc_np_f64[:, :, c] = model(lab_hwc_np_f64[:, :, c])
    
    corrected_rgb_hwc_np_f64 = lab2rgb(corrected_lab_hwc_np_f64)
    
    corrected_rgb_hwc_np_u8 = (corrected_rgb_hwc_np_f64 * 255).round().clip(0, 255).astype(np.uint8)

    return corrected_rgb_hwc_np_u8
    

if __name__ == "__main__":
    # todo for loop over all pairs of images in color_correction.json5
    lst = bj.load("~/r/color_calibration/color_correction.json5")

    for dct in lst:
        acceptable = True
        if "footage" not in dct:
            print("This needs a footage key:")
            color_print_json(dct)
            continue
        if "led_video" not in dct:
            print("This needs a led_video key:")
            continue
        if "path" not in dct["footage"]:
            print("This needs a path key in the footage section:")
            continue
        if "path" not in dct["led_video"]:
            print("This needs a path key in the led_video section:")
            continue
        led_video_path = Path(dct["led_video"]["path"]).expanduser()
        footage_path = Path(dct["footage"]["path"]).expanduser()

        led_video_rgb = open_as_rgb_hwc_np_u8(led_video_path)
        

        rgb = open_as_rgb_hwc_np_u8(led_video_path)
        fixed = color_correct_in_lab_space(rgb)
        prii(rgb)
        prii(fixed)
        prii(footage_path)
    