from convert_nonlinear_f32_to_u8 import (
     convert_nonlinear_f32_to_u8
)
from prii import (
     prii
)
import numpy as np


def prii_rgb_chw_alexnet_f32(
    x: np.ndarray,
    caption=None,
    out=None,
):
    """
    If you have a hwc np.float32 image with values in [0, 1] in the naive / nonlinear sense,
    this function will convert it to a hwc np.uint8 image and print it.
    """
    hwc_alexnet = x.transpose(1, 2, 0)  # Convert from CHW to HWC
    alexnet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    alexnet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    hwc_in01 = hwc_alexnet * alexnet_std + alexnet_mean
    np_u8 = convert_nonlinear_f32_to_u8(hwc_in01)
    prii(np_u8, out=out, caption=caption)