from convert_nonlinear_f32_to_u8 import (
     convert_nonlinear_f32_to_u8
)
from prii import (
     prii
)
import numpy as np


def prii_rescaled_nonlinear_f32(
    x: np.ndarray,
    caption=None,
    out=None,
):
    """
    If you have a hwc np.float32 image with values in [0, 1] in the naive / nonlinear sense,
    this function will convert it to a hwc np.uint8 image and print it.
    """
    assert x.dtype == np.float32
    m = np.min(x)
    M = np.max(x)
    if m == M:
        x = np.zeros_like(x, dtype=np.float32)
    else:
        x = (x - m) / (M - m)
    np_u8 = convert_nonlinear_f32_to_u8(x)
    prii(np_u8, out=out, caption=caption)