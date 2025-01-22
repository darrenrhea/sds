import PIL.Image
from prii import (
     prii
)
import numpy as np


def prii_hw_np_nonlinear_u16(
    hw_np_nonlinear_u16: np.ndarray,
    caption=None,
    out=None,
):
    """
    If you have a hwc np.float32 image with values in [0, 1] in the naive / nonlinear sense,
    this function will convert it to a hwc np.uint8 image and print it.
    """
    assert isinstance(hw_np_nonlinear_u16, np.ndarray)
    assert hw_np_nonlinear_u16.ndim == 2
   
    image_pil = PIL.Image.fromarray(
        hw_np_nonlinear_u16
    )
    
    prii(image_pil)