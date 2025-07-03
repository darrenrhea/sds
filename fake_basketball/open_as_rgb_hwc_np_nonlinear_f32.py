from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np
from pathlib import Path


def open_as_rgb_hwc_np_nonlinear_f32(
    image_path: Path
) -> np.ndarray:
    """
    TODO: this should be able to handle 16-bit PNG images invisibly.
    A u8 rgb image, for instance jpg, has nonlinear values.
    This keeps that nonlinearity, but scales it into float32s in [0.0, 1.0]
    See also open_as_rgb_hwc_np_linear_f32
    """
    rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(image_path)
    rgb_hwc_np_nonlinear_f32 = rgb_hwc_np_u8.astype(np.float32) / 255.0

    assert rgb_hwc_np_nonlinear_f32.dtype == np.float32
    assert rgb_hwc_np_nonlinear_f32.ndim == 3
    assert rgb_hwc_np_nonlinear_f32.shape[2] == 3
    
    return rgb_hwc_np_nonlinear_f32

