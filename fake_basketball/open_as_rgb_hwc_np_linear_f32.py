from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np
from pathlib import Path


def open_as_rgb_hwc_np_linear_f32(
    image_path: Path
) -> np.ndarray:
    """
    A u8 rgb image has nonlinear values.
    This removes that nonlinearity and scales it into float32s in [0.0, 1.0]
    For the nonlinear-preserving version, see also open_as_rgb_hwc_np_nonlinear_f32
    """
    rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(image_path)
    
    rgb_hwc_np_linear_f32 = convert_u8_to_linear_f32(
        x=rgb_hwc_np_u8
    )

    assert rgb_hwc_np_linear_f32.dtype == np.float32
    assert rgb_hwc_np_linear_f32.ndim == 3
    assert rgb_hwc_np_linear_f32.shape[2] == 3
    return rgb_hwc_np_linear_f32

