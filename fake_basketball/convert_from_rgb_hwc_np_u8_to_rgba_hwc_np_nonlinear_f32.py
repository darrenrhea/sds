from get_a_random_adrip_file_path import (
     get_a_random_adrip_file_path
)
from prii_linear_f32 import (
     prii_linear_f32
)
from prii import (
     prii
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from augment_texture import (
     augment_texture
)
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)


def convert_from_rgb_hwc_np_u8_to_rgba_hwc_np_nonlinear_f32(
    rgb_hwc_np_u8: np.ndarray
) -> np.ndarray:
    """
    Convert a uint8 RGB image to a nonlinear float32 RGBA image
    which is fully opaque.
    """
    assert rgb_hwc_np_u8.dtype == np.uint8
    assert rgb_hwc_np_u8.ndim == 3
    assert rgb_hwc_np_u8.shape[2] == 3

    rgba_hwc_np_nonlinear_f32 = np.zeros(
        (rgb_hwc_np_u8.shape[0], rgb_hwc_np_u8.shape[1], 4),
        dtype=np.float32
    )

    rgba_hwc_np_nonlinear_f32[:, :, :3] = rgb_hwc_np_u8.astype(np.float32) / 255.0
    rgba_hwc_np_nonlinear_f32[:, :, 3] = 1.0

    assert rgba_hwc_np_nonlinear_f32.dtype == np.float32
    assert rgba_hwc_np_nonlinear_f32.ndim == 3
    assert rgba_hwc_np_nonlinear_f32.shape[2] == 4

    return rgba_hwc_np_nonlinear_f32

