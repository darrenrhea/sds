from pathlib import Path
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np


def get_metric_distance_between_two_image_paths(
    a_path: Path,
    b_path: Path,
) -> float:
    

    a = open_as_rgb_hwc_np_u8(
        image_path=a_path,
    )

    b = open_as_rgb_hwc_np_u8(
        image_path=b_path,
    )
    assert a.shape == b.shape, f"ERROR: {a.shape=} not the same shape as {b.shape=}"
    rgb_values = a.reshape(-1, 3).astype(np.float32)
    rgb_values_again = b.reshape(-1, 3).astype(np.float32)
    residuals = np.abs(rgb_values - rgb_values_again)
    L_1_error = np.mean(residuals)
    L_2_error = np.sqrt(np.mean(residuals**2))
    L_infinity_error = np.max(residuals)
    # print(f"{L_1_error=}")
    # print(f"{L_2_error=}")
    # print(f"{L_infinity_error=}")
    return L_2_error
    