import numpy as np


def print_metric_distances_between_hwc_np_u8s(
    a: np.ndarray,
    b: np.ndarray,
):
    assert a.shape == b.shape
    rgb_values = a.reshape(-1, 3).astype(np.float32)
    rgb_values_again = b.reshape(-1, 3).astype(np.float32)
    residuals = np.abs(rgb_values - rgb_values_again)
    L_1_error = np.mean(residuals)
    L_2_error = np.sqrt(np.mean(residuals**2))
    L_infinity_error = np.max(residuals)
    print(f"{L_1_error=}")
    print(f"{L_2_error=}")
    print(f"{L_infinity_error=}")
    