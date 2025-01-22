from use_polynomial_regression_coefficients import (
     use_polynomial_regression_coefficients
)

import numpy as np


def color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
    degree: int,
    coefficients: np.ndarray,
    rgb_hwc_np_linear_f32: np.ndarray
) -> np.ndarray:
    """
    Given a maximum total degree and polynomial coefficients,
    and an RGB hwc uint8 image, usually an LED ad texture they gave us,
    this corrects the color of the image and returns the corrected image.
    """
    assert rgb_hwc_np_linear_f32.dtype == np.float32
    assert rgb_hwc_np_linear_f32.ndim == 3
    assert rgb_hwc_np_linear_f32.shape[2] == 3

    h, w = rgb_hwc_np_linear_f32.shape[:2]
    input_vectors = rgb_hwc_np_linear_f32.reshape(-1, 3)
   
    corrected = use_polynomial_regression_coefficients(
        degree=degree,
        coefficients=coefficients.astype(np.float32),
        input_vectors=input_vectors,
    )

    corrected = np.clip(corrected, 0.0, 1.0)

    reshaped = corrected.reshape(h, w, 3)
    return reshaped

