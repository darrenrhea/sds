from use_polynomial_regression_coefficients import (
     use_polynomial_regression_coefficients
)

import numpy as np


def color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients(
    degree: int,
    coefficients: np.ndarray,
    rgb_hwc_np_u8: np.ndarray
) -> np.ndarray:
    """
    Given a maximum total degree and polynomial coefficients,
    and an RGB hwc uint8 image, usually an LED ad texture they gave us,
    this corrects the color of the image and returns the corrected image.
    """
    assert rgb_hwc_np_u8.dtype == np.uint8
    assert rgb_hwc_np_u8.ndim == 3
    assert rgb_hwc_np_u8.shape[2] == 3

    h, w = rgb_hwc_np_u8.shape[:2]
    inputs = rgb_hwc_np_u8.reshape(-1, 3)
   
    corrected = use_polynomial_regression_coefficients(
        degree=degree,
        coefficients=coefficients,
        input_vectors=inputs.astype(np.float64) / 255.0
    )

    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    reshaped = corrected.reshape(h, w, 3)
    return reshaped

