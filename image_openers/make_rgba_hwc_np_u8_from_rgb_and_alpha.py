import numpy as np


def make_rgba_hwc_np_u8_from_rgb_and_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Stack the rgb and alpha channels to create an rgba image.
    """
    assert rgb.dtype == np.uint8
    assert alpha.dtype == np.uint8
    assert rgb.ndim == 3
    assert alpha.ndim == 2
    assert rgb.shape[0:2] == alpha.shape[0:2]
    assert rgb.shape[2] == 3
    rgba = np.zeros(
        shape=(
            alpha.shape[0],
            alpha.shape[1],
            4
        ),
        dtype=np.uint8
    )
    rgba[:, :, 0:3] = rgb
    rgba[:, :, 3] = alpha

    return rgba