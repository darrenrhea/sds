
import numpy as np


def gather_colors_on_indicator(
    rgb_or_rgba_np_u8: np.ndarray,
    indicator: np.ndarray
):
    """
    For the purpose of color correction, we need to gather the colors in a subregion of the image
    specified by indicator.
    """
    assert rgb_or_rgba_np_u8.ndim == 3
    assert rgb_or_rgba_np_u8.dtype == np.uint8
    assert rgb_or_rgba_np_u8.shape[2] in (3, 4)
    assert indicator.ndim == 2
    assert indicator.shape == rgb_or_rgba_np_u8.shape[:2]

    ijs = np.argwhere(indicator)
    colors = rgb_or_rgba_np_u8[ijs[:, 0], ijs[:, 1], :3]
    return colors

