import numpy as np
from scipy.ndimage.filters import gaussian_filter


def blur(
    np_hw_u8: np.ndarray,
    r: float
) -> np.ndarray:
    """
    blur an image
    """
    blurred = gaussian_filter(
        np_hw_u8,
        sigma=r
    )
    return blurred


