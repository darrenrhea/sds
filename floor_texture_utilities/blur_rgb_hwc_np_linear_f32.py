import numpy as np
from scipy.ndimage.filters import gaussian_filter


def blur_rgb_hwc_np_linear_f32(
    rgb_hwc_np_linear_f32: np.ndarray,
    sigma_x: float,
    sigma_y: float,
):
    """
    We should not blur unless we are in linear light.
    Floor textures should probably be blurred more in the x direction than the y direction,
    since fast panning between left and right is more common than panning up and down.
    """
    blurred_rgb_linear_f32 = np.zeros(
        shape=(rgb_hwc_np_linear_f32.shape[0], rgb_hwc_np_linear_f32.shape[1], 3),
        dtype=np.float32,
    )

    for i in range(3):
        blurred_rgb_linear_f32[:, :, i] = gaussian_filter(
            rgb_hwc_np_linear_f32[:, :, i],
            sigma=(sigma_x, sigma_y)
        )

    return blurred_rgb_linear_f32

