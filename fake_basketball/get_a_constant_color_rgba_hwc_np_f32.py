import numpy as np


def get_a_constant_color_rgba_hwc_np_f32(
    color,
    width: int,
    height: int,
):
    M = np.zeros(
        shape=(height, width, 4),
        dtype=np.float32,
    )
    M[...] = color
    return M
