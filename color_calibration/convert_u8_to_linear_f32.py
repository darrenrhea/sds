import numpy as np


def convert_u8_to_linear_f32(
    x: np.ndarray
) -> np.ndarray:
    """
    Convert a uint8 image to a linear float32 image.
    """
    assert x.dtype == np.uint8
    assert x.ndim == 3
    assert x.shape[2] in [3, 4]

    rgb = (x[:, :, :3].astype(np.float32) / 255.0) ** 2.2
    
    if x.shape[2] == 3:
        ans = rgb
    else:
        ans = np.dstack(
            (
                rgb,
                x[:, :, 3].astype(np.float32) / 255.0
            )
        )
    assert ans.dtype == np.float32
    return ans