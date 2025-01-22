import numpy as np


def convert_linear_f32_to_u8(x: np.ndarray) -> np.ndarray:
    """
    Convert an rgb or rgba linear float32 image back to the standard uint8 format.
    Linear means that it is appropriate to add or average colors in this space.
    If there is an alpha channel, it is assumed to be in the last channel.
    """
    # assert np.max(x) <= 1.0, f"ERROR: {np.max(x)=}, but at most it should be 1.0"
    # assert np.min(x) >= 0.0, f"ERROR: {np.min(x)=}, but is should be at least 0.0"
    assert x.dtype == np.float32
    assert x.shape[2] in [3, 4], f"x has shape {x.shape}, but it should have 3 or 4 channels"
    rgb_linear_f32 = x[:, :, :3]
    rgb_u8 = np.round(rgb_linear_f32.clip(0.0, 1.0) ** (1.0 / 2.2) * 255.0).clip(0, 255).astype(np.uint8)
    if x.shape[2] == 3:
        return rgb_u8
    else:
        alpha_u8 = np.round(x[:, :, 3] * 255.0).clip(0, 255).astype(np.uint8)
        return np.dstack(
            (rgb_u8, alpha_u8)
        )
    