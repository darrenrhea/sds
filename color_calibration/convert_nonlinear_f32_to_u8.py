import numpy as np


def convert_nonlinear_f32_to_u8(x: np.ndarray) -> np.ndarray:
    """
    Convert an rgb or rgba nonlinear float32 image back to the standard uint8 format.
    Nonlinear means that it is just the usual values [0, 255] normalized into [0, 1].
    If there is an alpha channel, it is assumed to be in the last channel.
    """
    # assert np.max(x) <= 1.0, f"ERROR: {np.max(x)=}, but at most it should be 1.0"
    # assert np.min(x) >= 0.0, f"ERROR: {np.min(x)=}, but is should be at least 0.0"
    assert x.dtype == np.float32
    if x.ndim == 2:
        return np.round(x[:, :] * 255.0).clip(0, 255).astype(np.uint8)
    
    assert x.ndim == 3
    assert x.shape[2] in [1, 3, 4], f"x has shape {x.shape}, but it should have 3 or 4 channels"
   

    rgb_nonlinear_f32 = x[:, :, :3]
    rgb_u8 = np.round(rgb_nonlinear_f32.clip(0.0, 1.0) * 255.0).clip(0, 255).astype(np.uint8)
    if x.shape[2] == 3:
        return rgb_u8
    else:
        alpha_u8 = np.round(x[:, :, 3] * 255.0).clip(0, 255).astype(np.uint8)
        return np.dstack(
            (rgb_u8, alpha_u8)
        )
    