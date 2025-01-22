import numpy as np


def add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32(
    rgb_hwc_np_f32: np.ndarray,
) -> np.ndarray:
    """
    Adds a constantly 1.0 fourth channel to the image.
    """
    assert rgb_hwc_np_f32.dtype == np.float32
    assert rgb_hwc_np_f32.ndim == 3
    assert rgb_hwc_np_f32.shape[2] == 3
       
    rgba_hwc_np_f32 = np.zeros(
        shape=(
            rgb_hwc_np_f32.shape[0],
            rgb_hwc_np_f32.shape[1],
            4
        ),
        dtype=np.float32
    )

    rgba_hwc_np_f32[:, :, :3] = rgb_hwc_np_f32
    rgba_hwc_np_f32[:, :, 3] = 1.0
    
    return rgba_hwc_np_f32

    
