import numpy as np


def add_noise_to_render(rgba_hwc_np_u8):
    """
    We need to make realistically distributed fake data,
    but this is not it, at least not without more work.
    Why not f32 all the way?
    The floating points ranging over [0.0, 255.0] is confusing.
    """
    assert False, "ERROR: add_noise_to_render seems half-baked.  Why are you calling it?"
    rgba_hwc_np_float32 = rgba_hwc_np_u8.astype(np.float32)
    
    scale = np.random.uniform(
        low=0.0,
        high=1.0
    )
    noise = np.random.normal(
        loc=0.0,
        scale=scale,
        size=(
            rgba_hwc_np_float32.shape[0],
            rgba_hwc_np_float32.shape[1],
            3
        )
    ).astype(np.float32)

    rgba_hwc_np_float32[:, :, :3] += noise
    rgba_hwc_np_u8[:, :, :] = rgba_hwc_np_float32.round().clip(0.0, 255.0).astype(np.uint8)

   
    
