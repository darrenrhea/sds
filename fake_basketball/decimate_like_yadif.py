import numpy as np


def decimate_like_yadif(
    rgb_hwc_np_u8: np.ndarray,
    parity=None
):
    height = rgb_hwc_np_u8.shape[0]

    if parity is None:
        parity = np.random.randint(0, 2)

    field = rgb_hwc_np_u8[parity::2, :, :]
    field_float = field.astype(np.float32)
    avg = (field_float[1:, :, :] + field_float[:-1, :, :]) / 2.0
    
    avg_u8 = np.clip(np.round(avg), 0, 255).astype(np.uint8)
    
    yadif = np.zeros_like(rgb_hwc_np_u8)
    yadif[parity::2, :, :] = field
    if parity == 0:
        yadif[-1, :, :] = field[-1, :, :]
        yadif[1:(height-2):2, :, :] = avg_u8

    else:
        yadif[0, :, :] = field[0, :, :]
        yadif[2:(height-1):2, :, :] = avg_u8

    return yadif
