import numpy as np
import PIL.Image


def make_rgb_hwc_np_u8_from_rgba_hwc_np_u8(
    rgba_hwc_np_u8: np.ndarray
) -> np.ndarray:
    """
    Convert an RGBA image to an RGB image by using the alpha channel as an alpha channel.
    """
    assert isinstance(rgba_hwc_np_u8, np.ndarray), f"Expected np.ndarray, but got {type(rgba_hwc_np_u8)}"
    assert rgba_hwc_np_u8.ndim == 3, f"Expected 3 dimensions, but got {rgba_hwc_np_u8.ndim}"
    assert rgba_hwc_np_u8.dtype == np.uint8, f"Expected np.uint8, but got {rgba_hwc_np_u8.dtype}"
    assert rgba_hwc_np_u8.shape[2] == 4, f"Expected 3rd dimension to be 4, but got {rgba_hwc_np_u8.shape[2]}"
    image_pil = PIL.Image.fromarray(rgba_hwc_np_u8)
    background = PIL.Image.new("RGB", image_pil.size, (0, 0, 0))
    background.paste(image_pil, mask=image_pil.split()[3]) # 3 is the alpha channel
    rgb_hwc_np_u8 = np.array(background)
    return rgb_hwc_np_u8
