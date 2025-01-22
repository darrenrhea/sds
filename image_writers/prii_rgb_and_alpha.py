import numpy as np
import PIL.Image
from print_image_in_iterm2 import print_image_in_iterm2


def prii_rgb_and_alpha(
    rgb_hwc_np_u8: np.ndarray,
    alpha_hw_np_u8: np.ndarray, 
) -> None:
    """
    rgb_hwc_np_u8: numpy array of shape (height, width, 3) and dtype uint8
    alpha_hw_np_u8: numpy array of shape (height, width) and dtype uint8
    Returns the combined RGBA image as a numpy array of shape (height, width, 4) and dtype uint8
    """
    assert (
        isinstance(rgb_hwc_np_u8, np.ndarray)
    ), f"rgb_hwc_np_u8 must be a numpy array, not {type(rgb_hwc_np_u8)}"

    assert (
        isinstance(alpha_hw_np_u8, np.ndarray)
    ), f"alpha_hw_np_u8 must be a numpy array, not {type(alpha_hw_np_u8)}"

    assert (
        rgb_hwc_np_u8.dtype == np.uint8
    ), f"rgb_hwc_np_u8 must have dtype uint8, not {rgb_hwc_np_u8.dtype}"

    assert (
        alpha_hw_np_u8.dtype == np.uint8
    ), f"alpha_hw_np_u8 must have dtype uint8, not {alpha_hw_np_u8.dtype}"

    assert (
        rgb_hwc_np_u8.shape[:2] == alpha_hw_np_u8.shape
    ), f"ERROR: expected rgb_hwc_np_u8 and alpha_hw_np_u8 to have the same height and width, but they have {rgb_hwc_np_u8.shape=} and {alpha_hw_np_u8.shape=}"
    
    assert (
        rgb_hwc_np_u8.shape[2] == 3
    ), f"rgb_hwc_np_u8 must have 3 channels, but it has {rgb_hwc_np_u8.shape=}"
    
    output_rgba_np_u8 = np.zeros(
        shape=(
            rgb_hwc_np_u8.shape[0],
            rgb_hwc_np_u8.shape[1],
            4
        ),
        dtype=np.uint8
    )
    output_rgba_np_u8[:, :, :3] = rgb_hwc_np_u8
    output_rgba_np_u8[:, :, 3] = alpha_hw_np_u8
    
    output_rgba_pil = PIL.Image.fromarray(output_rgba_np_u8)
    
    print_image_in_iterm2(image_pil=output_rgba_pil)
    
