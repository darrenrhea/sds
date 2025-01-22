import numpy as np
from pathlib import Path
import PIL.Image
from print_image_in_iterm2 import print_image_in_iterm2


def write_rgb_and_alpha_to_png(
    rgb_hwc_np_u8: np.ndarray,
    alpha_hw_np_u8: np.ndarray, 
    out_abs_file_path: Path,
    verbose: bool = False,
    display_image_in_terminal: bool = False,
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
        out_abs_file_path is None or isinstance(out_abs_file_path, Path)
    ), f"out_abs_file_path must be None or a Path, not {out_abs_file_path}"

    assert (
        rgb_hwc_np_u8.dtype == np.uint8
    ), f"rgb_hwc_np_u8 must have dtype uint8, not {rgb_hwc_np_u8.dtype}"

    assert (
        alpha_hw_np_u8.dtype == np.uint8
    ), f"alpha_hw_np_u8 must have dtype uint8, not {alpha_hw_np_u8.dtype}"

    assert (
        rgb_hwc_np_u8.shape[:2] == alpha_hw_np_u8.shape
    ), "ERROR: expected rgb_hwc_np_u8 and alpha_hw_np_u8 to have the same height and width"
    
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
    
    if verbose:
        print(f"pri {out_abs_file_path}")
    
    if display_image_in_terminal:
        print_image_in_iterm2(image_pil=output_rgba_pil)
    
    output_rgba_pil.save(
        fp=out_abs_file_path,
        format="PNG"
    )
    
    return None
