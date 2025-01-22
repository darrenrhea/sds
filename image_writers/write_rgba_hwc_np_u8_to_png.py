import numpy as np
from pathlib import Path
import PIL.Image
from typing import Optional


def write_rgba_hwc_np_u8_to_png(
    rgba_hwc_np_u8: np.ndarray,
    out_abs_file_path: Optional[Path],
    verbose: bool = False
):
    """
    rgb_hwc_np_u8: numpy array of shape (height, width, 3) and dtype uint8
    alpha_hw_np_u8: numpy array of shape (height, width) and dtype uint8
    Returns the combined RGBA image as a numpy array of shape (height, width, 4) and dtype uint8
    """
    assert (
        isinstance(rgba_hwc_np_u8, np.ndarray)
    ), f"rgba_hwc_np_u8 must be a numpy array, not {type(rgba_hwc_np_u8)}"

    assert (
        out_abs_file_path is None or isinstance(out_abs_file_path, Path)
    ), f"out_abs_file_path must be None or a Path, not {out_abs_file_path}"

    assert (
        rgba_hwc_np_u8.dtype == np.uint8
    ), f"rgba_hwc_np_u8 must have dtype uint8, not {rgba_hwc_np_u8.dtype}"

    assert (
        rgba_hwc_np_u8.shape[2] == 4
    ), f"rgba_hwc_np_u8 must have 4 channels but is has shape {rgba_hwc_np_u8.shape}"
    

    output_rgba_pil = PIL.Image.fromarray(rgba_hwc_np_u8)
     
    output_rgba_pil.save(
        fp=out_abs_file_path,
        format="PNG"
    )

    if verbose:
        print(f"Wrote {out_abs_file_path}")

    return

