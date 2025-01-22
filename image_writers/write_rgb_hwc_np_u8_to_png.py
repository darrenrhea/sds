import numpy as np
from pathlib import Path
import PIL.Image
from print_image_in_iterm2 import print_image_in_iterm2


def write_rgb_hwc_np_u8_to_png(
    rgb_hwc_np_u8: np.ndarray,
    out_abs_file_path: Path,
    verbose: bool = False,
    prii: bool = False,
):
    """
    rgb_hwc_np_u8: a numpy array of shape (height, width, 3) and dtype np.uint8
    """
    assert (
        isinstance(rgb_hwc_np_u8, np.ndarray)
    ), f"rgb_hwc_np_u8 must be a numpy array, not {type(rgb_hwc_np_u8)}"

    assert (
        isinstance(out_abs_file_path, Path)
    ), f"out_abs_file_path must a Path, not {type(out_abs_file_path)=}"

    assert (
        rgb_hwc_np_u8.dtype == np.uint8
    ), f"rgb_hwc_np_u8 must have dtype uint8, not {rgb_hwc_np_u8.dtype}"

    assert (
        rgb_hwc_np_u8.shape[2] == 3
    ), "rgba_hwc_np_u8 must have 3 channels"

    assert (
        out_abs_file_path.suffix == ".png"
    ), f"{out_abs_file_path=} must have a .png suffix"

    output_rgb_pil = PIL.Image.fromarray(rgb_hwc_np_u8)
    
    if verbose:
        print(f"pri {out_abs_file_path}")
    
    if prii:
        print_image_in_iterm2(image_pil=output_rgb_pil)
    
    output_rgb_pil.save(
        fp=out_abs_file_path,
        format="PNG"
    )

    return

