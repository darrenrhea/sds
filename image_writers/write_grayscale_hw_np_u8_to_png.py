
import numpy as np
from pathlib import Path
import PIL.Image
from print_image_in_iterm2 import print_image_in_iterm2


def write_grayscale_hw_np_u8_to_png(
    grayscale_hw_np_u8: np.ndarray,
    out_abs_file_path: Path,
    verbose: bool = False
) -> None:
    """
    grayscale_hw_np_u8: numpy array of shape (height, width) and dtype uint8.
    """
    assert (
        isinstance(grayscale_hw_np_u8, np.ndarray)
    ), f"grayscale_hw_np_u8 must be a numpy array, not {type(grayscale_hw_np_u8)}"

   
    assert out_abs_file_path.parent.is_dir(), f"ERROR: {out_abs_file_path.parent} is not a directory.  We don't make directories for you."

    assert (
        grayscale_hw_np_u8.dtype == np.uint8
    ), f"grayscale_hw_np_u8 must have dtype uint8, not {grayscale_hw_np_u8.dtype}"

    assert (
        grayscale_hw_np_u8.ndim == 2
    ), f"ERROR: grayscale_hw_np_u8 must have 2 dimensions but it has {grayscale_hw_np_u8.ndim=}"
    
    assert (
        isinstance(out_abs_file_path, Path)
    ), f"out_abs_file_path must a Path, not {type(out_abs_file_path)=}"

    assert (
        out_abs_file_path.suffix == ".png"
    ), f"ERROR: out_abs_file_path must have a .png extension, but you gave {out_abs_file_path=}"

    assert (
        out_abs_file_path.is_absolute()
    ), f"ERROR: {out_abs_file_path} must be absolute."

    assert (
        out_abs_file_path.parent.is_dir()
    ), f"ERROR: {out_abs_file_path.parent=} is not a directory. We don't make directories for you."

    output_pil = PIL.Image.fromarray(grayscale_hw_np_u8)
    
    if verbose:
        print_image_in_iterm2(image_pil=output_pil)
    
    output_pil.save(
        fp=out_abs_file_path,
        format="PNG"
    )

    return

