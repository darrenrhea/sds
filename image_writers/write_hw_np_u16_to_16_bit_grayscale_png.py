from prii import (
     prii
)
import textwrap
import numpy as np
from pathlib import Path
import PIL.Image


def write_hw_np_u16_to_16_bit_grayscale_png(
    hw_np_u16: np.ndarray,
    out_abs_file_path: Path,
    verbose: bool = False,
    display_image_in_iterm2: bool = False,
) -> None:
    """
    Saves a 2-dimensional numpy array of float16 values between 0 and 1 to a 16-bit grayscale PNG on disk.

    Given a hw_np_u16
    
    * 2-dimensional numpy array of
    * shape `(height, width)`
    * dtype `uint16`
    * the entries of which can only range in [0, 65535]

    it will be interpreted as a grayscale image with the usual
    row i grows down, column j grows right convention to
    write a 16-bit grayscale PNG to the file path
    `out_abs_file_path`,
    where black=0 and white=65535.
    
    This does not make directories for you,
    and will raise an error if the parent directory of `out_abs_file_path`
    does not exist.

    Returns None.
    """
    assert (
        isinstance(hw_np_u16, np.ndarray)
    ), f"ERROR: hw_np_u16 must be a numpy array, not {type(hw_np_u16)=}"

    assert (
        hw_np_u16.ndim == 2
    ), f"ERROR: hw_np_u16 must have 2 dimensions, but instead {hw_np_u16.ndim=}"

    assert (
        hw_np_u16.dtype == np.uint16
    ), f"hw_np_u16 must have dtype uint16, not {hw_np_u16.dtype}"

    assert (
        out_abs_file_path.parent.is_dir()
    ), f"ERROR: {out_abs_file_path.parent} is not a directory.  We don't make directories for you."

    assert (
        isinstance(out_abs_file_path, Path)
    ), f"out_abs_file_path must a Path, not {type(out_abs_file_path)=}"

    assert (
        out_abs_file_path.suffix == ".png"
    ), f"ERROR: out_abs_file_path must have a the extension .png, but you gave {out_abs_file_path=}"

    assert (
        out_abs_file_path.is_absolute()
    ), f"ERROR: {out_abs_file_path} must be absolute."

    assert (
        out_abs_file_path.parent.is_dir()
    ), f"ERROR: {out_abs_file_path.parent=} is not a directory. We don't make directories for you."

    

    output_pil = PIL.Image.fromarray(hw_np_u16)
    
    output_pil.save(
        fp=out_abs_file_path,
        format="PNG"
    )
    if verbose:
        print(
            textwrap.dedent(
                f"""\
                Saved a 16-bit graycale PNG to:

                    pri {out_abs_file_path}
                """
            )
        )

    if display_image_in_iterm2:
        prii(out_abs_file_path)
    
    return

