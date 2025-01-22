from prii import (
     prii
)
import textwrap
import numpy as np
from pathlib import Path
import PIL.Image


def write_hw_np_f32_to_16_bit_grayscale_png(
    hw_np_f32: np.ndarray,
    out_abs_file_path: Path,
    verbose: bool = False,
    display_image_in_iterm2: bool = False,
) -> None:
    """
    Saves a 2-dimensional numpy array of float32 values between 0 and 1 to a 16-bit grayscale PNG on disk.

    Given `hw_np_f32`,
    
    * 2-dimensional numpy array of
    * shape `(height, width)`
    * dtype `float32`
    * the entries of which range in `[0, 1]`
    * (values outside of this range will be clipped to 0 or 1)

    it will be interpreted as a grayscale image with the usual
    row i grows down, column j grows right convention to
    write a 16-bit grayscale PNG to the file path
    `out_abs_file_path`,
    where 0.0 becomes black=0 and and 1.0 becomes white=65535.
    
    This does not make directories for you,
    and will raise an error if the parent directory of `out_abs_file_path`
    does not exist.

    Returns None.
    """
    assert (
        isinstance(hw_np_f32, np.ndarray)
    ), f"ERROR: hw_np_f32 must be a numpy array, not {type(hw_np_f32)=}"

    assert (
        hw_np_f32.ndim == 2
    ), f"ERROR: hw_np_f32 must have 2 dimensions, but instead {hw_np_f32.ndim=}"

    assert (
        hw_np_f32.dtype == np.float32
    ), f"hw_np_f32 must have dtype float32, not {hw_np_f32.dtype}"

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

    hw_np_u16 = np.round(
        hw_np_f32 * 65535.0
    ).clip(0, 65535).astype(np.uint16)

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

