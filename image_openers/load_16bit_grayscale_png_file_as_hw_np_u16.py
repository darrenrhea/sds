import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def load_16bit_grayscale_png_file_as_hw_np_u16(
    image_path: Path,
) -> np.ndarray:
    """
    TODO: make another function that loads just the alpha mask if the input is a 16-bit RGBA PNG,
    or the whole thing if the input is a 16-bit grayscale PNG.
    
    Designed initially to open depth-maps.
    Opens an 16-bit grayscale PNG image_path to be np.uint16.
    """
    assert image_path.resolve().is_file(), f"ERROR: {image_path.resolve()} is not an extant file!"
    assert image_path.suffix == ".png", f"ERROR: {image_path} must have the extension .png"
    pil = PIL.Image.open(str(image_path))
    hw_np_i32 = np.array(pil)
    assert hw_np_i32.ndim == 2
    assert hw_np_i32.shape[0] == 1080
    assert hw_np_i32.shape[1] == 1920
    assert hw_np_i32.dtype == np.int32
    m = np.min(hw_np_i32)
    M = np.max(hw_np_i32)
    
    assert m >= 0
    assert M <= 65535

    hw_np_u16 = hw_np_i32.astype(np.uint16)
    
    assert hw_np_u16.ndim == 2
    assert hw_np_u16.shape[0] == 1080
    assert hw_np_u16.shape[1] == 1920
    assert hw_np_u16.dtype == np.uint16
    return hw_np_u16
