import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_a_grayscale_png_barfing_if_it_is_not_grayscale(image_path: Path):
    """
    Opens a PNG image file path if and only if it is "grayscale"
    i.e. a single channel (not RGB, not RGBA).
    """
    pil = PIL.Image.open(str(image_path))
    assert pil.mode == "L", (
        f"Barfing because the file {image_path} is not a grayscale PNG. "
        f"Confirm for yourself via:\n\n    exiftool {image_path}"
    )
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 2
    assert image_np_uint8.dtype == np.uint8
    return image_np_uint8
