import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_as_hwc_rgba_np_uint8(image_path: Path) -> np.ndarray:
    """
    Opens an image file path to be RGBA np.uint8 H x W x C=4
    regardless of whether the image is actually RGBA or not,
    i.e. it will add a fully opaque alpha channel if the image
    does not have one.
    """
    pil = PIL.Image.open(str(image_path)).convert("RGBA")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 4
    return image_np_uint8

