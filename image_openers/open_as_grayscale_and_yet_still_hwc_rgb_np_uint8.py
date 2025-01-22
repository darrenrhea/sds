import PIL.Image
import numpy as np
from pathlib import Path


def open_as_grayscale_and_yet_still_hwc_rgb_np_uint8(
    image_path: Path
) -> np.ndarray:
    """
    Opens an image file path to be grayscale and yet still 3 channeled RGB np.uint8 H x W x C

    We will be starting off with a grayscale variant of the image, but
    then we plan on mutating it to have colorful parts, so it needs to have
    all three channels R G and B, despite that those
    3 channels contain identical data listed three times over.
    """
    pil = PIL.Image.open(image_path)
    pil = pil.convert("Y").convert("RGB")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 3
    assert image_np_uint8.dtype == np.uint8
    return image_np_uint8