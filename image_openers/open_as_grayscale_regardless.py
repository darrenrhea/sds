import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_as_grayscale_regardless(image_path: Path):
    """
    There are images that are truly grayscale, and there are images that look a
    lot like grayscale, but are actually RGB(a) images where there are R, G, and B channels, and maybe an alpha channel.
    This function opens the image as grayscale regardless of whether it is truly grayscale or not.
    """
    pil = PIL.Image.open(image_path)
    if pil.mode == "L":
        gray_np_u8 = np.array(pil)
       
        return gray_np_u8
    else:
        image_np_uint8 = np.array(pil)
        assert image_np_uint8.ndim == 3
        assert image_np_uint8.dtype == np.uint8
        assert image_np_uint8.shape[2] == 3 or image_np_uint8.shape[2] == 4
        image_np_uint8 = image_np_uint8[:, :, :3]
        gray_np_u8 = np.mean(image_np_uint8, axis=2).astype(np.uint8)

    assert gray_np_u8.ndim == 2
    assert gray_np_u8.dtype == np.uint8
    return gray_np_u8
