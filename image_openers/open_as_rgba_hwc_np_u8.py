import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from typing import Optional

def open_as_rgba_hwc_np_u8(image_path: Path) -> Optional[np.ndarray]:
    """
    Opens an image file path to be RGBA np.uint8 H x W x C=4
    regardless of whether the image is actually RGB or RGBA or not,
    i.e. it will add on a fully opaque alpha channel if the image
    does not have one already.
    """
    pil = PIL.Image.open(str(image_path))

    np_u8 = np.array(pil)
    if np_u8.ndim == 2:
        return None
    if np_u8.shape[2] == 3:
        image_np_uint8 = np.zeros((np_u8.shape[0], np_u8.shape[1], 4), dtype=np.uint8)
        image_np_uint8[:, :, :3] = np_u8
        image_np_uint8[:, :, 3] = 255
    else:
        image_np_uint8 = np_u8

    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 4
    return image_np_uint8

