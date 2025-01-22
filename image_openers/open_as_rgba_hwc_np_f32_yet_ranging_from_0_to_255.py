import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_as_rgba_hwc_np_f32_yet_ranging_from_0_to_255(
    image_path: Path
) -> np.ndarray:
    """
    Opens an image file path to be a
    numpy array of dtype=np.float32
    H x W x C
    where C = 4 and the channels are in the order of RGBA.

    regardless of whether the image is actually RGBA or not,
    i.e. it will add a fully opaque alpha channel if the image file
    does not have an alpha channel.
    """
    pil = PIL.Image.open(str(image_path)).convert("RGBA")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 4
    image_rgba_hwc_np_f32 = image_np_uint8.astype(np.float32)
    return image_rgba_hwc_np_f32

