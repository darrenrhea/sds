import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_as_rgba_hwc_np_f32_all_channels_ranging_from_0_to_1(
    image_path: Path
) -> np.ndarray:
    """
    Opens an image file path to be a
    numpy array of dtype=np.float32
    H x W x C
    where C = 4 and the channels are in the order of RGBA,

    regardless of whether the image is actually RGBA or merely RGB,
    i.e. it will add a fully opaque alpha channel if the image file
    does not have an alpha channel.
    All 4 channels R, G, B, and A will be in the range [0, 1].
    """
    pil = PIL.Image.open(str(image_path)).convert("RGBA")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 4
    image_rgba_hwc_np_f32 = image_np_uint8.astype(np.float32)
    image_rgba_hwc_np_f32 /= 255.0
    return image_rgba_hwc_np_f32

