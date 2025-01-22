import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_as_hwc_rgb_np_uint8(image_path: Path) -> np.ndarray:
    """
    New name: open_as_rgb_hwc_np_u8.
    This is now just a redirect:
    """
    pil = PIL.Image.open(str(image_path)).convert("RGB")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 3
    return image_np_uint8
