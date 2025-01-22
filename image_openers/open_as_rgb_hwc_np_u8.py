import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_as_rgb_hwc_np_u8(
    image_path: Path,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Opens an image file path to be RGB np.uint8 H x W x C.
    Any alpha channel present in the image file is completely ignored, but tolerated.
    Grayscale images become RGB triplets where all three channels are equal.
    """
    assert image_path.resolve().is_file(), f"ERROR: {image_path.resolve()} is not an extant file!"
    pil = PIL.Image.open(str(image_path)).convert("RGB")
    if scale != 1.0:
        pil = pil.resize(
            (
                int(pil.width * scale),
                int(pil.height * scale)
            ),
            resample=PIL.Image.Resampling.BILINEAR
        )
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 3
    assert image_np_uint8.dtype == np.uint8
    return image_np_uint8
