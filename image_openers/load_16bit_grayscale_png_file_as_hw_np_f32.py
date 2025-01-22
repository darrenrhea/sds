import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def load_16bit_grayscale_png_file_as_hw_np_f32(
    image_path: Path,
) -> np.ndarray:
    """
    Opens an image file path to be RGB np.uint8 H x W x C.
    Any alpha channel present in the image file is completely ignored, but tolerated.
    Grayscale images become RGB triplets where all three channels are equal.
    """
    assert image_path.resolve().is_file(), f"ERROR: {image_path.resolve()} is not an extant file!"
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

    hw_np_f32 = hw_np_i32.astype(np.float32) / 65535.0
    
    assert hw_np_f32.ndim == 2
    assert hw_np_f32.shape[0] == 1080
    assert hw_np_f32.shape[1] == 1920
    assert hw_np_f32.dtype == np.float32
    return hw_np_f32
