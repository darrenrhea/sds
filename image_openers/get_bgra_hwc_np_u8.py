from pathlib import Path
import cv2
import numpy as np
import PIL.Image

def get_bgra_hwc_np_u8(image_path: Path) -> np.ndarray:
    """
    Open an image with an alpha channel as a 4 channel bgra image.
    """
    assert isinstance(image_path, Path), f"{image_path=} is not a Path"
    assert image_path.is_file(), f"{image_path=} is not a file"
    bgra = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    # rgba = np.array(PIL.Image.open(image_path))
    # bgra = rgba[:, :, [2, 1, 0, 3]]

    assert bgra is not None, f"ERROR: {image_path=} does not exist or is not readable"

    assert isinstance(bgra, np.ndarray), f"ERROR: not a numpy array"
    assert bgra.ndim == 3, f"ERROR: {image_path=} is not 3 dimensional"
    assert bgra.shape[2] == 4, f"ERROR: {image_path=} is not a 4 channel hwc image"
    assert bgra.dtype == np.uint8, f"ERROR: {image_path=} is not a uint8"
    
    return bgra

