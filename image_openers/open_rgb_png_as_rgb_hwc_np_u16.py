import cv2
import numpy as np
from pathlib import Path


def open_rgb_png_as_rgb_hwc_np_u16(
    original_path: Path,
) -> np.ndarray:
    """
    I.e. the minimum value is 0 and the maximum value is 65535.
    """
    assert isinstance(original_path, Path), f"{original_path=} is not a Path"
    assert original_path.exists(), f"{original_path=} does not exist"
    assert original_path.suffix == ".png", f"{original_path=} is not a png file"
    assert original_path.is_file(), f"{original_path=} is not a file"

    original_bgr = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
    original_rgb_unknown_bit_depth = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    assert (
        original_rgb_unknown_bit_depth.shape[2] == 3
    ), f"{original_path=} is not a 3 channel image"
    
    assert original_rgb_unknown_bit_depth.dtype == np.uint16

    return original_rgb_unknown_bit_depth
