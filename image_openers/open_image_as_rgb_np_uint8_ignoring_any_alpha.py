"""
Opening image files has a lot more nuance that people are ready to admit. It defines:
* open_as_hwc_rgb_np_uint8
* open_as_hwc_rgba_np_uint8
"""
import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_image_as_rgb_np_uint8_ignoring_any_alpha(abs_file_path: Path) -> np.ndarray:
    """
    Opens an image that contains color for the color part,
    ignoring any alpha
    """
    attempt = np.asarray(PIL.Image.open(abs_file_path))
    assert attempt.ndim == 3
    assert (
        attempt.shape[2] == 3 or attempt.shape[2] == 4
    ), f"{abs_file_path} need to either have 3 or 4 channels to get out the rgb part"
    ans = attempt[:, :, :3].copy()
    return ans