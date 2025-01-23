import cv2
import numpy as np
from pathlib import Path


def open_just_the_rgb_part_of_image(
    image_path: Path
) -> np.ndarray:
    """
    Test this function via:
    python test_open_just_the_rgb_part_of_image.py
    """
    assert isinstance(image_path, Path)
    assert image_path.is_file(), f"ERROR: {image_path} does not exist"

    assert (
        image_path.name.endswith('.png')
        or
        image_path.name.endswith("jpg")
    ), f"ERROR: {image_path} does not end with .jpg nor .png"

    # the -1 means load the image as is, with alpha channel if it exists, grayscale if that is the case, etc.
    raw_frame = cv2.imread(str(image_path), -1)

    if raw_frame is None:
        raise Exception('error loading frame')
    
    assert raw_frame.ndim == 3, "ERROR: frame must be 3 dimensional"
    assert raw_frame.dtype == np.uint8, "ERROR: frame must be uint8 for now"

    if raw_frame.shape[2] == 4:
        print(f'WARNING: removing alpha channel from {image_path}')
        raw_frame = raw_frame[:, :, :3]

    image_hwc_rgb_np_u8 = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    
    assert image_hwc_rgb_np_u8.dtype == np.uint8
    assert image_hwc_rgb_np_u8.ndim == 3
    assert image_hwc_rgb_np_u8.shape[2] == 3
    return image_hwc_rgb_np_u8
