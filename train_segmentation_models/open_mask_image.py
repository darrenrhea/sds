import cv2
import numpy as np
from pathlib import Path


def open_mask_image(mask_path: Path) -> np.ndarray:
    """
    Test this function via:
    python test_open_mask_image.py
    or pytest.
    We want to open a png which is either grayscale,
    or it is rgba (4 channels) and we only want the alpha channel.
    because some people want to represent masks as files that also contain rgb info,
    but others just use a grayscale image to represent the mask.
    """
    assert isinstance(mask_path, Path)
    assert mask_path.is_file()
    assert mask_path.name.endswith('.png'), f"ERROR: {mask_path} does not end with .png"

    maybe_bgra = cv2.imread(str(mask_path), -1)

    if maybe_bgra is None:
        raise Exception(f"error loading mask {mask_path}")
    
    assert maybe_bgra.dtype == np.uint8, "ERROR: all mask files must be uint8 for now"

    if  maybe_bgra.ndim == 3 and maybe_bgra.shape[2] == 4:
        mask = maybe_bgra[:, :, 3].copy()
    elif maybe_bgra.ndim == 2:
        mask = maybe_bgra
    else:
        raise Exception("The mask must be either grayscale or rgba (4 channels)")
    
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    return mask
