
from get_file_path_of_sha256 import get_file_path_of_sha256
import numpy as np
from typing import Optional
from open_as_rgba_hwc_np_u8 import open_as_rgba_hwc_np_u8


def open_sha256_as_rgba_hwc_np_u8(
    sha256: str
) -> Optional[np.ndarray]:
    """
    Opens a sha256 as an RGBA np.uint8 H x W x C=4
    """
    
    local_file_path = get_file_path_of_sha256(
        sha256=sha256,
        check=True,
    )

    rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(local_file_path)
    return rgba_hwc_np_u8
