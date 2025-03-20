
import numpy as np
from typing import List, Optional
from open_sha256_as_rgba_hwc_np_u8 import open_sha256_as_rgba_hwc_np_u8


def open_sha256s_as_rgba_hwc_np_u8s(
    sha256s: List[str]
) -> List[Optional[np.ndarray]]:
    """
    Opens a list of sha256 as (RGBA np.uint8 H x W x C=4)s
    """
   
    return [
        open_sha256_as_rgba_hwc_np_u8(sha256)
        for sha256 in sha256s
    ]
