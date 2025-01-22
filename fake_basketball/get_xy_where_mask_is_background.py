import numpy as np
from typing import Tuple, Optional


def get_xy_where_mask_is_background(
    mask: np.ndarray
) -> Optional[Tuple[int, int]]:
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    index_array = np.argwhere(mask < 64)
    if index_array.shape[0] == 0:
        return None
    assert index_array.dtype == np.int64, f"ERROR: {index_array.dtype=}"
    index = np.random.randint(0, len(index_array))
    xy = (
        int(index_array[index][1]),
        int(index_array[index][0]),
    )
    assert isinstance(xy[0], int)
    assert isinstance(xy[1], int)
    assert isinstance(xy, tuple)
    return xy
