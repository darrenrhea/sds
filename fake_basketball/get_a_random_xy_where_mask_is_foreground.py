import numpy as np
from typing import Tuple, Optional


def get_a_random_xy_where_mask_is_foreground(
    mask: np.ndarray
) -> Optional[Tuple[int, int]]:
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    index_array = np.argwhere(mask > 192)
    if index_array.shape[0] == 0:
        return None
    assert index_array.dtype == np.int64, f"ERROR: {index_array.dtype=}"
    index = np.random.randint(0, len(index_array))
    xy = (
        index_array[index][1],
        index_array[index][0]
    )
    return xy
